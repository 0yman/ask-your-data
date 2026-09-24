"""Who is asking: one workspace per visitor when the app is public.

Run on your own computer, the app has one user and one workspace - the same
behaviour it always had, with the dataset choice remembered across restarts.
Run as a public demo (``AGENT_PUBLIC_MODE=true``), every visitor gets their own:
their uploaded tables, their choice of dataset, their key if they add one. None
of it is visible to anyone else, and none of it outlives an hour of inactivity.

A visitor is recognised by a random token the page sends in a header rather
than by a cookie, so the demo also works embedded in another site - inside an
iframe on another domain, where browsers increasingly refuse cookies.
"""

from __future__ import annotations

import json
import logging
import secrets
import shutil
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .agent import PortAnalystAgent, build_agent
from .config import ModelOption, Settings
from .datasets import table_count

logger = logging.getLogger(__name__)

DATASETS = ("sample", "retail", "co2", "mine")
# The picker's id for Gemini, which is configured outside MODEL_CATALOG.
GEMINI = "gemini"
_TOKEN_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


@dataclass(eq=False)
class Session:
    id: str
    settings: Settings      # this workspace's copy: its own files, maybe its own key
    dataset: str = "sample"
    agent: PortAnalystAgent | None = None
    error: str | None = None
    own_key: bool = False
    # Which model answers: a MODEL_CATALOG id, GEMINI, or None for the single
    # model configured the old way (no catalog keys set).
    model_id: str | None = None
    # A visitor's own Gemini key, kept for this visit only so they can switch
    # away from it and back.
    visitor_key: str | None = None
    # Set when a visitor's token had expired and this workspace replaced it.
    # The page still shows the old one, so the first question is refused
    # rather than answered against data the visitor is not looking at.
    replaced_stale: bool = False
    last_seen: float = field(default_factory=time.monotonic)
    # One connection per workspace: switching datasets or importing a file
    # mid-question would otherwise pull it out from under a running query.
    lock: threading.RLock = field(default_factory=threading.RLock)

    def close_agent(self) -> None:
        if self.agent is not None:
            self.agent.warehouse.close()
        self.agent = None

    def open(self, dataset: str) -> None:
        """Point the agent at a dataset. Call holding `lock`."""
        self.close_agent()
        self.dataset = dataset
        self.error = None
        if dataset == "mine" and table_count(self.settings.user_db_path) == 0:
            return  # nothing uploaded yet: a normal state, not an error
        try:
            self.agent = build_agent(self.settings, dataset=dataset)
        except Exception as exc:
            logger.warning("Could not open the %s dataset: %s", dataset, exc)
            self.error = str(exc)

    def use_settings(self, settings: Settings, own_key: bool) -> None:
        with self.lock:
            self.settings = settings
            self.own_key = own_key
            self.open(self.dataset)

    def use_model(self, option: ModelOption) -> None:
        """Answer with a model from the catalog, on the server's key."""
        with self.lock:
            self.model_id = option.id
            self.use_settings(self.settings.with_model(option), own_key=False)

    def use_gemini(self, key: str, visitors_own: bool) -> None:
        """Answer with Gemini: the visitor's own key, or the server's on a
        computer where the key is the user's anyway."""
        with self.lock:
            if visitors_own:
                self.visitor_key = key
            self.model_id = GEMINI
            self.use_settings(
                self.settings.model_copy(update={"llm_backend": "gemini", "google_api_key": key}),
                own_key=visitors_own,
            )


class SessionStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.public = settings.public_mode
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.local: Session | None = None
        if self.public:
            # Workspaces live in memory, so files left by a previous run
            # belong to no one.
            shutil.rmtree(settings.sessions_dir, ignore_errors=True)
            settings.sessions_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.local = Session(id="local", settings=settings)
            dataset, model = self._load_choice()
            self._start_model(self.local, model)
            with self.local.lock:
                self.local.open(dataset)

    # --- local mode: the choice of dataset survives a restart -------------

    def _load_choice(self) -> tuple[str, str | None]:
        try:
            state = json.loads(self.settings.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "sample", None
        dataset = state.get("dataset")
        return (dataset if dataset in DATASETS else "sample"), state.get("model")

    def save_choice(self, session: Session) -> None:
        if self.public:
            return
        path = self.settings.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dataset": session.dataset, "model": session.model_id}), encoding="utf-8")

    # --- which workspace a request belongs to -----------------------------

    def get(self, token: str | None) -> Session:
        if not self.public:
            assert self.local is not None
            return self.local
        token = token if looks_like_token(token) else None
        now = time.monotonic()
        with self._lock:
            doomed = self._take_idle(now)
            session = self._sessions.get(token or "")
            if session is None:
                if len(self._sessions) >= self.settings.max_sessions:
                    oldest = min(self._sessions.values(), key=lambda s: s.last_seen)
                    doomed.append(self._sessions.pop(oldest.id))
                session = self._create(stale=bool(token))
            session.last_seen = now
        # Outside the store's lock: waiting for a workspace's own lock here
        # would hold up every other visitor's request.
        for old in doomed:
            self._cleanup(old)
        return session

    def _create(self, stale: bool) -> Session:
        sid = secrets.token_urlsafe(24)
        folder = self.settings.sessions_dir / sid
        session = Session(
            id=sid,
            settings=self.settings.model_copy(update={
                "user_db_path": folder / "my_data.duckdb",
                "uploads_dir": folder / "uploads",
            }),
            replaced_stale=stale,
        )
        self._start_model(session, None)
        with session.lock:
            session.open("sample")
        self._sessions[sid] = session
        return session

    def _start_model(self, session: Session, wanted: str | None) -> None:
        """A new workspace's model: the one asked for if it is still offered,
        else the default. Settings only - the caller opens the dataset."""
        settings = self.settings
        if wanted == GEMINI and self.gemini_offered(session):
            session.model_id = GEMINI
            session.settings = session.settings.model_copy(update={"llm_backend": "gemini"})
            return
        option = settings.model_option(wanted) or settings.default_model_option()
        if option is not None:
            session.model_id = option.id
            session.settings = session.settings.with_model(option)
        elif settings.llm_backend == "gemini" and self.gemini_offered(session):
            session.model_id = GEMINI

    def gemini_offered(self, session: Session) -> bool:
        """On a public server only with the visitor's own key: the server's
        Gemini key is not shared. On your own computer, whenever there is one."""
        if self.public:
            return bool(session.visitor_key)
        return bool(self.settings.google_api_key)

    def _take_idle(self, now: float) -> list[Session]:
        idle = self.settings.session_idle_minutes * 60
        gone = [s for s in self._sessions.values() if now - s.last_seen > idle]
        for session in gone:
            del self._sessions[session.id]
        return gone

    @staticmethod
    def _cleanup(session: Session) -> None:
        # A question still running in it finishes first; the files go after.
        with session.lock:
            session.close_agent()
            shutil.rmtree(session.settings.user_db_path.parent, ignore_errors=True)

    def __len__(self) -> int:
        return len(self._sessions) if self.public else 1

    def close_all(self) -> None:
        if self.local is not None:
            with self.local.lock:
                self.local.close_agent()
        with self._lock:
            sessions, self._sessions = list(self._sessions.values()), {}
        for session in sessions:
            self._cleanup(session)


def looks_like_token(value: str | None) -> bool:
    return bool(value) and len(value) <= 64 and set(value) <= _TOKEN_CHARS


class QuotaExceeded(RuntimeError):
    """The message is shown to the visitor as-is."""


class Quota:
    """Rations questions answered on the server's key.

    Per address per hour, so one visitor cannot use up the demo, and per day
    in total, so the free tier's daily allowance is never the thing that
    fails. Visitors who add their own key are not counted: they pay with it.
    """

    def __init__(self, per_hour: int, per_day: int, clock=time.time) -> None:
        self.per_hour = per_hour
        self.per_day = per_day  # for a model without a cap of its own
        self._clock = clock
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        self._day = ""
        # Per model: each host's free tier has its own daily allowance.
        self._today: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def _roll(self, now: float) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if day != self._day:
            self._day, self._today = day, defaultdict(int)
            self._recent = defaultdict(deque, {k: v for k, v in self._recent.items() if v})

    def _prune(self, address: str, now: float) -> deque[float]:
        recent = self._recent[address]
        while recent and now - recent[0] > 3600:
            recent.popleft()
        return recent

    def take(self, address: str, model: str = "default", per_day: int | None = None, label: str = "") -> None:
        cap = per_day or self.per_day
        with self._lock:
            now = self._clock()
            self._roll(now)
            if self._today[model] >= cap:
                which = f" for {label}" if label else ""
                raise QuotaExceeded(
                    f"Today's free demo questions{which} have all been used. Pick another model, "
                    "add your own free Gemini key with the key button at the top, or come back tomorrow."
                )
            recent = self._prune(address, now)
            if len(recent) >= self.per_hour:
                minutes = max(1, int((3600 - (now - recent[0])) // 60) + 1)
                raise QuotaExceeded(
                    f"You have asked {self.per_hour} questions in the last hour, the demo's limit. "
                    f"Try again in about {minutes} minutes, or add your own free Gemini key "
                    "with the key button at the top."
                )
            recent.append(now)
            self._today[model] += 1

    def left(self, address: str, model: str = "default", per_day: int | None = None) -> int:
        cap = per_day or self.per_day
        with self._lock:
            now = self._clock()
            self._roll(now)
            used = len(self._prune(address, now))
            return max(0, min(self.per_hour - used, cap - self._today[model]))
