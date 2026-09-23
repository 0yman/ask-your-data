"""Who is asking: one workspace per visitor when the app is public.

Run on your own computer, the app has one user and one workspace - the same
behaviour it always had, with the dataset choice remembered across restarts.
Run as a public demo (``AGENT_PUBLIC_MODE=true``), every visitor gets their own:
their uploaded tables, their choice of dataset, their key if they add one. None
of it is visible to anyone else, and none of it outlives an hour of inactivity.

A visitor is recognised by a random token the page sends in a header rather
than by a cookie. A Hugging Face Space is shown inside an iframe on another
domain, and browsers increasingly refuse cookies there.
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
from .config import Settings
from .datasets import table_count

logger = logging.getLogger(__name__)

DATASETS = ("sample", "retail", "co2", "mine")
_TOKEN_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


@dataclass(eq=False)
class Session:
    id: str
    settings: Settings      # this workspace's copy: its own files, maybe its own key
    dataset: str = "sample"
    agent: PortAnalystAgent | None = None
    error: str | None = None
    own_key: bool = False
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
            with self.local.lock:
                self.local.open(self._load_choice())

    # --- local mode: the choice of dataset survives a restart -------------

    def _load_choice(self) -> str:
        try:
            choice = json.loads(self.settings.state_path.read_text(encoding="utf-8")).get("dataset")
        except (OSError, ValueError):
            return "sample"
        return choice if choice in DATASETS else "sample"

    def save_choice(self, session: Session) -> None:
        if self.public:
            return
        path = self.settings.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dataset": session.dataset}), encoding="utf-8")

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
        with session.lock:
            session.open("sample")
        self._sessions[sid] = session
        return session

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
        self.per_day = per_day
        self._clock = clock
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        self._day = ""
        self._today = 0
        self._lock = threading.Lock()

    def _roll(self, now: float) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if day != self._day:
            self._day, self._today = day, 0
            self._recent = defaultdict(deque, {k: v for k, v in self._recent.items() if v})

    def _prune(self, address: str, now: float) -> deque[float]:
        recent = self._recent[address]
        while recent and now - recent[0] > 3600:
            recent.popleft()
        return recent

    def take(self, address: str) -> None:
        with self._lock:
            now = self._clock()
            self._roll(now)
            if self._today >= self.per_day:
                raise QuotaExceeded(
                    "Today's free demo questions have all been used. Add your own free "
                    "Gemini key with the key button at the top to keep going, or come back tomorrow."
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
            self._today += 1

    def left(self, address: str) -> int:
        with self._lock:
            now = self._clock()
            self._roll(now)
            used = len(self._prune(address, now))
            return max(0, min(self.per_hour - used, self.per_day - self._today))
