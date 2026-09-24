"""FastAPI service: the web app at `/`, and the JSON API behind it.

Several datasets can be active: the sample port warehouse, two real public
datasets, or tables built from the user's own CSV and Excel files. One is
shown at a time - an agent looking at all of them would have to guess which
tables a question is about.

`/ask` returns the full trace alongside the answer. For an agent that is the
difference between a product and a magic box: the caller can see which tools
ran, which SQL executed, what failed and was corrected, and the rows behind
the figures.

The same app runs on one person's computer or as a public demo; `sessions.py`
describes what changes between the two.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import logging
import queue
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import PortAnalystAgent
from .config import ENV_FILE, ModelOption, Settings, get_settings
from .datasets import (
    SUPPORTED_SUFFIXES,
    DatasetError,
    drop_table,
    import_file,
    table_name_for,
    table_names,
)
from .envfile import set_env_value
from .guardrails import UnsafeSQLError
from .llm import ModelLimitReached
from .sessions import GEMINI, Quota, QuotaExceeded, Session, SessionStore
from .warehouse import QueryResult, QueryTimeout, set_memory_limit

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
RESULT_ROW_LIMIT = 200
SESSION_HEADER = "X-Session"
# How long a question waits for a free model slot on a busy public server.
SLOT_WAIT_SECONDS = 90

Dataset = Literal["sample", "retail", "co2", "mine"]

_store: SessionStore | None = None
_quota: Quota | None = None
_slots: threading.BoundedSemaphore | None = None
# Per catalog model: how many questions each host's free tier can work on at
# once. Anything else shares `_slots`.
_model_slots: dict[str, threading.BoundedSemaphore] = {}
# Catalog models whose host said their allowance is used up, and until when
# (time.monotonic). The picker says so instead of showing a quota that is
# only this app's count.
_resting: dict[str, float] = {}
# When the host does not say how long, assume this.
DEFAULT_REST_SECONDS = 15 * 60


def _rest_left(model_id: str) -> int:
    """Seconds until a resting model is worth trying again (0 = not resting)."""
    until = _resting.get(model_id, 0.0)
    left = until - time.monotonic()
    if left <= 0:
        _resting.pop(model_id, None)
        return 0
    return int(left)


# --- state ------------------------------------------------------------------


def _settings() -> Settings:
    """The app-wide settings. A visitor's own key lives in their session."""
    assert _store is not None
    return _store.settings


def current_session(request: Request) -> Session:
    assert _store is not None
    session = _store.get(request.headers.get(SESSION_HEADER))
    if _store.public:
        request.state.session_id = session.id
    return session


def _refuse_if_replaced(session: Session) -> None:
    """A visitor whose workspace expired is still looking at the old one. Say
    so, rather than answer against data they are not looking at."""
    if session.replaced_stale:
        session.replaced_stale = False
        raise HTTPException(
            status_code=409,
            detail="This page sat idle for over an hour, so its workspace was cleared. "
            "The page has been reset - pick your data and ask again.",
        )


def get_agent(session: Session) -> PortAnalystAgent:
    if session.agent is None:
        if session.dataset == "mine":
            raise HTTPException(
                status_code=409,
                detail="You have not added any files yet. Upload a CSV or Excel file first.",
            )
        raise HTTPException(
            status_code=503,
            detail=f"The {session.dataset} data could not be opened: {session.error or 'unknown error'}",
        )
    return session.agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _quota, _slots
    settings = get_settings()
    set_memory_limit(settings.duckdb_memory_limit)
    _store = SessionStore(settings)
    _quota = Quota(settings.public_questions_per_hour, settings.public_questions_per_day)
    _slots = threading.BoundedSemaphore(max(1, settings.public_concurrent_questions))
    _model_slots.clear()
    _resting.clear()
    _model_slots.update({
        option.id: threading.BoundedSemaphore(max(1, option.concurrent))
        for option in settings.available_models()
    })
    if settings.public_mode:
        logger.info("Public mode: one private workspace per visitor.")
    yield
    _store.close_all()


app = FastAPI(
    title="Ask your data",
    version="2.1.0",
    description=(
        "Ask questions about a database in plain English. A tool-calling agent "
        "writes read-only SQL, runs it through guardrails, and shows its work."
    ),
    lifespan=lifespan,
)

try:  # optional: /metrics for Prometheus scraping
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, include_in_schema=False)
except ImportError:  # pragma: no cover
    pass


@app.middleware("http")
async def session_header(request: Request, call_next):
    """Tell the page which workspace it is using - on errors too, so a page
    whose workspace expired learns the new one either way."""
    response = await call_next(request)
    session_id = getattr(request.state, "session_id", None)
    if session_id:
        response.headers[SESSION_HEADER] = session_id
    return response


# --- helpers ----------------------------------------------------------------


def _cell(value: Any) -> Any:
    """Make a DuckDB value JSON-safe without losing what it said."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if value == value else None  # NaN has no JSON form
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()
    return str(value)


def _table_json(result: QueryResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    rows = result.rows[:RESULT_ROW_LIMIT]
    return {
        "columns": result.columns,
        "rows": [[_cell(v) for v in row] for row in rows],
        "row_count": result.row_count,
        "shown": len(rows),
    }


_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_filename(name: str) -> str:
    base = Path(name or "").name
    stem, suffix = Path(base).stem, Path(base).suffix.lower()
    stem = _UNSAFE_CHARS.sub("_", stem).strip(" ._") or "data"
    return f"{stem[:120]}{suffix}"


def _is_local(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in {"127.0.0.1", "::1", "localhost", "testclient"}


def _client_address(request: Request) -> str:
    """Who to count a question against. Behind the host's proxy the socket
    address is the proxy's, so the forwarded one is used when present. It can
    be forged, which is why the daily total, not this, is the hard limit."""
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


def _engine_label(settings: Settings) -> str:
    if settings.llm_backend == "openai":
        model = settings.openai_model.split("/")[-1]
        for marker, host in (("groq.com", "Groq"), ("cerebras.ai", "Cerebras")):
            if marker in settings.openai_base_url:
                return f"{host} · {model}"
        return model if settings.openai_base_url else f"OpenAI · {model}"
    if settings.llm_backend == "gemini":
        return "Google Gemini"
    return settings.llm_backend


# --- models -----------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class ToolCallOut(BaseModel):
    name: str
    arguments: dict[str, Any]
    ok: bool = True
    error_kind: str | None = None


class StepOut(BaseModel):
    index: int
    text: str | None
    tool_calls: list[ToolCallOut]
    latency_ms: float


class AskResponse(BaseModel):
    question: str
    answer: str
    succeeded: bool
    stop_reason: str
    sql_queries: list[str]
    steps: list[StepOut]
    failed_attempts: int
    total_latency_ms: float
    usage: dict[str, int]
    result: dict[str, Any] | None = None
    model: str | None = None     # which model answered, as the picker names it
    # A question split into parts: each part's question, answer, SQL and rows.
    parts: list[dict[str, Any]] = []
    # The verifier's verdict: {"ok", "problems", "redo", "guidance"}.
    review: dict[str, Any] | None = None


class SQLRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=20000)


class DatasetRequest(BaseModel):
    dataset: Dataset


class ModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=40)


class KeyRequest(BaseModel):
    key: str = Field(min_length=10, max_length=300)


class UploadResult(BaseModel):
    filename: str
    status: str            # "added" or "skipped"
    table: str | None = None
    rows: int = 0
    columns: int = 0
    reason: str | None = None


# --- pages ------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# --- status -----------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    settings = _settings()
    if _store.public:
        return {
            "status": "ok" if settings.db_path.exists() else "down",
            "public": True,
            "llm_ready": settings.has_model_key(),
            "workspaces": len(_store),
        }
    session = _store.local
    return {
        "status": "ok" if session.agent or session.dataset == "mine" else "down",
        "dataset": session.dataset,
        "warehouse_ready": session.agent is not None,
        "llm_ready": settings.has_model_key(),
        "error": session.error,
    }


def _option(session: Session) -> ModelOption | None:
    return _settings().model_option(session.model_id)


def _metered(session: Session) -> bool:
    return _store.public and not session.own_key


def _models(session: Session, request: Request) -> list[dict[str, Any]]:
    """What the model picker offers this visitor."""
    address = _client_address(request)
    out = [
        {
            "id": option.id, "name": option.name, "host": option.host, "note": option.note,
            "questions_left": _quota.left(address, option.id, option.per_day) if _store.public else None,
            "resting_minutes": -(-_rest_left(option.id) // 60) or None,
        }
        for option in _settings().available_models()
    ]
    if _store.gemini_offered(session):
        own = _store.public
        out.append({
            "id": GEMINI, "name": "Gemini", "host": "your key" if own else "Google",
            "note": "Your own Google key, for this visit only." if own else "Your Gemini key from .env.",
            "questions_left": None,
            "resting_minutes": None,
        })
    return out


def _engine(session: Session) -> str:
    option = _option(session)
    if option is not None:
        return option.label
    if session.model_id == GEMINI:
        return "Gemini · your key" if session.own_key else "Gemini · Google"
    return _engine_label(session.settings)


@app.get("/status")
def status(request: Request, session: Session = Depends(current_session)) -> dict[str, Any]:
    settings = session.settings
    public = _store.public
    option = _option(session)
    return {
        "has_key": settings.has_model_key(),
        "engine_label": _engine(session),
        "models": _models(session, request),
        "model": session.model_id,
        "dataset": session.dataset,
        "sample_available": settings.db_path.exists(),
        "examples": settings.available_examples(),
        "my_tables": len(table_names(settings.user_db_path)),
        "max_upload_mb": settings.upload_limit_mb(),
        "accepted_types": sorted(SUPPORTED_SUFFIXES),
        "public": public,
        "own_key": session.own_key,
        "max_tables": settings.public_max_tables if public else None,
        "questions_left": (
            _quota.left(_client_address(request), option.id if option else "default",
                        option.per_day if option else None)
            if _metered(session) else None
        ),
        "session_idle_minutes": settings.session_idle_minutes if public else None,
    }


@app.get("/tables")
def tables(session: Session = Depends(current_session)) -> list[dict[str, Any]]:
    """The active dataset's tables, with columns - what the sidebar lists."""
    with session.lock:
        agent = session.agent
        if agent is None:
            return []
        out = []
        for table in agent.warehouse.list_tables():
            info = agent.warehouse.describe_table(table.name)
            out.append({
                "name": info.name,
                "kind": info.kind if session.dataset == "sample" else "table",
                "rows": info.row_count,
                "columns": [{"name": c.name, "type": c.type} for c in info.columns],
            })
        return out


@app.get("/schema")
def schema(session: Session = Depends(current_session)) -> dict[str, Any]:
    return {"tables": tables(session)}


# --- choosing and managing data ----------------------------------------------


@app.post("/dataset")
def choose_dataset(body: DatasetRequest, session: Session = Depends(current_session)) -> dict[str, Any]:
    if body.dataset in ("retail", "co2") and body.dataset not in session.settings.available_examples():
        raise HTTPException(status_code=404, detail="That dataset is not installed here.")
    session.replaced_stale = False  # the page is choosing afresh
    with session.lock:
        session.open(body.dataset)
        _store.save_choice(session)
    return {"dataset": body.dataset}


@app.post("/model")
def choose_model(body: ModelRequest, session: Session = Depends(current_session)) -> dict[str, Any]:
    if body.model == GEMINI:
        if not _store.gemini_offered(session):
            raise HTTPException(status_code=404, detail="Add your own Gemini key first.")
        key = session.visitor_key if _store.public else _settings().google_api_key
        session.use_gemini(key, visitors_own=_store.public)
    else:
        option = _settings().model_option(body.model)
        if option is None:
            raise HTTPException(status_code=404, detail="That model is not available here.")
        session.use_model(option)
    _store.save_choice(session)
    return {"model": session.model_id, "engine_label": _engine(session)}


@app.post("/data", response_model=list[UploadResult])
def upload_data(
    files: list[UploadFile] = File(...), session: Session = Depends(current_session)
) -> list[UploadResult]:
    settings = session.settings
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    limit_mb = settings.upload_limit_mb()
    max_bytes = limit_mb * 1024 * 1024
    max_tables = settings.public_max_tables if _store.public else None
    results: list[UploadResult] = []
    session.replaced_stale = False

    with session.lock:
        # DuckDB will not open one file read-only and read-write at once, so
        # the agent lets go of it while the tables are written.
        session.close_agent()
        try:
            existing = set(table_names(settings.user_db_path))
            for upload in files:
                name = safe_filename(upload.filename or "")
                if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                    results.append(UploadResult(
                        filename=upload.filename or name, status="skipped",
                        reason="unsupported type - use CSV, TSV or Excel (.xlsx)",
                    ))
                    continue
                full = max_tables is not None and len(existing) >= max_tables
                if full and table_name_for(name) not in existing:
                    results.append(UploadResult(
                        filename=name, status="skipped",
                        reason=f"the demo holds {max_tables} tables at a time - remove one first",
                    ))
                    continue
                destination = settings.uploads_dir / name
                try:
                    written = 0
                    with destination.open("wb") as handle:
                        while chunk := upload.file.read(1024 * 1024):
                            written += len(chunk)
                            if written > max_bytes:
                                raise DatasetError(f"larger than the {limit_mb} MB limit")
                            handle.write(chunk)
                    imported = import_file(destination, settings.user_db_path)
                except DatasetError as exc:
                    results.append(UploadResult(filename=name, status="skipped", reason=str(exc)))
                    continue
                finally:
                    # The table holds the data now; the raw file is not needed.
                    destination.unlink(missing_ok=True)
                existing.add(imported.table)
                results.append(UploadResult(
                    filename=name, status="added", table=imported.table,
                    rows=imported.rows, columns=imported.columns,
                ))
        finally:
            added = any(r.status == "added" for r in results)
            session.open("mine" if added else session.dataset)
            _store.save_choice(session)
    return results


@app.delete("/data/{table}")
def delete_table(table: str, session: Session = Depends(current_session)) -> dict[str, Any]:
    _refuse_if_replaced(session)
    with session.lock:
        session.close_agent()
        try:
            removed = drop_table(table, session.settings.user_db_path)
        finally:
            session.open(session.dataset)
    if not removed:
        raise HTTPException(status_code=404, detail="No such table in your data.")
    return {"removed": table}


# --- the API key --------------------------------------------------------------


@app.post("/settings/key")
def save_key(body: KeyRequest, request: Request, session: Session = Depends(current_session)) -> dict[str, Any]:
    """Use a Gemini key given on the web page.

    On your own computer it is saved into .env - but only from this computer:
    a key-writing endpoint reachable from the network would let anyone on it
    swap in their own key. (Docker users set the key in .env or the
    environment instead.)

    On the public demo it is kept in the visitor's own workspace, in memory,
    and used for their questions only. Nothing is written anywhere.
    """
    if not _store.public and not _is_local(request):
        raise HTTPException(status_code=403, detail="Keys can only be set from this computer.")

    key = body.key.strip()
    # API keys never contain whitespace. Refusing it here also means a pasted
    # value can never carry a newline into .env and write a second setting.
    if any(ch.isspace() for ch in key):
        raise HTTPException(
            status_code=400,
            detail="That does not look like an API key - it contains spaces or line breaks.",
        )
    candidate = session.settings.model_copy(update={"google_api_key": key, "llm_backend": "gemini"})
    verdict = _check_key(candidate)
    if verdict == "rejected":
        raise HTTPException(
            status_code=400,
            detail="Google rejected that key. Copy it again from aistudio.google.com/apikey.",
        )

    if _store.public:
        session.use_gemini(key, visitors_own=True)
    else:
        set_env_value(ENV_FILE, "GOOGLE_API_KEY", key)
        _store.settings = _store.settings.model_copy(update={"google_api_key": key})
        session.use_gemini(key, visitors_own=False)
        _store.save_choice(session)
    return {
        "saved": True,
        "note": None if verdict == "ok"
        else "Saved. Google's free model is busy right now, so the first answer may be slow.",
    }


def _check_key(settings: Settings) -> str:
    """'ok', 'rejected', or 'unknown' when the model is too busy to say."""
    from .llm import Message, get_llm

    try:
        client = get_llm(settings.model_copy(update={"max_retries": 1}))
        client.complete("Reply with OK.", [Message(role="user", content="ping")], [])
        return "ok"
    except Exception as exc:
        message = str(exc).lower()
        if any(m in message for m in ("api key", "api_key", "permission_denied", "401", "403", "invalid")):
            return "rejected"
        return "unknown"


# --- questions ---------------------------------------------------------------


class AskFailed(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def _run_question(session: Session, question: str, address: str, on_event=None) -> AskResponse:
    """Everything /ask does, minus HTTP: the stream runs it on a thread."""
    if not session.settings.has_model_key():
        raise AskFailed(428, "Add a free Google Gemini key first - it is what writes the SQL.")
    if session.replaced_stale:
        session.replaced_stale = False
        raise AskFailed(409, "This page sat idle for over an hour, so its workspace was cleared. "
                             "The page has been reset - pick your data and ask again.")
    public = _store.public
    option = _option(session)
    with session.lock:
        if session.agent is None:
            if session.dataset == "mine":
                raise AskFailed(409, "You have not added any files yet. Upload a CSV or Excel file first.")
            raise AskFailed(503, f"The {session.dataset} data could not be opened: {session.error or 'unknown error'}")
        agent = session.agent
        if _metered(session):
            try:
                _quota.take(address, option.id if option else "default",
                            option.per_day if option else None, option.label if option else "")
            except QuotaExceeded as exc:
                raise AskFailed(429, str(exc)) from exc
        # On a shared server, questions queue for their model's slots rather
        # than all hitting a free tier's per-minute limit at once.
        slots = _model_slots.get(option.id) if option else None
        slots = slots or _slots
        if public:
            if on_event is not None:
                on_event({"type": "queued"})
            if not slots.acquire(timeout=SLOT_WAIT_SECONDS):
                raise AskFailed(503, "The demo is busy answering other people. Try again in a minute, "
                                     "or pick another model.")
        try:
            result = agent.ask(question, on_event=on_event)
        except ModelLimitReached as exc:
            logger.warning("Model allowance used up: %s", exc)
            if option is not None:
                _resting[option.id] = time.monotonic() + (exc.retry_after or DEFAULT_REST_SECONDS)
            name = option.name if option else "This model"
            raise AskFailed(429, f"{name} has used its free allowance for now. Pick another model"
                                 + (", or add your own free Gemini key." if public else ".")) from exc
        except Exception as exc:
            logger.warning("Ask failed: %s", exc)
            message = str(exc).lower()
            if "413" in message or "request too large" in message:
                detail = ("This question needed more working space than the free model allows. "
                          "Try asking about one part of it at a time, or pick another model.")
            elif "503" in message or "unavailable" in message or "429" in message or "exhausted" in message:
                detail = ("The free AI model is overloaded right now. Wait a minute and ask again, "
                          "or pick another model.")
            elif "api key" in message or "403" in message or "401" in message:
                detail = "The API key was rejected. Add it again using the key button at the top."
            elif "402" in message or "payment" in message or "credit" in message:
                detail = "This model's free credit has run out for now. Pick another model."
            else:
                detail = "The AI model returned an error. Try again, or rephrase the question."
            raise AskFailed(503, detail) from exc
        finally:
            if public:
                slots.release()

    return AskResponse(
        question=result.question,
        answer=result.answer,
        succeeded=result.succeeded,
        stop_reason=result.stop_reason,
        sql_queries=result.sql_queries,
        steps=[
            StepOut(
                index=step.index,
                text=step.text,
                tool_calls=[ToolCallOut(**call) for call in step.tool_calls],
                latency_ms=round(step.latency_ms, 2),
            )
            for step in result.steps
        ],
        failed_attempts=result.failed_attempts,
        total_latency_ms=round(result.total_latency_ms, 2),
        usage=result.usage,
        result=_table_json(result.last_result),
        model=_engine(session),
        parts=[
            {"question": part["question"], "answer": part["answer"], "sql": part["sql"],
             "stop_reason": part["stop_reason"], "result": _table_json(part["result"])}
            for part in result.parts
        ],
        review=result.review,
    )


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request, session: Session = Depends(current_session)) -> AskResponse:
    try:
        return _run_question(session, body.question, _client_address(request))
    except AskFailed as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


# Seconds between keep-alive lines while the model is thinking or waiting out
# a rate limit, so no proxy on the way decides the connection is dead.
HEARTBEAT_SECONDS = 10


@app.post("/ask/stream")
def ask_stream(body: AskRequest, request: Request, session: Session = Depends(current_session)):
    """/ask, reported step by step as newline-delimited JSON.

    A question can take a minute on a free tier: each step is a model call,
    and a rate limit can add a wait. Streaming the steps as they happen turns
    a spinner with no explanation into "ran a query - 12 rows - waiting 15s
    for the free tier's limit". The last line is the full /ask response, or
    an error.
    """
    events: queue.Queue = queue.Queue()
    address = _client_address(request)

    def work() -> None:
        try:
            response = _run_question(session, body.question, address, on_event=events.put)
            events.put({"type": "result", "data": response.model_dump()})
        except AskFailed as exc:
            events.put({"type": "error", "status": exc.status, "detail": exc.detail})
        except Exception as exc:  # pragma: no cover - a bug, not a model error
            logger.exception("Streaming ask crashed")
            events.put({"type": "error", "status": 500, "detail": f"Something went wrong: {exc}"})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def lines():
        while True:
            try:
                event = events.get(timeout=HEARTBEAT_SECONDS)
            except queue.Empty:
                yield json.dumps({"type": "ping"}) + "\n"
                continue
            if event is None:
                return
            yield json.dumps(event, default=str) + "\n"

    return StreamingResponse(
        lines(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/sql")
def run_sql(body: SQLRequest, session: Session = Depends(current_session)) -> dict[str, Any]:
    """Run a SELECT directly, through the same guardrails the agent uses.

    Works with no API key, so the data can be explored before one is added -
    and so anyone can check what the agent is and is not allowed to do.
    """
    _refuse_if_replaced(session)
    with session.lock:
        agent = get_agent(session)
        try:
            result = agent.warehouse.run_sql(body.sql)
        except UnsafeSQLError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except QueryTimeout as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"SQL error: {exc}") from exc

    table = _table_json(result)
    return {
        "sql": result.sql,
        **table,
        "limit_applied": result.limit_applied,
        "elapsed_ms": round(result.elapsed_ms, 2),
    }
