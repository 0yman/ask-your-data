"""FastAPI service: the web app at `/`, and the JSON API behind it.

Two datasets can be active: the sample port warehouse, or tables built from
the user's own CSV and Excel files. One is shown at a time - an agent looking
at both would have to guess which tables a question is about.

`/ask` returns the full trace alongside the answer. For an agent that is the
difference between a product and a magic box: the caller can see which tools
ran, which SQL executed, what failed and was corrected, and the rows behind
the figures.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import logging
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import PortAnalystAgent, build_agent
from .config import ENV_FILE, Settings, get_settings
from .datasets import SUPPORTED_SUFFIXES, DatasetError, drop_table, import_file, table_count
from .envfile import set_env_value
from .guardrails import UnsafeSQLError
from .warehouse import QueryResult, QueryTimeout

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
RESULT_ROW_LIMIT = 200

Dataset = Literal["sample", "mine"]

_state: dict[str, Any] = {"settings": None, "dataset": "sample", "agent": None, "error": None}
# One user, one warehouse connection. FastAPI runs sync endpoints on a thread
# pool, and swapping datasets or importing a file mid-question would otherwise
# pull the connection out from under a running query.
_lock = threading.RLock()


# --- state ------------------------------------------------------------------


def _settings() -> Settings:
    return _state["settings"]


def _load_dataset_choice(settings: Settings) -> Dataset:
    try:
        choice = json.loads(settings.state_path.read_text(encoding="utf-8")).get("dataset")
    except (OSError, ValueError):
        return "sample"
    return "mine" if choice == "mine" else "sample"


def _save_dataset_choice(settings: Settings, dataset: Dataset) -> None:
    settings.state_path.parent.mkdir(parents=True, exist_ok=True)
    settings.state_path.write_text(json.dumps({"dataset": dataset}), encoding="utf-8")


def _close_agent() -> None:
    agent: PortAnalystAgent | None = _state.get("agent")
    if agent is not None:
        agent.warehouse.close()
    _state["agent"] = None


def _open(dataset: Dataset) -> None:
    """Point the agent at a dataset. Must be called holding `_lock`."""
    _close_agent()
    _state["dataset"] = dataset
    _state["error"] = None
    settings = _settings()
    if dataset == "mine" and table_count(settings.user_db_path) == 0:
        return  # nothing uploaded yet: a normal state, not an error
    try:
        _state["agent"] = build_agent(settings, dataset=dataset)
    except Exception as exc:
        logger.warning("Could not open the %s dataset: %s", dataset, exc)
        _state["error"] = str(exc)


def get_agent() -> PortAnalystAgent:
    agent = _state.get("agent")
    if agent is None:
        if _state.get("dataset") == "mine":
            raise HTTPException(
                status_code=409,
                detail="You have not added any files yet. Upload a CSV or Excel file first.",
            )
        raise HTTPException(
            status_code=503,
            detail=f"The sample data could not be opened: {_state.get('error') or 'unknown error'}",
        )
    return agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    with _lock:
        _state["settings"] = settings
        _open(_load_dataset_choice(settings))
    yield
    with _lock:
        _close_agent()


app = FastAPI(
    title="Ask your data",
    version="2.0.0",
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


def _engine_label(settings: Settings) -> str:
    if settings.llm_backend == "openai":
        return "OpenAI-compatible model"
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


class SQLRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=20000)


class DatasetRequest(BaseModel):
    dataset: Dataset


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
    agent = _state.get("agent")
    return {
        "status": "ok" if agent or _state.get("dataset") == "mine" else "down",
        "dataset": _state.get("dataset"),
        "warehouse_ready": agent is not None,
        "llm_ready": bool(settings and settings.has_model_key()),
        "error": _state.get("error"),
    }


@app.get("/status")
def status() -> dict[str, Any]:
    settings = _settings()
    return {
        "has_key": settings.has_model_key(),
        "engine_label": _engine_label(settings),
        "dataset": _state["dataset"],
        "sample_available": settings.db_path.exists(),
        "my_tables": table_count(settings.user_db_path),
        "max_upload_mb": settings.max_upload_mb,
        "accepted_types": sorted(SUPPORTED_SUFFIXES),
    }


@app.get("/tables")
def tables() -> list[dict[str, Any]]:
    """The active dataset's tables, with columns - what the sidebar lists."""
    with _lock:
        agent = _state.get("agent")
        if agent is None:
            return []
        out = []
        for table in agent.warehouse.list_tables():
            info = agent.warehouse.describe_table(table.name)
            out.append({
                "name": info.name,
                "kind": info.kind if _state["dataset"] == "sample" else "table",
                "rows": info.row_count,
                "columns": [{"name": c.name, "type": c.type} for c in info.columns],
            })
        return out


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {"tables": tables()}


# --- choosing and managing data ----------------------------------------------


@app.post("/dataset")
def choose_dataset(request: DatasetRequest) -> dict[str, Any]:
    with _lock:
        _open(request.dataset)
        _save_dataset_choice(_settings(), request.dataset)
    return {"dataset": request.dataset}


@app.post("/data", response_model=list[UploadResult])
def upload_data(files: list[UploadFile] = File(...)) -> list[UploadResult]:
    settings = _settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    results: list[UploadResult] = []

    with _lock:
        # DuckDB will not open one file read-only and read-write at once, so
        # the agent lets go of it while the tables are written.
        _close_agent()
        try:
            for upload in files:
                name = safe_filename(upload.filename or "")
                if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                    results.append(UploadResult(
                        filename=upload.filename or name, status="skipped",
                        reason="unsupported type - use CSV, TSV or Excel (.xlsx)",
                    ))
                    continue
                destination = settings.uploads_dir / name
                try:
                    written = 0
                    with destination.open("wb") as handle:
                        while chunk := upload.file.read(1024 * 1024):
                            written += len(chunk)
                            if written > max_bytes:
                                raise DatasetError(f"larger than the {settings.max_upload_mb} MB limit")
                            handle.write(chunk)
                    imported = import_file(destination, settings.user_db_path)
                except DatasetError as exc:
                    results.append(UploadResult(filename=name, status="skipped", reason=str(exc)))
                    continue
                finally:
                    # The table holds the data now; the raw file is not needed.
                    destination.unlink(missing_ok=True)
                results.append(UploadResult(
                    filename=name, status="added", table=imported.table,
                    rows=imported.rows, columns=imported.columns,
                ))
        finally:
            added = any(r.status == "added" for r in results)
            dataset: Dataset = "mine" if added else _state["dataset"]
            _open(dataset)
            _save_dataset_choice(settings, dataset)
    return results


@app.delete("/data/{table}")
def delete_table(table: str) -> dict[str, Any]:
    settings = _settings()
    with _lock:
        _close_agent()
        try:
            removed = drop_table(table, settings.user_db_path)
        finally:
            _open(_state["dataset"])
    if not removed:
        raise HTTPException(status_code=404, detail="No such table in your data.")
    return {"removed": table}


# --- the API key --------------------------------------------------------------


@app.post("/settings/key")
def save_key(body: KeyRequest, request: Request) -> dict[str, Any]:
    """Save a Gemini key from the web page into .env.

    Only from this computer: a key-writing endpoint reachable from the
    network would let anyone on it swap in their own key. (Docker users set
    the key in .env or the environment instead.)
    """
    if not _is_local(request):
        raise HTTPException(status_code=403, detail="Keys can only be set from this computer.")

    key = body.key.strip()
    # API keys never contain whitespace. Refusing it here also means a pasted
    # value can never carry a newline into .env and write a second setting.
    if any(ch.isspace() for ch in key):
        raise HTTPException(
            status_code=400,
            detail="That does not look like an API key - it contains spaces or line breaks.",
        )
    candidate =_settings().model_copy(update={"google_api_key": key, "llm_backend": "gemini"})
    verdict = _check_key(candidate)
    if verdict == "rejected":
        raise HTTPException(
            status_code=400,
            detail="Google rejected that key. Copy it again from aistudio.google.com/apikey.",
        )

    set_env_value(ENV_FILE, "GOOGLE_API_KEY", key)
    with _lock:
        _state["settings"] = candidate
        _open(_state["dataset"])
    return {
        "saved": True,
        "note": None if verdict == "ok"
        else "Saved. Google's free model is busy right now, so the first answer may be slow.",
    }


def _check_key(settings: Settings) -> str:
    """'ok', 'rejected', or 'unknown' when the model is too busy to say."""
    from .llm import LLMResponse, Message, get_llm  # noqa: F401 - LLMResponse kept for typing

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


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not _settings().has_model_key():
        raise HTTPException(
            status_code=428,
            detail="Add a free Google Gemini key first - it is what writes the SQL.",
        )
    with _lock:
        agent = get_agent()
        try:
            result = agent.ask(request.question)
        except Exception as exc:
            logger.warning("Ask failed: %s", exc)
            message = str(exc).lower()
            if "503" in message or "unavailable" in message or "429" in message or "exhausted" in message:
                detail = "Google's free AI model is overloaded right now. Wait a minute and ask again."
            elif "api key" in message or "403" in message or "401" in message:
                detail = "The API key was rejected. Add it again using the key button at the top."
            else:
                detail = "The AI model returned an error. Try again, or rephrase the question."
            raise HTTPException(status_code=503, detail=detail) from exc

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
    )


@app.post("/sql")
def run_sql(request: SQLRequest) -> dict[str, Any]:
    """Run a SELECT directly, through the same guardrails the agent uses.

    Works with no API key, so the data can be explored before one is added -
    and so anyone can check what the agent is and is not allowed to do.
    """
    with _lock:
        agent = get_agent()
        try:
            result = agent.warehouse.run_sql(request.sql)
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
