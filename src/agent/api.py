"""FastAPI service.

`/ask` returns the full trace alongside the answer, not just the prose. For an
agent that is the difference between a product and a magic box: the caller can
see which tools ran, which SQL executed, what failed and was corrected, and
what it cost.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import PortAnalystAgent, build_agent
from .config import get_settings
from .guardrails import UnsafeSQLError
from .warehouse import QueryTimeout

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_state: dict[str, Any] = {"agent": None, "error": None}


def get_agent() -> PortAnalystAgent:
    """The agent, for endpoints that only need the warehouse.

    Available whenever the warehouse loaded, with or without an API key.
    """
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail=f"Warehouse unavailable: {_state.get('error') or 'not initialised'}",
        )
    return agent


def get_agent_with_llm() -> PortAnalystAgent:
    """The agent, for /ask, which genuinely needs a model behind it."""
    agent = get_agent()
    try:
        agent.llm  # noqa: B018 - constructs the client, raising if no key is set
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["agent"] = build_agent(get_settings())
        logger.info("Agent ready")
    except Exception as exc:
        # A missing warehouse or API key should surface on /health as a clear
        # message, not crash-loop the container.
        logger.warning("Agent unavailable: %s", exc)
        _state["agent"], _state["error"] = None, str(exc)
    yield
    agent = _state.get("agent")
    if agent is not None:
        agent.warehouse.close()
    _state.clear()


app = FastAPI(
    title="Port Analyst Agent",
    version="1.0.0",
    description=(
        "A tool-calling agent that answers analytical questions about container "
        "terminal operations by writing and running read-only SQL against a "
        "DuckDB star schema. Every run returns its full trace."
    ),
    lifespan=lifespan,
)

try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, include_in_schema=False)
except ImportError:  # pragma: no cover
    logger.info("prometheus-fastapi-instrumentator not installed; /metrics disabled")


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


class SQLRequest(BaseModel):
    sql: str = Field(min_length=1, max_length=20000)


@app.get("/health")
def health() -> dict[str, Any]:
    agent = _state.get("agent")
    llm_ready = False
    llm_error = None
    if agent is not None:
        try:
            agent.llm  # noqa: B018 - probes whether a client can be built
            llm_ready = True
        except Exception as exc:
            llm_error = str(exc)
    return {
        # The warehouse endpoints work without a model, so a missing key is
        # "degraded", not "down".
        "status": "ok" if llm_ready else ("degraded" if agent else "down"),
        "warehouse_ready": agent is not None,
        "llm_ready": llm_ready,
        "error": _state.get("error") or llm_error,
    }


@app.get("/schema")
def schema() -> dict[str, Any]:
    agent = get_agent()
    return {
        "tables": [
            {
                "name": table.name,
                "kind": table.kind,
                "row_count": table.row_count,
                "columns": [
                    {"name": c.name, "type": c.type, "nullable": c.nullable}
                    for c in agent.warehouse.describe_table(table.name).columns
                ],
            }
            for table in agent.warehouse.list_tables()
        ]
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    agent = get_agent_with_llm()
    result = agent.ask(request.question)
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
    )


@app.post("/sql")
def run_sql(request: SQLRequest) -> dict[str, Any]:
    """Run a SELECT directly, through the same guardrails the agent uses.

    Exposed so the guardrails can be exercised without spending model calls -
    and so a human can check what the agent is allowed to do.
    """
    agent = get_agent()
    try:
        result = agent.warehouse.run_sql(request.sql)
    except UnsafeSQLError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except QueryTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"SQL error: {exc}") from exc

    return {
        "sql": result.sql,
        "columns": result.columns,
        "rows": [list(row) for row in result.rows],
        "row_count": result.row_count,
        "limit_applied": result.limit_applied,
        "elapsed_ms": round(result.elapsed_ms, 2),
    }
