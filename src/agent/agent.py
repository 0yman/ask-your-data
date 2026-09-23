"""The agent loop.

A plain ReAct-style loop: the model sees the question and the schema, calls
tools, reads what comes back, and calls `final_answer` when it has the
numbers. Three things in here are the substance:

* **Self-correction.** A rejected or broken query is returned to the model as
  a message rather than raised as an error, so the next turn can fix it. A
  counter bounds how many times that is allowed before the run is stopped -
  without it, a model that cannot write valid SQL will burn the whole quota
  retrying.
* **A step budget.** Every iteration is one API call. `max_steps` is what
  stops a loop that never decides it is finished.
* **A trace.** Every step is recorded with its tool calls, timings and token
  usage. An agent you cannot inspect after the fact is an agent you cannot
  debug, and "it gave a wrong answer" is not a bug report.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .llm import LLMClient, Message, ToolCall
from .tools import FINAL_ANSWER_TOOL, ToolBox
from .warehouse import QueryResult, Warehouse

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are a data analyst for a container port \
authority. You answer questions by querying a DuckDB warehouse, and you \
answer ONLY from what the queries return.

## Schema

{schema}

## Business rules

- A "vessel call" is one visit by one vessel to one berth. `fact_vessel_call` \
grains at one row per call.
- `fact_container_movement` grains at one row per customer/cargo-type group \
within a call, so a call has several movement rows.
- Berth productivity is `moves_per_hour`. Do not use `berth_hours` to compare \
operators: it scales with how much cargo the vessel carried, not with how \
well the berth performed.
- Dwell time is in days. Demurrage is charged only on dwell beyond {free_days} \
free days, so many rows are legitimately 0.
- Join to `dim_date` on `date_key` for anything involving time periods.

## How to work

1. If you are unsure what a column contains, use `describe_table` or \
`sample_rows` before writing SQL. Guessing a column name wastes a turn.
2. Write ONE SELECT at a time with `run_sql`. Aggregate in SQL rather than \
pulling rows back and counting them yourself.
3. If a query fails, read the error, fix the query, and try again.
4. When you have the numbers, call `final_answer` with the actual figures \
written out. Never say "as shown above" - the user does not see the tables.
5. If the warehouse genuinely cannot answer the question, say so in \
`final_answer` and explain what is missing. Do not invent a number."""

FREE_DAYS = 5

# For the user's own tables. Same working rules as the port prompt, minus the
# business rules - which describe the port warehouse and would only mislead
# the model about anyone else's data.
GENERIC_PROMPT_TEMPLATE = """You are a data analyst. You answer questions by querying a DuckDB database of tables the user uploaded from their own spreadsheets, and you answer ONLY from what the queries return.

## Schema

{schema}

## About this data

- Each table came from one uploaded CSV or Excel file and is named after it.
- Column names come from the file's header row. Values may be messy: check with `sample_rows` before filtering on a text column, since spelling and capitalisation vary.
- Nothing is known about how the tables relate. Only join them if the columns clearly match, and say so in your answer when you do.

## How to work

1. If you are unsure what a column contains, use `describe_table` or `sample_rows` before writing SQL. Guessing a column name wastes a turn.
2. Write ONE SELECT at a time with `run_sql`. Aggregate in SQL rather than pulling rows back and counting them yourself.
3. If a query fails, read the error, fix the query, and try again.
4. When you have the numbers, call `final_answer` with the actual figures written out. Never say "as shown above" - the user does not see the tables.
5. If the data genuinely cannot answer the question, say so in `final_answer` and explain what is missing. Do not invent a number."""


@dataclass(slots=True)
class Step:
    index: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    text: str | None = None
    latency_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class AgentResult:
    question: str
    answer: str
    steps: list[Step]
    sql_queries: list[str]
    stop_reason: str
    succeeded: bool
    total_latency_ms: float
    usage: dict[str, int]
    failed_attempts: int = 0
    #: The rows behind the answer - the last query that ran successfully - so
    #: a person can check the figures rather than take the prose on trust.
    last_result: QueryResult | None = None

    @property
    def final_sql(self) -> str | None:
        return self.sql_queries[-1] if self.sql_queries else None

    def format_trace(self) -> str:
        lines = [f"Q: {self.question}", ""]
        for step in self.steps:
            lines.append(f"--- step {step.index} ({step.latency_ms:.0f}ms) ---")
            if step.text:
                lines.append(f"  thought: {step.text[:200]}")
            for call in step.tool_calls:
                status = "ok" if call.get("ok", True) else f"FAILED ({call.get('error_kind')})"
                lines.append(f"  {call['name']}({_short_args(call['arguments'])}) -> {status}")
        lines.append("")
        lines.append(f"stop_reason: {self.stop_reason}")
        lines.append(f"answer: {self.answer}")
        return "\n".join(lines)


def _short_args(arguments: dict[str, Any], width: int = 90) -> str:
    rendered = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    return rendered if len(rendered) <= width else rendered[: width - 1] + "…"


class PortAnalystAgent:
    def __init__(
        self,
        warehouse: Warehouse,
        llm: LLMClient | None,
        settings: Settings,
        domain: str = "port",
    ) -> None:
        self.warehouse = warehouse
        self.settings = settings
        self.domain = domain
        self._llm = llm
        schema = (
            warehouse.schema_summary()
            if settings.include_schema_in_prompt
            else "(Not provided. Use list_tables and describe_table to discover it.)"
        )
        if domain == "port":
            # Kept byte-for-byte: the published evaluation ran on this text.
            self._system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                schema=schema, free_days=FREE_DAYS
            )
        else:
            self._system_prompt = GENERIC_PROMPT_TEMPLATE.format(schema=schema)

    @property
    def llm(self) -> LLMClient:
        """Built on first use.

        Schema introspection and direct SQL need no model at all. Building the
        client eagerly would make the whole service unavailable without an API
        key, when only `/ask` actually requires one.
        """
        if self._llm is None:
            from .llm import get_llm

            self._llm = get_llm(self.settings)
        return self._llm

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def ask(self, question: str) -> AgentResult:
        settings = self.settings
        toolbox = ToolBox(
            warehouse=self.warehouse, max_rows_to_model=settings.max_rows_to_model
        )
        messages: list[Message] = [Message(role="user", content=question)]
        steps: list[Step] = []
        usage = {"prompt_tokens": 0, "output_tokens": 0}

        started = time.perf_counter()
        answer: str | None = None
        stop_reason = "max_steps"
        failed_attempts = 0

        for index in range(1, settings.max_steps + 1):
            step_started = time.perf_counter()
            response = self.llm.complete(self._system_prompt, messages, toolbox.specs)
            step = Step(
                index=index,
                text=response.text,
                latency_ms=(time.perf_counter() - step_started) * 1000,
                usage=response.usage,
            )
            for key, value in response.usage.items():
                usage[key] = usage.get(key, 0) + value

            if not response.wants_tools:
                # The model answered in prose without calling final_answer.
                # Accept it rather than nagging - it has the numbers or it
                # does not, and another turn costs a request either way.
                steps.append(step)
                answer = response.text or "The model returned an empty response."
                stop_reason = "text_answer"
                break

            messages.append(
                Message(role="assistant", content=response.text, tool_calls=response.tool_calls)
            )

            finished = False
            for call in response.tool_calls:
                if call.name == FINAL_ANSWER_TOOL:
                    answer = str(call.arguments.get("answer", "")).strip()
                    step.tool_calls.append(
                        {"name": call.name, "arguments": call.arguments, "ok": True}
                    )
                    stop_reason = "final_answer"
                    finished = True
                    break

                outcome = toolbox.dispatch(call.name, call.arguments)
                if not outcome.ok:
                    failed_attempts += 1
                step.tool_calls.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "ok": outcome.ok,
                        "error_kind": outcome.error_kind,
                    }
                )
                messages.append(
                    Message(
                        role="tool",
                        content=outcome.content,
                        tool_name=call.name,
                        tool_call_id=call.id,
                    )
                )

            steps.append(step)
            if finished:
                break

            if failed_attempts > settings.max_sql_retries:
                stop_reason = "too_many_failures"
                answer = (
                    f"I could not produce a working query after "
                    f"{failed_attempts} failed attempts. The last error was: "
                    f"{_last_error(messages)}"
                )
                break

        if answer is None:
            answer = (
                f"I ran out of steps ({settings.max_steps}) before reaching an "
                "answer. The question may need to be narrowed."
            )

        return AgentResult(
            question=question,
            answer=answer,
            steps=steps,
            sql_queries=[query.sql for query in toolbox.executed_queries],
            stop_reason=stop_reason,
            succeeded=stop_reason in {"final_answer", "text_answer"},
            total_latency_ms=(time.perf_counter() - started) * 1000,
            usage=usage,
            failed_attempts=failed_attempts,
            last_result=toolbox.executed_queries[-1] if toolbox.executed_queries else None,
        )


def _last_error(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "tool" and message.content:
            return message.content[:200]
    return "unknown"


def build_agent(
    settings: Settings, llm: LLMClient | None = None, dataset: str = "sample"
) -> PortAnalystAgent:
    """Construct an agent over the sample warehouse, a public example dataset,
    or the user's own tables.

    The LLM client is left unbuilt unless one is given, so this succeeds - and
    the schema and SQL tools work - without an API key.
    """
    path = settings.dataset_path(dataset)
    warehouse = Warehouse(
        path,
        max_rows=settings.max_rows,
        timeout_seconds=settings.query_timeout_seconds,
    )
    return PortAnalystAgent(warehouse, llm, settings, domain="port" if dataset == "sample" else "user")


__all__ = ["AgentResult", "PortAnalystAgent", "Step", "ToolCall", "build_agent"]
