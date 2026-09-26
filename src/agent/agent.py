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

import copy
import logging
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from .config import Settings
from .llm import LLMClient, Message, ToolCall
from .tools import FINAL_ANSWER_TOOL, ToolBox
from .warehouse import QueryResult, Warehouse

logger = logging.getLogger(__name__)

# How to work, shared by both prompts: the method of an expert analyst.
# Written after the agent failed broad questions by trying to answer them in
# one 30-line query; measured on a harder, multi-part question set
# (eval/user_data/hard_questions.jsonl) before and after - see the README.
HOW_TO_WORK = """## How to work

1. **Understand the question.** Work out exactly what is asked and what each part needs. If a term is ambiguous ("sales", "customers", "last year"), pick the most reasonable reading and state it in your answer. If a premise is impossible - a date that does not exist, a column the data does not have - say so instead of computing something else.
2. **Split big questions.** A question with several parts, or a broad one ("how can the business...", "what patterns..."), becomes a short list of concrete sub-questions. Answer them one at a time, each with its own small query. Never try to answer everything in one giant query.
3. **Look before you compute.** Before filtering or summing a column, check what it holds: its type, NULLs, negative values (returns, refunds), cancelled or test records, duplicates, units, the date range, and rows that are totals or groups rather than single entities ("World", "Total", "All", regions). Exclude what the question asks you to exclude, and rows that are plainly not what is being counted, such as totals; mention anything else unusual as a caveat instead of silently dropping it.
4. **Write careful SQL.** One SELECT per call, aggregated in SQL. Build multi-step logic with CTEs (`WITH ...`); use window functions and `QUALIFY` for rankings and top-N per group, `COUNT(*) FILTER (WHERE ...)` for conditional counts, `year()`, `month()` and `date_trunc()` for time, and `TRY_CAST` for numbers stored as text. On timestamp columns, bound dates by the next day: "after 30 June" is `>= DATE '...-07-01'`, and "up to 30 June" is `< DATE '...-07-01'` - a time on 30 June is not after 30 June. Compute percentages in SQL. Keep each query short enough to check - about 25 lines at most.
5. **Check the numbers.** Before answering, re-read the question against your SQL: every condition it states must be in the query (a "first purchase in 2011" means no earlier purchase, not just a purchase in 2011), and nothing it did not ask for. Then sanity-check the figures: parts should add up to their totals, shares should not exceed 100%, magnitudes and units should be plausible. If a result surprises you, verify it with a second query instead of explaining it away.
6. **Recover from errors.** Read the error, fix its cause and retry. If a query keeps failing, simplify it or approach that sub-question another way.
7. **Answer.** Call `final_answer` with every part of the question answered, each with its actual figures - never "as shown above", the user does not see the tables. For a multi-part question, give one short line per part. State the assumptions you made. If the data cannot answer a part, say which part and what is missing. Never invent a number."""

SYSTEM_PROMPT_TEMPLATE = """You are an expert data analyst and SQL engineer \
for a container port authority. You answer questions by querying a DuckDB \
warehouse, and you answer ONLY from what the queries return.

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

""" + HOW_TO_WORK

FREE_DAYS = 5

# For the user's own tables. Same working rules as the port prompt, minus the
# business rules - which describe the port warehouse and would only mislead
# the model about anyone else's data.
GENERIC_PROMPT_TEMPLATE = """You are an expert data analyst and SQL engineer. You answer questions by querying a DuckDB database of tables the user uploaded from their own spreadsheets, and you answer ONLY from what the queries return.

## Schema

{schema}

## About this data

- Each table came from one uploaded CSV or Excel file and is named after it.
- Column names come from the file's header row. Values may be messy: check with `sample_rows` before filtering on a text column, since spelling and capitalisation vary.
- Nothing is known about how the tables relate. Only join them if the columns clearly match, and say so in your answer when you do.

""" + HOW_TO_WORK


# Sent when failures or the step budget stop the loop but some queries did
# work. Answering from those beats discarding them: a broad question that
# needed five queries and got three still deserves the three.
SALVAGE_NUDGE = (
    "You cannot run any more queries. Answer the question now with the "
    "final_answer tool, using only the results of the queries that succeeded "
    "above. Say plainly which parts of the question you could not compute."
)

# Stand-in for the SQL of a failed attempt once a later attempt exists. The
# error stays, the 30-line query goes: retries otherwise grow the context
# until a host with a small per-request limit (Groq's free tier: 7K tokens)
# refuses the whole question.
FAILED_SQL_PLACEHOLDER = "(failed query omitted to save space - see the error below)"


# Sent once when the model returns an empty turn. Not part of the system
# prompt, so the evaluated port prompt stays byte-identical.
EMPTY_REPLY_NUDGE = (
    "Your last reply was empty. Using the query results above, give the answer "
    "now with the final_answer tool."
)

# Models put a currency symbol in front of money figures whether or not the
# data says which currency it is in: on the UK shop's sales, 18 of 63 stored
# answers carried "$" or "£" with nothing in the data to back either. A
# prompt rule fixed that but moved the hard set from 9.7 to 7.3 of 12 (the
# model's SQL changed along with its wording), so the fix is here, applied to
# the finished answer, where it cannot touch the queries.
_CURRENCY_NAMED = re.compile(r"[$£€]|usd|eur|gbp|dollar|euro|pound|sterling|currency", re.IGNORECASE)
_STRAY_CURRENCY = re.compile(r"[$£€]\s?(?=-?\d)")


_FIGURE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _figures(answer: str) -> frozenset[str]:
    """The numbers in an answer, to four significant digits, so "284,661.54"
    and "284,662" count as the same figure."""
    found = set()
    for text in _FIGURE.findall(answer or ""):
        try:
            found.add(f"{float(text.replace(',', '')):.4g}")
        except ValueError:
            continue
    return frozenset(found)


def pick_by_vote(results: list[AgentResult]) -> tuple[int, int]:
    """The run whose answer the most runs agree with, and how many agree
    (itself included). Two answers agree when at least half of their figures
    are shared; two answers with no figures (two declines) agree. Runs that
    did not finish only count when none did. Ties go to the earliest run -
    the one at the configured temperature."""
    finished = [i for i, r in enumerate(results) if r.succeeded] or list(range(len(results)))
    figures = {i: _figures(results[i].answer) for i in finished}

    def agree(a: int, b: int) -> bool:
        union = figures[a] | figures[b]
        return not union or len(figures[a] & figures[b]) / len(union) >= 0.5

    best = max(finished, key=lambda i: (sum(agree(i, j) for j in finished), -i))
    return best, sum(agree(best, j) for j in finished)


# Sent once when a final answer carries figures that are in no query result,
# no query and not in the question: numbers worked out in the model's head, or
# misread. Checking this in code costs nothing when the answer is grounded -
# a model reviewing itself (planning.py's verifier) cost more than it caught.
GROUNDING_NUDGE = (
    "Before this answer is shown: these figures in it are not in any query "
    "result above - {figures}. If they come from a result, copy them exactly; "
    "if you worked them out yourself, compute them with a query instead. Then "
    "call final_answer again."
)
# A figure matches a result value within its own rounding, at any of these
# scales: "22.2 billion" for 22,237.81 million, "49.3%" for 0.493.
_SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9, 100.0, 0.01)


def _numbers(text: str) -> list[float]:
    found = []
    for piece in _FIGURE.findall(text or ""):
        try:
            found.append(abs(float(piece.replace(",", ""))))
        except ValueError:
            continue
    return found


def _known_numbers(question: str, results: list[QueryResult]) -> list[float]:
    known = _numbers(question)
    for result in results:
        known += _numbers(result.sql)
        for row in result.rows:
            for value in row:
                if isinstance(value, bool) or value is None:
                    continue
                if isinstance(value, str):
                    known += _numbers(value)
                    continue
                try:
                    known.append(abs(float(value)))
                except (TypeError, ValueError):
                    continue
    return known


def ungrounded_figures(answer: str, question: str, results: list[QueryResult], context: str = "") -> list[str]:
    """Figures in `answer` found in no query result, no query, not in the
    question and not in `context` - the schema the model was shown, whose row
    counts it may quote. Years and counts up to 10 are left alone: they come
    from the question's own framing as often as from a result."""
    known = _known_numbers(f"{question}\n{context}", results)
    loose = []
    for piece in _FIGURE.findall(answer or ""):
        try:
            value = abs(float(piece.replace(",", "")))
        except ValueError:
            continue
        if value.is_integer() and (value <= 10 or 1900 <= value <= 2100):
            continue
        decimals = len(piece.split(".")[1]) if "." in piece else 0
        tolerance = 0.5 * 10 ** -decimals
        if not any(abs(value - k * scale) <= tolerance for k in known for scale in _SCALES):
            loose.append(piece)
    return loose


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text).lstrip("0")


def _edits(a: str, b: str) -> int:
    """Levenshtein distance between two short digit strings."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def nearest_result_value(figure: str, question: str, results: list[QueryResult]) -> str | None:
    """The result value a long figure was probably copied from, when it is
    one or two digits off - "1,064,456.42" for 10,644,560.42. In stored runs
    the model, told only that a figure was wrong, wrote another wrong one."""
    wanted = _digits(figure)
    if len(wanted) < 5:
        return None
    best, best_edits = None, 3
    for value in _known_numbers(question, results):
        shown = f"{value:,.4f}".rstrip("0").rstrip(".")
        digits = _digits(shown)
        if abs(len(digits) - len(wanted)) > 2:
            continue
        edits = _edits(wanted, digits)
        if 0 < edits < best_edits:
            best, best_edits = shown, edits
    return best


def drop_unstated_currency(answer: str, question: str, schema: str) -> str:
    """Take the currency symbol off figures when neither the question nor any
    table or column name says which currency the data is in. A column such as
    `demurrage_usd` does say, and then the answer is left as it is."""
    if not answer or _CURRENCY_NAMED.search(question) or _CURRENCY_NAMED.search(schema):
        return answer
    return _STRAY_CURRENCY.sub("", answer)


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
    #: When the question was split (planning.py): each part's question,
    #: answer, final SQL and rows. Empty for a question answered in one pass.
    parts: list[dict[str, Any]] = field(default_factory=list)
    #: The verifier's verdict on the answer, when one ran.
    review: dict[str, Any] | None = None
    #: With vote_runs > 1: how many runs answered, and how many of them
    #: gave the figures of the answer kept.
    votes: dict[str, int] | None = None

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
        llm_factory: Callable[[Settings], LLMClient] | None = None,
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
        # Builds the clients for the extra vote_runs, at vote_temperature.
        self._llm_factory = llm_factory
        # Table and column names, for drop_unstated_currency. Read on first
        # use when the prompt does not carry them.
        self._schema_text = schema if settings.include_schema_in_prompt else None
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

    def ask(self, question: str, on_event: Callable[[dict[str, Any]], None] | None = None) -> AgentResult:
        """Answer a question. `on_event`, if given, hears each step as it
        happens - what a page shows while a slow question is being worked
        on, instead of a spinner and no explanation."""
        emit = on_event or (lambda event: None)
        llm = self.llm
        llm.on_wait = (lambda seconds: emit({"type": "wait", "seconds": round(seconds)})) if on_event else None
        try:
            if self.settings.plan_questions or self.settings.verify_answers:
                from .planning import ask_planned

                return self._tidy(question, ask_planned(self, question, emit))
            if self.settings.vote_runs > 1:
                return self._tidy(question, self._ask_voted(question, emit))
            return self._tidy(question, self._ask(question, emit))
        finally:
            llm.on_wait = None

    def _ask_voted(self, question: str, emit: Callable[[dict[str, Any]], None]) -> AgentResult:
        """Self-consistency. The page follows the first run; the others work
        quietly beside it. A run lost to a host error (a 429 on a free tier,
        where three runs at once hit the per-minute limit) is left out; the
        question fails only when every run does."""
        from .llm import get_llm

        started = time.perf_counter()
        warm = self.settings.model_copy(update={"temperature": self.settings.vote_temperature})
        runners = [self]
        for _ in range(self.settings.vote_runs - 1):
            twin = copy.copy(self)
            twin.settings = warm
            twin._llm = (self._llm_factory or get_llm)(warm)
            runners.append(twin)

        def run(index: int) -> AgentResult:
            return runners[index]._ask(question, emit if index == 0 else (lambda event: None))

        results: list[AgentResult] = []
        first_error: Exception | None = None
        with ThreadPoolExecutor(max_workers=len(runners)) as pool:
            futures = [pool.submit(run, i) for i in range(len(runners))]
            for index, future in enumerate(futures):
                try:
                    results.append(future.result())
                except Exception as exc:         # noqa: BLE001 - the other runs may still answer
                    logger.warning("Vote run %d failed: %s", index, exc)
                    first_error = first_error or exc
        if not results:
            raise first_error

        best, agreeing = pick_by_vote(results)
        usage = {key: sum(r.usage.get(key, 0) for r in results) for key in ("prompt_tokens", "output_tokens")}
        emit({"type": "vote", "runs": len(results), "agreeing": agreeing})
        return replace(results[best], usage=usage, votes={"runs": len(results), "agreeing": agreeing},
                       total_latency_ms=(time.perf_counter() - started) * 1000)

    def _tidy(self, question: str, result: AgentResult) -> AgentResult:
        if self._schema_text is None:
            self._schema_text = self.warehouse.schema_summary()
        def tidy(text):
            return drop_unstated_currency(text, question, self._schema_text)
        return replace(result, answer=tidy(result.answer),
                       parts=[{**part, "answer": tidy(part.get("answer"))} for part in result.parts])

    def _ask(
        self, question: str, emit: Callable[[dict[str, Any]], None], max_steps: int | None = None
    ) -> AgentResult:
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
        # Failures in a row, not in total: a broad question split into six
        # sub-queries can fix one in each and still be on track.
        failed_in_a_row = 0
        nudged = False
        rechecked = False
        failed_ids: set[str] = set()

        for index in range(1, (max_steps or settings.max_steps) + 1):
            step_started = time.perf_counter()
            emit({"type": "thinking", "step": index})
            response = self._complete(messages, toolbox.specs, failed_ids)
            step = Step(
                index=index,
                text=response.text,
                latency_ms=(time.perf_counter() - step_started) * 1000,
                usage=response.usage,
            )
            for key, value in response.usage.items():
                usage[key] = usage.get(key, 0) + value

            if not response.wants_tools and not response.text and not nudged:
                # An empty turn - no text, no tool call. gpt-oss does this
                # after a tool result now and then; one nudge gets the answer
                # it already has, where stopping would throw the work away.
                steps.append(step)
                messages.append(Message(role="user", content=EMPTY_REPLY_NUDGE))
                nudged = True
                continue

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
                    draft = str(call.arguments.get("answer", "")).strip()
                    loose = [] if rechecked or index == (max_steps or settings.max_steps) else (
                        ungrounded_figures(draft, question, toolbox.executed_queries, self._schema_text or ""))
                    if loose:
                        rechecked = True
                        logger.warning("Answer figures in no query result: %s - asked to recheck", ", ".join(loose))
                        step.tool_calls.append({"name": call.name, "arguments": call.arguments,
                                                "ok": False, "error_kind": "ungrounded"})
                        messages.append(Message(role="tool", tool_name=call.name, tool_call_id=call.id,
                                                content=GROUNDING_NUDGE.format(figures=", ".join(
                                                    f"{f} (a result has {near})" if (near := nearest_result_value(
                                                        f, question, toolbox.executed_queries)) else f
                                                    for f in loose))))
                        emit({"type": "recheck", "figures": loose})
                        break
                    answer = draft
                    step.tool_calls.append(
                        {"name": call.name, "arguments": call.arguments, "ok": True}
                    )
                    stop_reason = "final_answer"
                    finished = True
                    break

                outcome = toolbox.dispatch(call.name, call.arguments)
                if not outcome.ok:
                    failed_attempts += 1
                    failed_in_a_row += 1
                    failed_ids.add(call.id)
                else:
                    failed_in_a_row = 0
                emit({
                    "type": "tool", "step": index, "name": call.name,
                    "arguments": call.arguments, "ok": outcome.ok,
                    "error_kind": outcome.error_kind,
                    "rows": outcome.result.row_count if outcome.result is not None else None,
                })
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

            if failed_in_a_row > settings.max_sql_retries:
                stop_reason = "too_many_failures"
                break

        if answer is None:
            if toolbox.executed_queries:
                emit({"type": "salvage"})
            salvaged = self._salvage(messages, toolbox, failed_ids, steps, usage)
            if salvaged is not None:
                answer, stop_reason = salvaged, "partial_answer"
            elif stop_reason == "too_many_failures":
                answer = (
                    f"I could not produce a working query after "
                    f"{failed_attempts} failed attempts. The last error was: "
                    f"{_last_error(messages)}\n\nTry asking about one part of the "
                    "question at a time."
                )
            else:
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
            succeeded=stop_reason in {"final_answer", "text_answer", "partial_answer"},
            total_latency_ms=(time.perf_counter() - started) * 1000,
            usage=usage,
            failed_attempts=failed_attempts,
            last_result=toolbox.executed_queries[-1] if toolbox.executed_queries else None,
        )


    # --- keeping the context small -------------------------------------------

    def _complete(self, messages: list[Message], specs, failed_ids: set[str], system: str | None = None):
        """One model call, sent with failed attempts compacted.

        If the host still refuses the request as too large, retry once with
        every tool result cut short - the last chance before the question
        fails for reasons the visitor cannot see.
        """
        system = system or self._system_prompt
        compact = _compact(messages, failed_ids)
        try:
            return self.llm.complete(system, compact, specs)
        except Exception as exc:
            if not _too_large(exc):
                raise
            logger.warning("Request too large for the model host; retrying trimmed")
            return self.llm.complete(system, _compact(messages, failed_ids, max_result_chars=1200), specs)

    def _salvage(self, messages, toolbox: ToolBox, failed_ids, steps, usage) -> str | None:
        """Ask for an answer from the queries that did work, if any did."""
        if not toolbox.executed_queries:
            return None
        final_only = [spec for spec in toolbox.specs if spec.name == FINAL_ANSWER_TOOL]
        request = messages + [Message(role="user", content=SALVAGE_NUDGE)]
        started = time.perf_counter()
        try:
            response = self._complete(request, final_only, failed_ids)
        except Exception as exc:
            logger.warning("Salvage call failed: %s", exc)
            return None
        step = Step(index=len(steps) + 1, text=response.text,
                    latency_ms=(time.perf_counter() - started) * 1000, usage=response.usage)
        for key, value in response.usage.items():
            usage[key] = usage.get(key, 0) + value
        answer = None
        for call in response.tool_calls:
            if call.name == FINAL_ANSWER_TOOL:
                answer = str(call.arguments.get("answer", "")).strip()
                step.tool_calls.append({"name": call.name, "arguments": call.arguments, "ok": True})
                break
        answer = answer or (response.text or "").strip() or None
        steps.append(step)
        return answer


def _compact(messages: list[Message], failed_ids: set[str], max_result_chars: int | None = None) -> list[Message]:
    """The conversation as sent: failed SQL replaced by a placeholder once a
    later attempt exists (the newest failure stays whole, so the model can
    fix it), and optionally every tool result cut to `max_result_chars`."""
    latest_failure = None
    for message in reversed(messages):
        if message.role == "assistant" and any(c.id in failed_ids for c in message.tool_calls):
            latest_failure = message
            break
    out = []
    for message in messages:
        if message.role == "assistant" and message is not latest_failure and any(
            c.id in failed_ids for c in message.tool_calls
        ):
            calls = [
                replace(c, arguments={"sql": FAILED_SQL_PLACEHOLDER})
                if c.id in failed_ids and c.name == "run_sql" else c
                for c in message.tool_calls
            ]
            message = replace(message, tool_calls=calls)
        if max_result_chars and message.role == "tool" and message.content and len(message.content) > max_result_chars:
            message = replace(message, content=message.content[:max_result_chars] + "\n(... cut short)")
        out.append(message)
    return out


def _too_large(exc: Exception) -> bool:
    text = str(exc).lower()
    return "413" in text or "request too large" in text or "context length" in text or "too many tokens" in text


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
