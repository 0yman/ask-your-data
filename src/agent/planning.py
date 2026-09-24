"""Plan, solve, synthesise, verify: how the agent takes on a big question.

Better instructions alone did not make a 14B model better at multi-part
questions - measured, three runs each way (see the README). The limit was
not what the model was told but how much it had to hold at once: one long
conversation carrying every part of the question, every query and every
result. So the work is split in code:

1. **Plan.** One call, no queries: break the question into self-contained
   sub-questions, each spelling out its own conditions. A simple question
   comes back as one step and costs one extra call, nothing more.
2. **Solve.** Each sub-question runs through the ordinary agent loop in a
   fresh, short conversation, told only what earlier parts found.
3. **Synthesise.** One call writes the answer to the whole question from the
   parts' answers alone.
4. **Verify.** An independent call reads the question, each part's final SQL
   and the draft answer, and checks that every condition in the question is
   in the SQL - not just in the prose - and that every part is answered. A
   part it flags is solved once more with its findings, then the answer is
   rewritten. One round, so a picky reviewer cannot run up the bill.

Every step degrades to the one before it: no usable plan means a single
pass, a failed synthesis means the parts' answers joined, a failed review
means the draft stands.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .llm import Message, ToolSpec
from .tools import FINAL_ANSWER_TOOL, TOOL_SPECS

if TYPE_CHECKING:
    from .agent import AgentResult, PortAnalystAgent, Step

logger = logging.getLogger(__name__)

Emit = Callable[[dict[str, Any]], None]

PLAN_TOOL = ToolSpec(
    name="submit_plan",
    description="Submit the plan: the sub-questions that together answer the user's question.",
    parameters={
        "type": "object",
        "properties": {
            "sub_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One entry per sub-question, in the order to solve them.",
            },
        },
        "required": ["sub_questions"],
    },
)

REVIEW_TOOL = ToolSpec(
    name="submit_review",
    description="Submit the review of the draft answer.",
    parameters={
        "type": "object",
        "properties": {
            "ok": {"type": "boolean", "description": "True if the answer is correct and complete."},
            "problems": {"type": "array", "items": {"type": "string"},
                         "description": "Each concrete problem found."},
            "redo": {"type": "array", "items": {"type": "integer"},
                     "description": "Numbers of the parts to solve again (1 = first part)."},
            "guidance": {"type": "string", "description": "How to fix them."},
        },
        "required": ["ok"],
    },
)

FINAL_ANSWER_SPEC = next(spec for spec in TOOL_SPECS if spec.name == FINAL_ANSWER_TOOL)

PLAN_TASK = """## Your task now: plan, do not query

Break the user's question into the smallest set of sub-questions that answer it completely, then call `submit_plan` once.

- Each sub-question must be answerable with one or two SQL queries on the schema above.
- Each must spell out every condition that applies to it - dates, filters, exclusions, what counts as what - so it can be solved without seeing the original question.
- Make them self-contained in SQL: if a later part depends on an earlier one, restate the earlier condition inside it instead of relying on a list of values found before.
- If the question is simple enough for one query, return exactly one sub-question: the question itself, made explicit.
- If part of the question cannot be answered from these tables, still list it; it will be reported as missing.
- At most {max_parts} sub-questions."""

SYNTH_TASK = """## Your task now: write the final answer

The user's question was split into parts, and each part was answered from the data. Using only those answers, call `final_answer` with the answer to the whole question: every part answered with its figures, one short line per part for a multi-part question, the assumptions that were stated, and a plain statement of anything that could not be answered. Do not add figures that are not in the parts' answers."""

REVIEW_TASK = """## Your task now: review, strictly

You are checking another analyst's work before it reaches the user. Check:
1. Every condition in the question - dates, filters, exclusions, "first", "only", "excluding", "after", units - is implemented in the SQL of the part that needs it, not just mentioned in the prose.
2. Each part's SQL computes what that part asks: the right table and grain, correct joins with no double counting, total or group rows excluded when entities are asked for.
3. The draft answers every part of the question, and its figures match the parts' answers.
Then call `submit_review`: ok=true if all three hold. Otherwise list each concrete problem, the numbers of the parts to solve again, and exact guidance for fixing them. Do not flag style, wording, or reasonable interpretations that the answer states. A trailing `LIMIT 1000` is added to every query by the system - ignore it."""


@dataclass(slots=True)
class Part:
    question: str
    result: AgentResult

    @property
    def sql(self) -> str | None:
        return self.result.final_sql


def ask_planned(agent: PortAnalystAgent, question: str, emit: Emit) -> AgentResult:
    """Answer `question` by plan, solve, synthesise, verify."""
    settings = agent.settings
    started = time.perf_counter()
    extra_steps: list[Step] = []
    usage: dict[str, int] = {}

    plan = _plan(agent, question, extra_steps, usage) if settings.plan_questions else None
    if not plan or len(plan) == 1:
        # A simple question: solve it as asked. The planner's rewording is
        # not used - it can only lose something the user said.
        parts = [Part(question, agent._ask(question, emit))]
    else:
        emit({"type": "plan", "parts": plan})
        parts = []
        for index, sub_question in enumerate(plan):
            emit({"type": "part", "index": index, "question": sub_question})
            result = agent._ask(_part_prompt(question, parts, sub_question), emit,
                                max_steps=settings.part_max_steps)
            parts.append(Part(sub_question, result))

    answer = _answer(agent, question, parts, emit, extra_steps, usage)

    review: dict[str, Any] | None = None
    if settings.verify_answers and answer:
        emit({"type": "reviewing"})
        review = _review(agent, question, parts, answer, extra_steps, usage)
        if review is not None:
            emit({"type": "review", "ok": review["ok"], "problems": review["problems"]})
        if review is not None and not review["ok"] and review["redo"]:
            for number in review["redo"][:2]:
                index = number - 1
                emit({"type": "redo", "index": index})
                redo = agent._ask(_redo_prompt(question, parts, index, review), emit,
                                  max_steps=settings.part_max_steps)
                parts[index] = Part(parts[index].question, redo)
            answer = _answer(agent, question, parts, emit, extra_steps, usage)

    return _merge(question, answer, parts, extra_steps, usage, review, started)


# --- the four steps ----------------------------------------------------------


def _plan(agent: PortAnalystAgent, question: str, steps: list[Step], usage: dict[str, int]) -> list[str] | None:
    task = PLAN_TASK.format(max_parts=agent.settings.max_parts)
    response = _call(agent, task, [Message(role="user", content=question)], PLAN_TOOL, steps, usage)
    if response is None:
        return None
    raw: Any = None
    for call in response.tool_calls:
        if call.name == PLAN_TOOL.name:
            raw = call.arguments.get("sub_questions")
    if raw is None and response.text:
        try:
            raw = json.loads(response.text).get("sub_questions")
        except (ValueError, AttributeError):
            raw = None
    if not isinstance(raw, list):
        return None
    plan = [str(item).strip() for item in raw if str(item).strip()]
    return plan[: agent.settings.max_parts] or None


def _answer(agent: PortAnalystAgent, question: str, parts: list[Part], emit: Emit,
            steps: list[Step], usage: dict[str, int]) -> str:
    if len(parts) == 1:
        return parts[0].result.answer
    emit({"type": "synthesising"})
    summary = "\n\n".join(
        f"{n}. {part.question}\n   Answer: {part.result.answer}"
        + ("" if part.result.succeeded else "\n   (This part could not be completed.)")
        for n, part in enumerate(parts, start=1)
    )
    prompt = f"The user's question:\n{question}\n\nThe parts and their answers:\n\n{summary}"
    response = _call(agent, SYNTH_TASK, [Message(role="user", content=prompt)], FINAL_ANSWER_SPEC, steps, usage)
    if response is not None:
        for call in response.tool_calls:
            if call.name == FINAL_ANSWER_TOOL and str(call.arguments.get("answer", "")).strip():
                return str(call.arguments["answer"]).strip()
        if response.text:
            return response.text.strip()
    # No synthesis: the parts' own answers, in order, are still the answer.
    return "\n".join(f"{n}. {part.result.answer}" for n, part in enumerate(parts, start=1))


def _review(agent: PortAnalystAgent, question: str, parts: list[Part], answer: str,
            steps: list[Step], usage: dict[str, int]) -> dict[str, Any] | None:
    work = "\n\n".join(
        f"Part {n}: {part.question}\n   Final SQL: {part.sql or '(no query ran)'}\n   Answer: {part.result.answer}"
        for n, part in enumerate(parts, start=1)
    )
    prompt = f"Question:\n{question}\n\n{work}\n\nDraft final answer:\n{answer}"
    response = _call(agent, REVIEW_TASK, [Message(role="user", content=prompt)], REVIEW_TOOL, steps, usage)
    if response is None:
        return None
    for call in response.tool_calls:
        if call.name == REVIEW_TOOL.name:
            args = call.arguments
            redo = []
            for value in args.get("redo") or []:
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if 1 <= number <= len(parts) and number not in redo:
                    redo.append(number)
            ok = bool(args.get("ok", True))
            if not ok and not redo and len(parts) == 1:
                redo = [1]  # one part: nothing else it could mean
            return {
                "ok": ok,
                "problems": [str(p) for p in args.get("problems") or []][:5],
                "redo": redo,
                "guidance": str(args.get("guidance") or ""),
            }
    return None


# --- prompts for the solver ----------------------------------------------------


def _part_prompt(question: str, done: list[Part], sub_question: str) -> str:
    found = "".join(
        f"\n- Part {n}: {part.question}\n  Found: {part.result.answer}" for n, part in enumerate(done, start=1)
    )
    context = f"\n\nAlready found:{found}" if done else ""
    return (
        f"You are answering one part of a larger question.\n\n"
        f"The larger question, for context only: {question}{context}\n\n"
        f"Your part: {sub_question}\n\n"
        f"Answer only your part, with its figures, and call final_answer."
    )


def _redo_prompt(question: str, parts: list[Part], index: int, review: dict[str, Any]) -> str:
    part = parts[index]
    problems = "\n".join(f"- {p}" for p in review["problems"]) or "- (not specified)"
    base = question if len(parts) == 1 else _part_prompt(question, parts[:index], part.question)
    return (
        f"{base}\n\n"
        f"A reviewer checked an earlier attempt at this and found problems:\n{problems}\n"
        f"Guidance: {review['guidance'] or 'fix the problems above'}\n"
        f"The earlier attempt's final SQL was:\n{part.sql or '(none)'}\n\n"
        f"Solve it again, correctly, and call final_answer."
    )


# --- plumbing --------------------------------------------------------------------


def _call(agent: PortAnalystAgent, task: str, messages: list[Message], tool: ToolSpec,
          steps: list[Step], usage: dict[str, int]):
    """One tool-forced model call on the agent's own prompt plus a task. A
    failure here is logged and returns None: every step has a fallback."""
    from .agent import Step

    started = time.perf_counter()
    try:
        response = agent._complete(messages, [tool], set(), system=f"{agent.system_prompt}\n\n{task}")
    except Exception as exc:
        logger.warning("%s call failed: %s", tool.name, exc)
        if "limit" in type(exc).__name__.lower():
            raise  # a model out of allowance must reach the caller, not be hidden
        return None
    for key, value in response.usage.items():
        usage[key] = usage.get(key, 0) + value
    steps.append(Step(
        index=0, text=response.text, latency_ms=(time.perf_counter() - started) * 1000, usage=response.usage,
        tool_calls=[{"name": c.name, "arguments": c.arguments, "ok": True} for c in response.tool_calls],
    ))
    return response


def _merge(question: str, answer: str, parts: list[Part], extra_steps: list[Step], usage: dict[str, int],
           review: dict[str, Any] | None, started: float) -> AgentResult:
    from .agent import AgentResult

    steps = [step for part in parts for step in part.result.steps] + extra_steps
    for number, step in enumerate(steps, start=1):
        step.index = number
    total = dict(usage)
    for part in parts:
        for key, value in part.result.usage.items():
            total[key] = total.get(key, 0) + value
    stopped_short = [p for p in parts if p.result.stop_reason not in ("final_answer", "text_answer")]
    if len(parts) == 1:
        stop_reason = parts[0].result.stop_reason
    else:
        stop_reason = "partial_answer" if stopped_short else "final_answer"
    results = [p.result.last_result for p in parts if p.result.last_result is not None]
    return AgentResult(
        question=question,
        answer=answer,
        steps=steps,
        sql_queries=[sql for part in parts for sql in part.result.sql_queries],
        stop_reason=stop_reason,
        succeeded=bool(answer) and stop_reason in ("final_answer", "text_answer", "partial_answer"),
        total_latency_ms=(time.perf_counter() - started) * 1000,
        usage=total,
        failed_attempts=sum(p.result.failed_attempts for p in parts),
        last_result=results[-1] if results else None,
        parts=[{"question": p.question, "answer": p.result.answer, "sql": p.sql,
                "result": p.result.last_result, "stop_reason": p.result.stop_reason} for p in parts]
        if len(parts) > 1 else [],
        review=review,
    )
