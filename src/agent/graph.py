"""The same agent as `PortAnalystAgent._ask_loop`, expressed as a LangGraph.

Each decision the hand-written loop makes is a node, and each branch is a
conditional edge:

    model ─┬─> nudge ──> model            an empty turn gets one nudge
           ├─> text_answer ──> END        prose without the answer tool is accepted
           └─> act ─┬─> END               final answer accepted
                    ├─> model             tool results, or one grounding recheck
                    └─> salvage ──> END   failures in a row, or out of steps

The prompt, tools, guardrails, context compaction, answer check and salvage
are the loop's own functions, so the two engines differ only in how control
flows. tests/test_graph.py runs the same scripted conversations through both
and requires identical results.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import LLMResponse, Message
from .tools import FINAL_ANSWER_TOOL, ToolBox

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    messages: list[Message]
    steps: list[Any]              # agent.Step
    step: Any                     # the step being filled in
    response: LLMResponse | None  # the model's latest turn
    index: int
    usage: dict[str, int]
    failed_attempts: int
    failed_in_a_row: int
    failed_ids: set[str]
    nudged: bool
    rechecked: bool
    answer: str | None
    stop_reason: str
    finished: bool


def ask_graph(agent, question: str, emit: Callable[[dict[str, Any]], None], max_steps: int | None = None):
    """Answer `question` with the LangGraph engine; returns an AgentResult."""
    from .agent import (
        EMPTY_REPLY_NUDGE,
        GROUNDING_NUDGE,
        AgentResult,
        Step,
        _last_error,
        nearest_result_value,
        ungrounded_figures,
    )

    settings = agent.settings
    budget = max_steps or settings.max_steps
    toolbox = ToolBox(warehouse=agent.warehouse, max_rows_to_model=settings.max_rows_to_model)
    started = time.perf_counter()

    def model(state: AgentState) -> AgentState:
        index = state["index"] + 1
        step_started = time.perf_counter()
        emit({"type": "thinking", "step": index})
        response = agent._complete(state["messages"], toolbox.specs, state["failed_ids"])
        usage = dict(state["usage"])
        for key, value in response.usage.items():
            usage[key] = usage.get(key, 0) + value
        step = Step(index=index, text=response.text, usage=response.usage,
                    latency_ms=(time.perf_counter() - step_started) * 1000)
        return {"index": index, "response": response, "step": step, "usage": usage}

    def after_model(state: AgentState) -> str:
        response = state["response"]
        if not response.wants_tools and not response.text and not state["nudged"]:
            return "nudge"
        if not response.wants_tools:
            return "text_answer"
        return "act"

    def nudge(state: AgentState) -> AgentState:
        return {"steps": state["steps"] + [state["step"]], "nudged": True,
                "messages": state["messages"] + [Message(role="user", content=EMPTY_REPLY_NUDGE)]}

    def text_answer(state: AgentState) -> AgentState:
        return {"steps": state["steps"] + [state["step"]], "stop_reason": "text_answer",
                "answer": state["response"].text or "The model returned an empty response."}

    def act(state: AgentState) -> AgentState:
        response, step, index = state["response"], state["step"], state["index"]
        messages = state["messages"] + [Message(role="assistant", content=response.text,
                                                tool_calls=response.tool_calls)]
        failed_attempts, failed_in_a_row = state["failed_attempts"], state["failed_in_a_row"]
        failed_ids, rechecked = set(state["failed_ids"]), state["rechecked"]
        update: AgentState = {}
        for call in response.tool_calls:
            if call.name == FINAL_ANSWER_TOOL:
                draft = str(call.arguments.get("answer", "")).strip()
                loose = [] if rechecked or index == budget else ungrounded_figures(
                    draft, question, toolbox.executed_queries, agent._schema_text or "")
                if loose:
                    rechecked = True
                    logger.warning("Answer figures in no query result: %s - asked to recheck", ", ".join(loose))
                    step.tool_calls.append({"name": call.name, "arguments": call.arguments,
                                            "ok": False, "error_kind": "ungrounded"})
                    named = ", ".join(f"{f} (a result has {near})" if (near := nearest_result_value(
                        f, question, toolbox.executed_queries)) else f for f in loose)
                    messages.append(Message(role="tool", tool_name=call.name, tool_call_id=call.id,
                                            content=GROUNDING_NUDGE.format(figures=named)))
                    emit({"type": "recheck", "figures": loose})
                    break
                step.tool_calls.append({"name": call.name, "arguments": call.arguments, "ok": True})
                update = {"answer": draft, "stop_reason": "final_answer", "finished": True}
                break
            outcome = toolbox.dispatch(call.name, call.arguments)
            if outcome.ok:
                failed_in_a_row = 0
            else:
                failed_attempts += 1
                failed_in_a_row += 1
                failed_ids.add(call.id)
            emit({"type": "tool", "step": index, "name": call.name, "arguments": call.arguments,
                  "ok": outcome.ok, "error_kind": outcome.error_kind,
                  "rows": outcome.result.row_count if outcome.result is not None else None})
            step.tool_calls.append({"name": call.name, "arguments": call.arguments,
                                    "ok": outcome.ok, "error_kind": outcome.error_kind})
            messages.append(Message(role="tool", content=outcome.content, tool_name=call.name,
                                    tool_call_id=call.id))
        return {**update, "messages": messages, "steps": state["steps"] + [step],
                "failed_attempts": failed_attempts, "failed_in_a_row": failed_in_a_row,
                "failed_ids": failed_ids, "rechecked": rechecked}

    def after_act(state: AgentState) -> str:
        if state.get("finished"):
            return END
        if state["failed_in_a_row"] > settings.max_sql_retries:
            return "too_many_failures"
        return "model" if state["index"] < budget else "salvage"

    def after_nudge(state: AgentState) -> str:
        return "model" if state["index"] < budget else "salvage"

    def too_many_failures(state: AgentState) -> AgentState:
        return {"stop_reason": "too_many_failures"}

    def salvage(state: AgentState) -> AgentState:
        steps, usage = list(state["steps"]), dict(state["usage"])
        if toolbox.executed_queries:
            emit({"type": "salvage"})
        salvaged = agent._salvage(state["messages"], toolbox, state["failed_ids"], steps, usage)
        if salvaged is not None:
            return {"answer": salvaged, "stop_reason": "partial_answer", "steps": steps, "usage": usage}
        if state["stop_reason"] == "too_many_failures":
            answer = (f"I could not produce a working query after {state['failed_attempts']} failed attempts. "
                      f"The last error was: {_last_error(state['messages'])}\n\nTry asking about one part "
                      "of the question at a time.")
        else:
            answer = (f"I ran out of steps ({settings.max_steps}) before reaching an answer. "
                      "The question may need to be narrowed.")
        return {"answer": answer, "steps": steps, "usage": usage}

    graph = StateGraph(AgentState)
    for name, node in (("model", model), ("nudge", nudge), ("text_answer", text_answer), ("act", act),
                       ("too_many_failures", too_many_failures), ("salvage", salvage)):
        graph.add_node(name, node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["nudge", "text_answer", "act"])
    graph.add_conditional_edges("nudge", after_nudge, ["model", "salvage"])
    graph.add_edge("text_answer", END)
    graph.add_conditional_edges("act", after_act, [END, "model", "salvage", "too_many_failures"])
    graph.add_edge("too_many_failures", "salvage")
    graph.add_edge("salvage", END)

    final = graph.compile().invoke(
        {"messages": [Message(role="user", content=question)], "steps": [], "index": 0,
         "usage": {"prompt_tokens": 0, "output_tokens": 0}, "failed_attempts": 0, "failed_in_a_row": 0,
         "failed_ids": set(), "nudged": False, "rechecked": False, "answer": None,
         "stop_reason": "max_steps", "finished": False},
        {"recursion_limit": 4 * budget + 20})
    stop_reason = final["stop_reason"]
    return AgentResult(
        question=question, answer=final["answer"], steps=final["steps"],
        sql_queries=[query.sql for query in toolbox.executed_queries], stop_reason=stop_reason,
        succeeded=stop_reason in {"final_answer", "text_answer", "partial_answer"},
        total_latency_ms=(time.perf_counter() - started) * 1000, usage=final["usage"],
        failed_attempts=final["failed_attempts"],
        last_result=toolbox.executed_queries[-1] if toolbox.executed_queries else None)
