"""The LangGraph engine must be the hand-written loop, re-expressed.

Every scenario the loop handles is scripted once and run through both
engines; the answer, stop reason, every step, every tool call and every event
the page hears must match exactly.
"""

from __future__ import annotations

import pytest

from agent.agent import PortAnalystAgent
from agent.llm import LLMResponse, ScriptedLLM, ToolCall

GOOD = "SELECT COUNT(*) AS calls FROM fact_vessel_call"
BROKEN = "SELECT nope FROM fact_vessel_call"


def call(name, **arguments):
    return LLMResponse(tool_calls=[ToolCall(name, arguments, id=f"{name}-{len(str(arguments))}")])


def sql(query):
    return call("run_sql", sql=query, purpose="count")


def answer(text):
    return call("final_answer", answer=text)


def scenarios(retries):
    salvage_answer = answer("From what worked: 4 calls.")
    return {
        "query then answer": [sql(GOOD), answer("There are 4 calls.")],
        "prose answer": [LLMResponse(text="There are 3 berths.")],
        "empty turn, nudged, then answer": [LLMResponse(), answer("3 berths.")],
        "two empty turns": [LLMResponse(), LLMResponse()],
        "failures fixed along the way": [sql(BROKEN), sql(BROKEN), sql(GOOD), answer("4 calls.")],
        "too many failures in a row, salvaged": [sql(GOOD)] + [sql(BROKEN)] * (retries + 1) + [salvage_answer],
        "too many failures, nothing to salvage": [sql(BROKEN)] * (retries + 1),
        "out of steps with nothing run": [call("list_tables")] * 10,
        "out of steps after a query, salvaged": [sql(GOOD)] + [call("list_tables")] * 10 + [salvage_answer],
        "answer sent back once for a figure from nowhere": [sql(GOOD), answer("There are 40 calls."),
                                                           answer("There are 4 calls.")],
        "two tools in one turn": [LLMResponse(tool_calls=[ToolCall("describe_table", {"table": "dim_berth"}, id="a"),
                                                          ToolCall("run_sql", {"sql": GOOD}, id="b")]),
                                  answer("4 calls.")],
    }


def run(settings, warehouse, engine, script):
    events = []
    agent = PortAnalystAgent(warehouse, ScriptedLLM(list(script)), settings.model_copy(update={"engine": engine}))
    result = agent.ask("How many vessel calls?", on_event=events.append)
    return {
        "answer": result.answer, "stop_reason": result.stop_reason, "succeeded": result.succeeded,
        "failed_attempts": result.failed_attempts, "sql_queries": result.sql_queries,
        "steps": [(s.index, s.text, s.tool_calls, s.usage) for s in result.steps],
        "last_rows": result.last_result.rows if result.last_result else None,
        "events": events,
    }


@pytest.mark.parametrize("name", list(scenarios(2)))
def test_both_engines_give_identical_results(settings, warehouse, name):
    script = scenarios(settings.max_sql_retries)[name]
    loop = run(settings, warehouse, "loop", script)
    graph = run(settings, warehouse, "langgraph", script)
    assert graph == loop


def test_the_scenarios_cover_every_way_a_question_ends(settings, warehouse):
    ends = {run(settings, warehouse, "langgraph", script)["stop_reason"]
            for script in scenarios(settings.max_sql_retries).values()}
    assert ends == {"final_answer", "text_answer", "partial_answer", "too_many_failures", "max_steps"}


def test_the_engine_is_a_setting(settings):
    assert settings.engine == "loop"
    assert settings.model_copy(update={"engine": "langgraph"}).engine == "langgraph"
