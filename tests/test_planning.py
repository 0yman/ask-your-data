"""Plan, solve, synthesise, verify - driven by a scripted model.

The script is consumed in call order, and every call records which tools it
was offered, so each test can check that the right step ran with the right
tools: the planner with submit_plan only, the synthesiser with final_answer
only, the reviewer with submit_review only, the solvers with the toolbox.
"""

from __future__ import annotations

import pytest

from agent.agent import PortAnalystAgent
from agent.llm import LLMResponse, ToolCall


class Scripted:
    def __init__(self, responses):
        self.name = "scripted"
        self.responses = list(responses)
        self.calls = []  # (tool names, last user message)
        self.on_wait = None

    def complete(self, system, messages, tools):
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        self.calls.append(([t.name for t in tools], user))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def call(name, **arguments):
    return LLMResponse(tool_calls=[ToolCall(name, arguments)])


def plan(*parts):
    return call("submit_plan", sub_questions=list(parts))


def solve(sql, answer):
    return [call("run_sql", sql=sql), call("final_answer", answer=answer)]


def review(ok=True, **extra):
    return call("submit_review", ok=ok, **extra)


@pytest.fixture
def planned(settings):
    return settings.model_copy(update={"plan_questions": True, "verify_answers": True})


def run(settings, warehouse, script, question="q"):
    llm = Scripted(script)
    events = []
    result = PortAnalystAgent(warehouse, llm, settings).ask(question, on_event=events.append)
    return result, llm, events


def test_a_multi_part_question_is_planned_solved_synthesised_and_checked(planned, warehouse):
    result, llm, events = run(planned, warehouse, [
        plan("How many berths are there?", "How many vessel calls are there?"),
        *solve("SELECT COUNT(*) AS berths FROM dim_berth", "There are 3 berths."),
        *solve("SELECT COUNT(*) AS calls FROM fact_vessel_call", "There are 4 calls."),
        call("final_answer", answer="3 berths; 4 vessel calls."),
        review(ok=True),
    ], question="How many berths and how many calls?")

    assert result.answer == "3 berths; 4 vessel calls."
    assert result.stop_reason == "final_answer"
    assert [p["answer"] for p in result.parts] == ["There are 3 berths.", "There are 4 calls."]
    assert result.parts[1]["sql"].startswith("SELECT COUNT(*) AS calls")
    assert result.review["ok"] is True
    offered = [tools for tools, _ in llm.calls]
    assert offered[0] == ["submit_plan"]
    assert offered[-2] == ["final_answer"] and offered[-1] == ["submit_review"]
    # The second part is told what the first found, and what its own part is.
    second_part_prompt = llm.calls[3][1]
    assert "There are 3 berths." in second_part_prompt and "Your part: How many vessel calls" in second_part_prompt
    kinds = [e["type"] for e in events]
    assert kinds[0] == "plan" and kinds.count("part") == 2
    assert "synthesising" in kinds and kinds[-1] == "review"


def test_a_simple_question_is_answered_as_asked(planned, warehouse):
    result, llm, events = run(planned, warehouse, [
        plan("How many berths are in dim_berth?"),
        *solve("SELECT COUNT(*) FROM dim_berth", "There are 3 berths."),
        review(ok=True),
    ], question="How many berths?")
    assert result.answer == "There are 3 berths."
    assert result.parts == []                        # nothing to break down
    assert llm.calls[1][1] == "How many berths?"     # the user's words, not the planner's
    assert "plan" not in [e["type"] for e in events]


def test_no_usable_plan_means_a_single_pass(planned, warehouse):
    result, llm, _ = run(planned, warehouse, [
        LLMResponse(text="I would look at the berths."),   # no plan
        *solve("SELECT COUNT(*) FROM dim_berth", "There are 3 berths."),
        review(ok=True),
    ], question="How many berths?")
    assert result.answer == "There are 3 berths."
    assert llm.calls[1][1] == "How many berths?"


def test_the_reviewer_can_send_a_part_back_once(planned, warehouse):
    result, llm, events = run(planned, warehouse, [
        plan("Count the berths.", "Count the vessel calls with more than 5 waiting hours."),
        *solve("SELECT COUNT(*) FROM dim_berth", "3 berths."),
        *solve("SELECT COUNT(*) FROM fact_vessel_call", "4 calls."),          # forgot the condition
        call("final_answer", answer="3 berths; 4 calls."),
        review(ok=False, problems=["Part 2 ignores the waiting-hours condition."], redo=[2],
               guidance="Filter waiting_hours > 5."),
        *solve("SELECT COUNT(*) FROM fact_vessel_call WHERE waiting_hours > 5", "2 calls."),
        call("final_answer", answer="3 berths; 2 calls waited more than 5 hours."),
    ])
    assert result.answer == "3 berths; 2 calls waited more than 5 hours."
    assert result.parts[1]["answer"] == "2 calls."
    redo_prompt = llm.calls[-3][1]
    assert "ignores the waiting-hours condition" in redo_prompt and "Filter waiting_hours > 5" in redo_prompt
    assert {"type": "redo", "index": 1} in events
    assert len(llm.calls) == 10                      # one review round, no second review


def test_a_single_part_flagged_by_the_reviewer_is_redone(planned, warehouse):
    result, llm, _ = run(planned, warehouse, [
        plan("Count calls that waited more than 5 hours."),
        *solve("SELECT COUNT(*) FROM fact_vessel_call", "4 calls."),
        review(ok=False, problems=["The condition is missing."]),
        *solve("SELECT COUNT(*) FROM fact_vessel_call WHERE waiting_hours > 5", "2 calls."),
    ], question="How many calls waited more than 5 hours?")
    assert result.answer == "2 calls."


def test_a_failed_review_leaves_the_draft_standing(planned, warehouse):
    result, _, _ = run(planned, warehouse, [
        plan("Count the berths."),
        *solve("SELECT COUNT(*) FROM dim_berth", "3 berths."),
        RuntimeError("500 internal error from the host"),
    ])
    assert result.answer == "3 berths."
    assert result.review is None


def test_a_model_out_of_allowance_is_not_hidden(planned, warehouse):
    from agent.llm import ModelLimitReached

    with pytest.raises(ModelLimitReached):
        run(planned, warehouse, [ModelLimitReached("tokens per day (TPD)")])


def test_planning_can_be_turned_off(settings, warehouse):
    result, llm, _ = run(settings, warehouse, solve("SELECT COUNT(*) FROM dim_berth", "3 berths."))
    assert result.answer == "3 berths."
    assert "submit_plan" not in [name for tools, _ in llm.calls for name in tools]


def test_verification_can_run_without_planning(settings, warehouse):
    checked = settings.model_copy(update={"plan_questions": False, "verify_answers": True})
    result, llm, _ = run(checked, warehouse, [
        *solve("SELECT COUNT(*) FROM fact_vessel_call", "4 calls."),
        review(ok=False, problems=["The waiting-hours condition is missing."]),
        *solve("SELECT COUNT(*) FROM fact_vessel_call WHERE waiting_hours > 5", "2 calls."),
    ], question="How many calls waited more than 5 hours?")
    assert result.answer == "2 calls."
    offered = [tools for tools, _ in llm.calls]
    assert ["submit_plan"] not in offered and ["submit_review"] in offered
