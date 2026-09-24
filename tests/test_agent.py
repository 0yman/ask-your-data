"""Agent loop and tool behaviour.

Every scenario is driven by a scripted LLM, so the behaviour under test is the
loop's - not a model's mood on the day.
"""

from __future__ import annotations

import pytest

from agent.agent import PortAnalystAgent
from agent.llm import LLMResponse, Message, ScriptedLLM, ToolCall
from agent.tools import ToolBox
from agent.warehouse import QueryTimeout, Warehouse


def call(name: str, **arguments) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name, arguments)])


def answer(text: str) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall("final_answer", {"answer": text})])


@pytest.fixture
def agent_factory(warehouse, settings):
    def build(responses):
        return PortAnalystAgent(warehouse, ScriptedLLM(responses), settings)
    return build


class TestToolBox:
    def test_list_tables_names_every_table(self, warehouse):
        content = ToolBox(warehouse).dispatch("list_tables", {}).content
        assert "fact_vessel_call" in content
        assert "dim_berth" in content

    def test_describe_table_lists_columns_with_types(self, warehouse):
        content = ToolBox(warehouse).dispatch("describe_table", {"table": "dim_berth"}).content
        assert "crane_count" in content
        assert "INTEGER" in content

    def test_unknown_table_returns_a_usable_message_not_an_exception(self, warehouse):
        outcome = ToolBox(warehouse).dispatch("describe_table", {"table": "dim_bert"})
        assert not outcome.ok
        assert outcome.error_kind == "unknown_table"
        # The agent can only self-correct if it is told what does exist.
        assert "dim_berth" in outcome.content

    def test_unknown_tool_lists_the_real_ones(self, warehouse):
        outcome = ToolBox(warehouse).dispatch("run_query", {})
        assert not outcome.ok
        assert "run_sql" in outcome.content

    def test_missing_argument_is_reported_not_raised(self, warehouse):
        outcome = ToolBox(warehouse).dispatch("run_sql", {})
        assert not outcome.ok
        assert outcome.error_kind == "bad_arguments"

    def test_run_sql_returns_rows_as_markdown(self, warehouse):
        outcome = ToolBox(warehouse).dispatch(
            "run_sql", {"sql": "SELECT berth_code FROM dim_berth ORDER BY berth_key"}
        )
        assert outcome.ok
        assert "B01" in outcome.content and "| --- |" in outcome.content

    def test_blocked_sql_explains_itself_to_the_agent(self, warehouse):
        outcome = ToolBox(warehouse).dispatch("run_sql", {"sql": "DROP TABLE dim_berth"})
        assert not outcome.ok
        assert outcome.error_kind == "unsafe_sql"
        assert "read-only" in outcome.content

    def test_broken_sql_returns_the_database_error(self, warehouse):
        outcome = ToolBox(warehouse).dispatch(
            "run_sql", {"sql": "SELECT no_such_column FROM dim_berth"}
        )
        assert not outcome.ok
        assert outcome.error_kind == "sql_error"
        assert "no_such_column" in outcome.content

    def test_successful_queries_are_recorded(self, warehouse):
        toolbox = ToolBox(warehouse)
        toolbox.dispatch("run_sql", {"sql": "SELECT 1"})
        toolbox.dispatch("run_sql", {"sql": "DROP TABLE dim_berth"})  # rejected
        assert len(toolbox.executed_queries) == 1

    def test_sample_rows_is_capped(self, warehouse):
        outcome = ToolBox(warehouse).dispatch("sample_rows", {"table": "dim_berth", "limit": 999})
        assert outcome.ok
        assert outcome.result.row_count <= 20


class TestAgentLoop:
    def test_answers_via_final_answer(self, agent_factory):
        result = agent_factory([
            call("run_sql", sql="SELECT COUNT(*) FROM dim_berth"),
            answer("There are 3 berths."),
        ]).ask("How many berths?")

        assert result.succeeded
        assert result.stop_reason == "final_answer"
        assert result.answer == "There are 3 berths."
        assert len(result.sql_queries) == 1

    def test_prose_without_final_answer_is_still_accepted(self, agent_factory):
        result = agent_factory([LLMResponse(text="There are 3 berths.")]).ask("How many berths?")
        assert result.succeeded
        assert result.stop_reason == "text_answer"

    def test_self_corrects_after_a_bad_query(self, agent_factory):
        """The behaviour the whole design exists for: a failure is fed back as
        a message so the next turn can fix it."""
        result = agent_factory([
            call("run_sql", sql="SELECT mph FROM fact_vessel_call"),      # wrong column
            call("describe_table", table="fact_vessel_call"),
            call("run_sql", sql="SELECT AVG(moves_per_hour) FROM fact_vessel_call"),
            answer("The average is 16.25 moves per hour."),
        ]).ask("What is the average productivity?")

        assert result.succeeded
        assert result.failed_attempts == 1
        assert len(result.sql_queries) == 1  # only the working query is kept

    def test_gives_up_after_too_many_failures(self, agent_factory):
        result = agent_factory([call("run_sql", sql="SELECT bad FROM dim_berth")] * 6).ask("x")
        assert not result.succeeded
        assert result.stop_reason == "too_many_failures"
        assert "failed attempts" in result.answer

    def test_step_budget_stops_a_runaway_loop(self, agent_factory):
        result = agent_factory([call("list_tables")] * 20).ask("loop")
        assert not result.succeeded
        assert result.stop_reason == "max_steps"
        assert len(result.steps) == 6  # settings.max_steps

    def test_blocked_sql_does_not_end_the_run(self, agent_factory):
        result = agent_factory([
            call("run_sql", sql="DROP TABLE dim_berth"),
            answer("I cannot modify the warehouse; it is read-only."),
        ]).ask("Delete the berths table")

        assert result.succeeded
        assert result.failed_attempts == 1
        assert result.sql_queries == []

    def test_trace_records_every_tool_call(self, agent_factory):
        result = agent_factory([
            call("list_tables"),
            call("run_sql", sql="SELECT 1"),
            answer("done"),
        ]).ask("q")

        names = [c["name"] for step in result.steps for c in step.tool_calls]
        assert names == ["list_tables", "run_sql", "final_answer"]
        assert "stop_reason: final_answer" in result.format_trace()

    def test_empty_script_does_not_hang(self, agent_factory):
        result = agent_factory([]).ask("q")
        assert result.stop_reason == "text_answer"

    def test_schema_is_in_the_system_prompt(self, agent_factory):
        prompt = agent_factory([]).system_prompt
        assert "fact_vessel_call" in prompt
        assert "moves_per_hour" in prompt

    def test_conversation_alternates_correctly(self, agent_factory):
        """Tool results must go back as tool messages, or the model cannot
        see what its own call returned."""
        agent = agent_factory([call("list_tables"), answer("done")])
        agent.ask("q")
        last_turn = agent.llm.calls[-1]
        assert [m.role for m in last_turn] == ["user", "assistant", "tool"]
        assert last_turn[-1].tool_name == "list_tables"


class TestWarehouse:
    def test_known_aggregate_is_correct(self, warehouse):
        """Hand-checkable: moves_per_hour values are 10, 30, 20, 5."""
        result = warehouse.run_sql("SELECT AVG(moves_per_hour) FROM fact_vessel_call")
        assert result.rows[0][0] == pytest.approx(16.25)

    def test_missing_warehouse_file_is_explicit(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build_warehouse"):
            Warehouse(tmp_path / "absent.duckdb")

    def test_connection_is_read_only_at_the_database_level(self, warehouse):
        """Defence in depth: even bypassing the parser cannot write."""
        with pytest.raises(Exception, match="read.only|read_only|Cannot execute"):
            warehouse._connection.execute("CREATE TABLE evil AS SELECT 1")

    def test_markdown_render_caps_rows(self, warehouse):
        result = warehouse.run_sql("SELECT * FROM fact_vessel_call")
        rendered = result.to_markdown(max_rows=2)
        assert "more rows not shown" in rendered

    def test_empty_result_renders_cleanly(self, warehouse):
        result = warehouse.run_sql("SELECT * FROM dim_berth WHERE berth_key = 999")
        assert result.to_markdown() == "(0 rows)"

    def test_timeout_raises_query_timeout(self, settings, tiny_db):
        """A cross join with no limit is the accidental version of this."""
        wh = Warehouse(tiny_db, timeout_seconds=0.001)
        try:
            with pytest.raises((QueryTimeout, Exception)):
                wh.run_sql(
                    "SELECT COUNT(*) FROM range(100000000) a, range(100) b"
                )
        finally:
            wh.close()


class TestMessages:
    def test_tool_message_carries_its_call_identity(self):
        message = Message(role="tool", content="result", tool_name="run_sql", tool_call_id="abc")
        assert message.tool_name == "run_sql"
        assert message.tool_call_id == "abc"

    def test_tool_calls_get_distinct_ids(self):
        first, second = ToolCall("a", {}), ToolCall("a", {})
        assert first.id != second.id


def test_an_empty_turn_gets_one_nudge_then_the_answer(settings, warehouse):
    from agent.agent import EMPTY_REPLY_NUDGE, PortAnalystAgent
    from agent.llm import LLMResponse, ScriptedLLM, ToolCall

    llm = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("run_sql", {"sql": "SELECT COUNT(*) FROM dim_berth"})]),
        LLMResponse(),  # empty: no text, no tool call
        LLMResponse(tool_calls=[ToolCall("final_answer", {"answer": "There are 3 berths."})]),
    ])
    result = PortAnalystAgent(warehouse, llm, settings).ask("How many berths?")
    assert result.answer == "There are 3 berths."
    assert result.stop_reason == "final_answer"
    assert llm.calls[-1][-1].content == EMPTY_REPLY_NUDGE


def test_a_second_empty_turn_is_not_nudged_forever(settings, warehouse):
    from agent.agent import PortAnalystAgent
    from agent.llm import LLMResponse, ScriptedLLM

    llm = ScriptedLLM([LLMResponse(), LLMResponse(), LLMResponse()])
    result = PortAnalystAgent(warehouse, llm, settings).ask("How many berths?")
    assert result.stop_reason == "text_answer"
    assert len(llm.calls) == 2
