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


class _Recording:
    """A scripted model that also records the tools and messages it was sent."""

    def __init__(self, responses, fail_first_with=None):
        self.name = "recording"
        self.responses = list(responses)
        self.sent = []
        self.fail_first_with = fail_first_with

    def complete(self, system, messages, tools):
        self.sent.append((list(messages), [t.name for t in tools]))
        if self.fail_first_with:
            error, self.fail_first_with = self.fail_first_with, None
            raise RuntimeError(error)
        from agent.llm import LLMResponse
        return self.responses.pop(0) if self.responses else LLMResponse(text="done")


def _sql(query):
    from agent.llm import LLMResponse, ToolCall
    return LLMResponse(tool_calls=[ToolCall("run_sql", {"sql": query})])


BROKEN = "SELECT nope\nFROM dim_berth"


class TestNoWorkIsThrownAway:
    def test_after_too_many_failures_it_answers_from_the_queries_that_worked(self, settings, warehouse):
        from agent.agent import SALVAGE_NUDGE, PortAnalystAgent
        from agent.llm import LLMResponse, ToolCall

        llm = _Recording([
            _sql("SELECT COUNT(*) AS berths FROM dim_berth"),
            *[_sql(BROKEN) for _ in range(settings.max_sql_retries + 1)],
            LLMResponse(tool_calls=[ToolCall("final_answer", {"answer": "3 berths; the rest could not be computed."})]),
        ])
        result = PortAnalystAgent(warehouse, llm, settings).ask("Tell me everything about berths")
        assert result.stop_reason == "partial_answer"
        assert result.succeeded
        assert result.answer.startswith("3 berths")
        last_messages, last_tools = llm.sent[-1]
        assert last_tools == ["final_answer"]           # no more queries allowed
        assert last_messages[-1].content == SALVAGE_NUDGE

    def test_with_nothing_that_worked_it_says_so_and_suggests_splitting(self, settings, warehouse):
        from agent.agent import PortAnalystAgent

        llm = _Recording([_sql(BROKEN) for _ in range(settings.max_sql_retries + 1)])
        result = PortAnalystAgent(warehouse, llm, settings).ask("q")
        assert result.stop_reason == "too_many_failures"
        assert not result.succeeded
        assert "one part of the question at a time" in result.answer


class TestContextStaysSmall:
    def test_older_failed_queries_are_replaced_but_the_newest_stays(self, settings, warehouse):
        from agent.agent import FAILED_SQL_PLACEHOLDER, PortAnalystAgent

        llm = _Recording([_sql(BROKEN + " -- first"), _sql(BROKEN + " -- second")])
        PortAnalystAgent(warehouse, llm, settings).ask("q")
        third_call_messages = llm.sent[2][0]
        sqls = [c.arguments["sql"] for m in third_call_messages if m.role == "assistant" for c in m.tool_calls]
        assert sqls == [FAILED_SQL_PLACEHOLDER, BROKEN + " -- second"]

    def test_a_request_too_large_is_retried_once_trimmed(self, settings, warehouse):
        from agent.agent import PortAnalystAgent
        from agent.llm import LLMResponse, ToolCall

        llm = _Recording(
            [LLMResponse(tool_calls=[ToolCall("final_answer", {"answer": "ok"})])],
            fail_first_with="Error code: 413 - Request too large for model",
        )
        result = PortAnalystAgent(warehouse, llm, settings).ask("q")
        assert result.answer == "ok"
        assert len(llm.sent) == 2

    def test_a_wide_result_is_cut_to_fit(self, warehouse):
        from agent.tools import MAX_RESULT_CHARS, ToolBox

        wide = ", ".join(f"repeat('x', 30) AS c{i}" for i in range(40))
        outcome = ToolBox(warehouse=warehouse, max_rows_to_model=30).dispatch(
            "run_sql", {"sql": f"SELECT {wide} FROM range(50)"}
        )
        assert outcome.ok
        assert len(outcome.content) < MAX_RESULT_CHARS + 400
        assert "select fewer columns" in outcome.content or "cut short" in outcome.content

    def test_a_long_failing_query_is_told_to_split_the_work(self, warehouse):
        from agent.tools import ToolBox

        long_sql = "SELECT\n" + "\n".join(f"  nope_{i}," for i in range(20)) + "\n  1\nFROM dim_berth"
        outcome = ToolBox(warehouse=warehouse, max_rows_to_model=30).dispatch("run_sql", {"sql": long_sql})
        assert not outcome.ok
        assert "Split the question into smaller queries" in outcome.content
        short = ToolBox(warehouse=warehouse, max_rows_to_model=30).dispatch("run_sql", {"sql": BROKEN})
        assert "Split" not in short.content


def test_failures_spread_across_sub_questions_do_not_stop_the_run(settings, warehouse):
    """A broad question: each sub-query fails once and is fixed. Four
    failures in total, never more than one in a row - so it carries on."""
    from agent.agent import PortAnalystAgent
    from agent.llm import LLMResponse, ToolCall

    good = "SELECT COUNT(*) FROM dim_berth"
    script = []
    for _ in range(settings.max_sql_retries + 1):
        script += [_sql(BROKEN), _sql(good)]
    script.append(LLMResponse(tool_calls=[ToolCall("final_answer", {"answer": "done"})]))
    llm = _Recording(script)
    result = PortAnalystAgent(warehouse, llm, settings.model_copy(update={"max_steps": 20}), ).ask("q")
    assert result.stop_reason == "final_answer"
    assert result.failed_attempts == settings.max_sql_retries + 1


def test_failures_in_a_row_still_stop_it(settings, warehouse):
    from agent.agent import PortAnalystAgent

    llm = _Recording([_sql("SELECT 1 AS ok")] + [_sql(BROKEN) for _ in range(settings.max_sql_retries + 1)])
    result = PortAnalystAgent(warehouse, llm, settings).ask("q")
    assert result.stop_reason == "partial_answer"   # stopped, then answered from the query that worked


class TestCurrencyTheDataNeverStated:
    """A "$" in front of a figure when nothing in the data says dollars is a
    wrong fact, not a style choice. It comes off; a stated currency stays."""

    SALES = "sales (3 rows): Country VARCHAR, Quantity BIGINT, UnitPrice DOUBLE"

    @pytest.mark.parametrize(("written", "shown"), [
        ("Revenue was $284,661.54.", "Revenue was 284,661.54."),
        ("**£8,998,790.91** in 2011", "**8,998,790.91** in 2011"),
        ("a loss of -€5 and $ 12", "a loss of -5 and 12"),
        ("No figures here.", "No figures here."),
    ])
    def test_unstated_symbols_come_off(self, written, shown):
        from agent.agent import drop_unstated_currency

        assert drop_unstated_currency(written, "What was the revenue?", self.SALES) == shown

    @pytest.mark.parametrize(("question", "schema"), [
        ("What was the revenue?", "fees (1 rows): demurrage_usd DOUBLE"),
        ("What was the revenue, in dollars?", SALES),
        ("How many orders were over £100?", SALES),
    ])
    def test_a_stated_currency_is_left_alone(self, question, schema):
        from agent.agent import drop_unstated_currency

        assert drop_unstated_currency("It was $120.", question, schema) == "It was $120."

    def test_the_port_data_names_its_currency(self, agent_factory):
        # fact_container_movement has demurrage_usd.
        result = agent_factory([call("run_sql", sql="SELECT 1250 AS demurrage_usd"),
                                answer("Demurrage came to $1,250.")]).ask("How much demurrage?")
        assert result.answer == "Demurrage came to $1,250."

    def test_data_without_a_currency_gets_plain_figures(self, settings, tmp_path):
        import duckdb

        path = tmp_path / "sales.duckdb"
        with duckdb.connect(str(path)) as con:
            con.execute("CREATE TABLE sales AS SELECT 'Netherlands' AS Country, 2 AS Quantity, 1.5 AS UnitPrice")
        warehouse = Warehouse(path, max_rows=100, timeout_seconds=5)
        try:
            agent = PortAnalystAgent(warehouse, ScriptedLLM([answer("The Netherlands: $3.00.")]), settings, domain="user")
            assert agent.ask("Which country spent most?").answer == "The Netherlands: 3.00."
        finally:
            warehouse.close()


class TestVoting:
    """Self-consistency: several runs, and the answer most of them agree on."""

    @staticmethod
    def result(text, succeeded=True):
        from agent.agent import AgentResult

        return AgentResult(question="q", answer=text, steps=[], sql_queries=[],
                           stop_reason="final_answer" if succeeded else "max_steps",
                           succeeded=succeeded, total_latency_ms=0.0, usage={})

    def test_the_majority_wins_and_rounding_does_not_split_it(self):
        from agent.agent import pick_by_vote

        runs = [self.result("Total: 100."), self.result("Total: 284,661.54."), self.result("About 284,662 in all.")]
        assert pick_by_vote(runs) == (1, 2)

    def test_a_tie_goes_to_the_first_run(self):
        from agent.agent import pick_by_vote

        assert pick_by_vote([self.result("1 berth."), self.result("2 berths."), self.result("3 berths.")]) == (0, 1)

    def test_two_declines_agree(self):
        from agent.agent import pick_by_vote

        runs = [self.result("The data has no rice-farming column."), self.result("Egypt: 46.5."),
                self.result("That is not in the data.")]
        assert pick_by_vote(runs) == (0, 2)

    def test_runs_that_did_not_finish_do_not_vote(self):
        from agent.agent import pick_by_vote

        runs = [self.result("7 calls.", succeeded=False), self.result("9 calls."), self.result("7 calls.", succeeded=False)]
        assert pick_by_vote(runs) == (1, 1)

    def test_three_runs_in_parallel_keep_the_majority_answer(self, settings, warehouse):
        count = "SELECT COUNT(*) AS calls FROM fact_vessel_call"
        first = ScriptedLLM([call("run_sql", sql=count), answer("There are 250 calls.")])
        others = iter([ScriptedLLM([call("run_sql", sql=count), answer("There are 4 calls.")]),
                       ScriptedLLM([call("run_sql", sql=count), answer("4 calls in all.")])])
        temperatures = []

        def factory(warm):
            temperatures.append(warm.temperature)
            return next(others)

        voted = settings.model_copy(update={"vote_runs": 3})
        agent = PortAnalystAgent(warehouse, first, voted, llm_factory=factory)
        events = []
        result = agent.ask("How many vessel calls?", on_event=events.append)
        assert result.answer == "There are 4 calls."
        assert result.votes == {"runs": 3, "agreeing": 2}
        assert temperatures == [voted.vote_temperature] * 2
        assert {"type": "vote", "runs": 3, "agreeing": 2} in events

    def test_one_run_is_the_default(self, agent_factory):
        result = agent_factory([answer("3 berths.")]).ask("How many berths?")
        assert result.votes is None

    def test_a_run_lost_to_the_host_does_not_sink_the_question(self, settings, warehouse):
        class RateLimited(ScriptedLLM):
            def complete(self, system, messages, tools):
                raise RuntimeError("Error code: 429 - Rate limit exceeded")

        others = iter([ScriptedLLM([answer("4 calls.")]), ScriptedLLM([answer("There are 4 calls.")])])
        voted = settings.model_copy(update={"vote_runs": 3, "max_retries": 0})
        agent = PortAnalystAgent(warehouse, RateLimited([]), voted, llm_factory=lambda warm: next(others))
        result = agent.ask("How many vessel calls?")
        assert result.answer == "4 calls."
        assert result.votes == {"runs": 2, "agreeing": 2}

    def test_every_run_lost_is_the_question_lost(self, settings, warehouse):
        class RateLimited(ScriptedLLM):
            def complete(self, system, messages, tools):
                raise RuntimeError("Error code: 429 - Rate limit exceeded")

        voted = settings.model_copy(update={"vote_runs": 2})
        agent = PortAnalystAgent(warehouse, RateLimited([]), voted, llm_factory=lambda warm: RateLimited([]))
        with pytest.raises(RuntimeError, match="429"):
            agent.ask("How many vessel calls?")


class TestGroupValuesInTheSchema:
    """A 'World' row among the countries is named in the schema; a product
    that merely starts with WORLD is not, and a clean table adds nothing."""

    def summary(self, tmp_path, create):
        import duckdb

        path = tmp_path / "t.duckdb"
        with duckdb.connect(str(path)) as con:
            con.execute(create)
        warehouse = Warehouse(path, max_rows=100, timeout_seconds=5)
        try:
            return warehouse.schema_summary()
        finally:
            warehouse.close()

    def test_group_rows_are_listed_under_the_table(self, tmp_path):
        text = self.summary(tmp_path, "CREATE TABLE co2 AS SELECT * FROM (VALUES ('Egypt', 1.0), ('World', 9.0), "
                                      "('Asia', 5.0), ('High-income countries', 4.0)) AS t(country, co2)")
        note = text.splitlines()[1]
        assert note.startswith("  - country also holds 3 group values")
        assert "'World', 'Asia', 'High-income countries'" in note and "Egypt" not in note

    def test_a_product_named_world_is_not_a_group(self, tmp_path):
        text = self.summary(tmp_path, "CREATE TABLE items AS SELECT 'WORLD WAR 2 GLIDERS' AS Description, 2 AS Quantity")
        assert text == "items (1 rows): Description VARCHAR, Quantity INTEGER"


class TestAnswerFiguresAreGrounded:
    """A figure in the answer must come from a result, a query or the
    question - within rounding and a change of unit - or it is sent back once."""

    @staticmethod
    def result(rows, sql="SELECT 1"):
        from agent.warehouse import QueryResult

        return QueryResult(sql=sql, columns=["v"], rows=rows, row_count=len(rows), truncated=False, elapsed_ms=1.0)

    @pytest.mark.parametrize("answer_text", [
        "Revenue was 284,661.54.",
        "About 284,662 in all.",
        "Asia emitted 22.2 billion tonnes.",          # 22,237.81 million
        "That is 49.3% of the total.",                 # 0.4934
        "It fell by 967.8 tonnes.",                    # -967.8
        "In 2011, 3 countries led.",                   # a year and a small count
        "StockCode 23843 sold most.",                  # a number inside a text value
    ])
    def test_figures_from_the_results_pass(self, answer_text):
        from agent.agent import ungrounded_figures

        results = [self.result([(284661.54,), (22237.81,), (0.4934,), (-967.8,), ("23843",)])]
        assert ungrounded_figures(answer_text, "Which country?", results) == []

    def test_a_figure_from_nowhere_is_named(self):
        from agent.agent import ungrounded_figures

        results = [self.result([(284661.54,)])]
        assert ungrounded_figures("Revenue was 284,661.54, up 12.5% on 2010.", "q", results) == ["12.5"]
        assert ungrounded_figures("Over 100 million people.", "Countries over 100 million?", []) == []

    def test_an_ungrounded_answer_is_sent_back_once(self, agent_factory):
        agent = agent_factory([
            call("run_sql", sql="SELECT COUNT(*) AS calls FROM fact_vessel_call"),
            answer("There are 40 calls, 12.5% of them late."),
            answer("There are 4 calls."),
        ])
        events = []
        result = agent.ask("How many vessel calls?", on_event=events.append)
        assert result.answer == "There are 4 calls."
        assert {"type": "recheck", "figures": ["40", "12.5"]} in events
        rechecked = [c for s in result.steps for c in s.tool_calls if c.get("error_kind") == "ungrounded"]
        assert len(rechecked) == 1

    def test_the_second_answer_stands_even_if_still_loose(self, agent_factory):
        agent = agent_factory([answer("There are 40 calls."), answer("There are 41 calls.")])
        assert agent.ask("How many vessel calls?").answer == "There are 41 calls."


def test_a_multi_row_scalar_subquery_gets_a_way_out(warehouse):
    outcome = ToolBox(warehouse=warehouse).dispatch(
        "run_sql", {"sql": "SELECT (SELECT berth_code FROM dim_berth) AS b"})
    assert not outcome.ok
    assert "Match it to the outer row" in outcome.content


def test_a_miscopied_figure_is_shown_the_value_it_came_from(agent_factory):
    from agent.agent import nearest_result_value
    from agent.warehouse import QueryResult

    results = [QueryResult(sql="SELECT 1", columns=["revenue"], rows=[(10644560.42,), (9747747.93,)],
                           row_count=2, truncated=False, elapsed_ms=1.0)]
    assert nearest_result_value("1,064,456.42", "q", results) == "10,644,560.42"
    assert nearest_result_value("3,142,905.42", "q", results) is None     # nothing close: no guess
    assert nearest_result_value("12.5", "q", results) is None             # too short to judge

    llm_messages = []
    agent = agent_factory([
        call("run_sql", sql="SELECT 10644560.42 AS revenue"),
        answer("Revenue was 1,064,456.42."),
        answer("Revenue was 10,644,560.42."),
    ])
    result = agent.ask("What was the revenue?")
    llm_messages = agent.llm.calls[-1]
    assert "1,064,456.42 (a result has 10,644,560.42)" in llm_messages[-1].content
    assert result.answer == "Revenue was 10,644,560.42."
