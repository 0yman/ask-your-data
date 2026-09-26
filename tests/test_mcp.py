"""The MCP server, driven by a real MCP client over an in-memory connection."""

from __future__ import annotations

import asyncio

from mcp.client import Client

from agent.agent import PortAnalystAgent
from agent.llm import LLMResponse, ScriptedLLM, ToolCall
from agent.mcp_server import build_server


def text(result) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


def talk(server, *calls):
    """Open a client session, then list the tools and make each call in turn."""
    async def go():
        async with Client(server) as client:
            tools = await client.list_tools()
            results = [await client.call_tool(name, arguments) for name, arguments in calls]
            return tools, results
    return asyncio.run(go())


def test_the_data_tools_are_offered_read_only(settings, warehouse):
    tools, _ = talk(build_server(warehouse, settings))
    names = {t.name: t for t in tools.tools}
    assert set(names) == {"list_tables", "describe_table", "sample_rows", "run_sql"}
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint for t in names.values())


def test_a_client_can_explore_and_query(settings, warehouse):
    _, (tables, columns, rows) = talk(
        build_server(warehouse, settings),
        ("list_tables", {}),
        ("describe_table", {"table": "dim_berth"}),
        ("run_sql", {"sql": "SELECT COUNT(*) AS berths FROM dim_berth"}))
    assert "fact_vessel_call" in text(tables) and not tables.is_error
    assert "berth" in text(columns).lower()
    assert "berths" in text(rows) and not rows.is_error


def test_writes_and_unknown_tables_come_back_as_errors(settings, warehouse):
    _, (drop, missing) = talk(
        build_server(warehouse, settings),
        ("run_sql", {"sql": "DROP TABLE dim_berth"}),
        ("describe_table", {"table": "no_such_table"}))
    assert drop.is_error and "rejected" in text(drop).lower()
    assert missing.is_error
    _, (still_there,) = talk(build_server(warehouse, settings), ("list_tables", {}))
    assert "dim_berth" in text(still_there)


def test_ask_answers_with_the_sql_behind_it(settings, warehouse):
    llm = ScriptedLLM([LLMResponse(tool_calls=[ToolCall("run_sql", {"sql": "SELECT COUNT(*) AS calls FROM fact_vessel_call"})]),
                       LLMResponse(tool_calls=[ToolCall("final_answer", {"answer": "There are 4 vessel calls."})])])
    agent = PortAnalystAgent(warehouse, llm, settings)
    tools, (reply,) = talk(build_server(warehouse, settings, agent), ("ask", {"question": "How many calls?"}))
    assert "ask" in {t.name for t in tools.tools}
    assert "There are 4 vessel calls." in text(reply)
    assert "SELECT COUNT(*) AS calls FROM fact_vessel_call" in text(reply)


def test_the_server_runs_over_stdio_as_a_client_would_launch_it(tiny_db):
    """What Claude Desktop does: spawn `python -m agent.mcp_server`, talk over
    stdin/stdout. Anything else printed to stdout would break the protocol."""
    import os
    import sys
    from pathlib import Path

    from mcp.client.stdio import StdioServerParameters

    src = str(Path(__file__).resolve().parents[1] / "src")
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "agent.mcp_server", "--db", str(tiny_db)],
        env={**os.environ, "PYTHONPATH": src, "AGENT_LLM_BACKEND": "scripted", "GOOGLE_API_KEY": ""})

    async def go():
        async with Client(params) as client:
            tools = await client.list_tools()
            rows = await client.call_tool("run_sql", {"sql": "SELECT COUNT(*) AS berths FROM dim_berth"})
            return {t.name for t in tools.tools}, rows
    names, rows = asyncio.run(go())
    assert {"list_tables", "run_sql"} <= names
    assert "berths" in text(rows) and not rows.is_error
