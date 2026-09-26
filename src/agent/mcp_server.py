"""ask-your-data as an MCP server.

Any MCP client - Claude Desktop, Cursor, an IDE agent - can explore a DuckDB
database through the same read-only tools the agent uses, behind the same
guardrails: one SELECT per call, no file access, a row limit, a timeout, and a
connection opened read-only. With a model key configured it also offers
`ask`, the agent's own answer with the SQL behind it.

    python -m agent.mcp_server                      # the sample port warehouse
    python -m agent.mcp_server --dataset retail     # a bundled public dataset
    python -m agent.mcp_server --db path/to/file.duckdb

The server speaks MCP over stdio, so nothing may print to stdout; logging goes
to stderr.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .agent import PortAnalystAgent, build_agent
from .config import Settings, get_settings
from .tools import ToolBox
from .warehouse import Warehouse, set_memory_limit

INSTRUCTIONS = (
    "Read-only SQL over one DuckDB database. Start with list_tables, look at a table with "
    "describe_table and sample_rows before filtering on it, then run one SELECT at a time with "
    "run_sql. Anything that writes, attaches files or reads from disk is refused."
)
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


def build_server(warehouse: Warehouse, settings: Settings, agent: PortAnalystAgent | None = None) -> MCPServer:
    toolbox = ToolBox(warehouse=warehouse, max_rows_to_model=settings.max_rows_to_model)
    server = MCPServer(name="ask-your-data", instructions=INSTRUCTIONS)

    def run(name: str, arguments: dict) -> str:
        outcome = toolbox.dispatch(name, arguments)
        if not outcome.ok:
            # ToolError reaches the client as an error result with this message;
            # any other exception would arrive as a bare "Error executing tool".
            raise ToolError(outcome.content)
        return outcome.content

    @server.tool(annotations=READ_ONLY)
    def list_tables() -> str:
        """Every table in the database, with its row count."""
        return run("list_tables", {})

    @server.tool(annotations=READ_ONLY)
    def describe_table(table: str) -> str:
        """A table's columns and their types."""
        return run("describe_table", {"table": table})

    @server.tool(annotations=READ_ONLY)
    def sample_rows(table: str, limit: int = 5) -> str:
        """A few rows of a table (at most 20), to see what its values look like."""
        return run("sample_rows", {"table": table, "limit": limit})

    @server.tool(annotations=READ_ONLY)
    def run_sql(sql: str) -> str:
        """Run one read-only SELECT (DuckDB dialect) and get the rows back as a table."""
        return run("run_sql", {"sql": sql})

    if agent is not None:
        @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True))
        def ask(question: str) -> str:
            """Ask the ask-your-data agent a question in plain language; it explores the data,
            writes and checks the SQL, and answers with the query it relied on."""
            result = agent.ask(question)
            sql = f"\n\nSQL:\n{result.final_sql}" if result.final_sql else ""
            return f"{result.answer}{sql}"

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", default="sample", help="sample, retail, co2 or mine")
    source.add_argument("--db", type=Path, help="any DuckDB file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    set_memory_limit(settings.duckdb_memory_limit)
    if args.db:
        warehouse = Warehouse(args.db, max_rows=settings.max_rows, timeout_seconds=settings.query_timeout_seconds)
        agent = PortAnalystAgent(warehouse, None, settings, domain="user")
    else:
        agent = build_agent(settings, dataset=args.dataset)
        warehouse = agent.warehouse
    build_server(warehouse, settings, agent if settings.has_model_key() else None).run("stdio")


if __name__ == "__main__":
    main()
