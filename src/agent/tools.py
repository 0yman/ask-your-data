"""The tools the agent can call, and the dispatcher that runs them.

Each tool returns a string the model reads. The rule throughout: **a failure
returns a message, it does not raise.** A rejected query or a misspelled table
name is information the agent can act on - telling it what went wrong and
letting it try again is the entire mechanism behind self-correction. Only a
genuinely unexpected error escapes as an exception.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .guardrails import UnsafeSQLError
from .llm import ToolSpec
from .warehouse import QueryResult, QueryTimeout, Warehouse

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolOutcome:
    content: str
    ok: bool = True
    # Populated by run_sql so the caller can keep the winning query without
    # re-parsing it out of the transcript.
    result: QueryResult | None = None
    error_kind: str | None = None


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="list_tables",
        description=(
            "List every table in the warehouse with its row count and whether "
            "it is a fact or a dimension table. Use this first if you are "
            "unsure what exists."
        ),
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="describe_table",
        description=(
            "Show the column names, types and nullability of one table. Use it "
            "before writing SQL against a table you have not inspected."
        ),
        parameters={
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Exact table name."}
            },
            "required": ["table"],
        },
    ),
    ToolSpec(
        name="sample_rows",
        description=(
            "Return a few rows from a table so you can see the actual values - "
            "useful for checking how a category or a date is encoded before "
            "filtering on it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "limit": {"type": "integer", "description": "Rows to return, 1-20.", "default": 5},
            },
            "required": ["table"],
        },
    ),
    ToolSpec(
        name="run_sql",
        description=(
            "Execute a read-only DuckDB SELECT and return the rows. Only a "
            "single SELECT is permitted; anything that writes, or that reads "
            "files, is rejected. If the query fails you will be told why and "
            "may correct it and try again."
        ),
        parameters={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "One DuckDB SELECT statement."},
                "purpose": {
                    "type": "string",
                    "description": "One line on what this query is meant to establish.",
                },
            },
            "required": ["sql"],
        },
    ),
    ToolSpec(
        name="final_answer",
        description=(
            "Give the final answer to the user's question. Call this once you "
            "have the numbers you need. State the figures explicitly - do not "
            "say 'as shown above'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The answer in prose, citing the actual numbers.",
                }
            },
            "required": ["answer"],
        },
    ),
]

FINAL_ANSWER_TOOL = "final_answer"


@dataclass
class ToolBox:
    warehouse: Warehouse
    max_rows_to_model: int = 30
    #: Every successful query, in order, for the trace and for evaluation.
    executed_queries: list[QueryResult] = field(default_factory=list)

    @property
    def specs(self) -> list[ToolSpec]:
        return TOOL_SPECS

    def dispatch(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        handler = {
            "list_tables": self._list_tables,
            "describe_table": self._describe_table,
            "sample_rows": self._sample_rows,
            "run_sql": self._run_sql,
        }.get(name)

        if handler is None:
            return ToolOutcome(
                content=(
                    f"There is no tool called {name!r}. Available tools: "
                    + ", ".join(spec.name for spec in TOOL_SPECS)
                ),
                ok=False,
                error_kind="unknown_tool",
            )
        try:
            return handler(arguments)
        except TypeError as exc:
            # A missing or misnamed argument is the model's mistake to fix.
            return ToolOutcome(
                content=f"Bad arguments for {name}: {exc}",
                ok=False,
                error_kind="bad_arguments",
            )

    # --- handlers --------------------------------------------------------

    def _list_tables(self, arguments: dict[str, Any]) -> ToolOutcome:
        lines = [
            f"{table.name} ({table.kind}, {table.row_count:,} rows)"
            for table in self.warehouse.list_tables()
        ]
        return ToolOutcome(content="\n".join(lines))

    def _describe_table(self, arguments: dict[str, Any]) -> ToolOutcome:
        table = arguments.get("table")
        if not table:
            return ToolOutcome(
                content="describe_table needs a 'table' argument.",
                ok=False,
                error_kind="bad_arguments",
            )
        try:
            info = self.warehouse.describe_table(str(table))
        except ValueError as exc:
            return ToolOutcome(content=str(exc), ok=False, error_kind="unknown_table")

        lines = [f"{info.name} ({info.kind}, {info.row_count:,} rows)"]
        lines += [
            f"  {column.name}: {column.type}{'' if column.nullable else ' NOT NULL'}"
            for column in info.columns
        ]
        return ToolOutcome(content="\n".join(lines))

    def _sample_rows(self, arguments: dict[str, Any]) -> ToolOutcome:
        table = arguments.get("table")
        if not table:
            return ToolOutcome(
                content="sample_rows needs a 'table' argument.",
                ok=False,
                error_kind="bad_arguments",
            )
        limit = max(1, min(int(arguments.get("limit") or 5), 20))
        try:
            result = self.warehouse.sample_rows(str(table), limit)
        except ValueError as exc:
            return ToolOutcome(content=str(exc), ok=False, error_kind="unknown_table")
        return ToolOutcome(
            content=result.to_markdown(max_rows=limit), result=result
        )

    def _run_sql(self, arguments: dict[str, Any]) -> ToolOutcome:
        sql = arguments.get("sql")
        if not sql:
            return ToolOutcome(
                content="run_sql needs a 'sql' argument containing one SELECT statement.",
                ok=False,
                error_kind="bad_arguments",
            )
        try:
            result = self.warehouse.run_sql(str(sql))
        except UnsafeSQLError as exc:
            return ToolOutcome(
                content=f"Query rejected: {exc}", ok=False, error_kind="unsafe_sql"
            )
        except QueryTimeout as exc:
            return ToolOutcome(content=str(exc), ok=False, error_kind="timeout")
        except Exception as exc:
            # Genuine SQL errors - a wrong column, a bad join - land here and
            # are the most valuable thing the agent can be told.
            return ToolOutcome(
                content=f"SQL error: {exc}\n\nCheck the schema and try again.",
                ok=False,
                error_kind="sql_error",
            )

        self.executed_queries.append(result)
        header = f"{result.row_count} row(s) in {result.elapsed_ms:.0f}ms"
        if result.limit_applied:
            header += f" (a LIMIT {self.warehouse.max_rows} was added automatically)"
        if result.truncated:
            header += " — results were truncated; aggregate instead of listing"
        return ToolOutcome(
            content=f"{header}\n\n{result.to_markdown(max_rows=self.max_rows_to_model)}",
            result=result,
        )
