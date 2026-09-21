"""SQL guardrails.

A text-to-SQL agent hands a language model a database connection. Anything
that reaches the model - a question, a column comment, a row of data echoed
back into context - is untrusted input that can try to steer the SQL it emits.
Prompting the model to "only write SELECT statements" is a request, not a
control. This module is the control: every statement is parsed and inspected
before it reaches the database, and rejected on anything but a plain read.

What is blocked and why:

* **Anything that is not a single SELECT.** Multiple statements let an
  injected instruction append `; DROP TABLE ...` to an otherwise valid query.
* **DML and DDL anywhere in the tree**, including inside CTEs, where an
  `INSERT` can hide behind a legitimate-looking `WITH` clause.
* **DuckDB's file and extension functions** - `read_csv`, `read_parquet`,
  `COPY`, `ATTACH`, `INSTALL`, `LOAD`. These turn a read-only SQL endpoint
  into arbitrary local file access, which is the actual exfiltration path.
* **Unbounded result sets.** A missing LIMIT is added rather than rejected;
  the agent should not have to remember, and an accidental cross join should
  not return ten million rows into a model's context window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

DIALECT = "duckdb"

# Statement types that write, alter, or reach outside the database.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.Merge, exp.Copy, exp.Attach, exp.Detach, exp.Command, exp.Set,
    exp.Grant, exp.Use, exp.TruncateTable,
)

# DuckDB functions that read or write the local filesystem, or load code.
_FORBIDDEN_FUNCTIONS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "read_text", "read_blob", "read_ndjson", "read_ndjson_auto", "glob",
    "parquet_scan", "csv_scan", "iceberg_scan", "delta_scan",
    "install", "load", "attach", "detach", "copy",
    "sniff_csv", "parquet_metadata", "parquet_file_metadata", "shell",
}

# Statement keywords that sqlglot may parse as a generic Command node.
_FORBIDDEN_KEYWORDS = re.compile(
    r"^\s*(insert|update|delete|drop|create|alter|truncate|merge|copy|attach|"
    r"detach|install|load|pragma|set|call|export|import|vacuum|checkpoint|"
    r"grant|revoke|begin|commit|rollback)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    """Raised when a statement is rejected. The message is fed back to the
    agent, so it says what was wrong rather than just that something was."""


@dataclass(slots=True)
class SafeQuery:
    sql: str
    limit_applied: bool
    original_sql: str


def _function_name(node: exp.Expression) -> str | None:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    if isinstance(node, exp.Func):
        name = getattr(node, "sql_name", None)
        if callable(name):
            return name().lower()
    return None


def validate(sql: str, max_rows: int = 1000) -> SafeQuery:
    """Parse, inspect and normalise a statement.

    Returns the statement to actually run - which may have gained a LIMIT.
    Raises `UnsafeSQLError` with a usable explanation otherwise.
    """
    if not sql or not sql.strip():
        raise UnsafeSQLError("Empty query.")

    original = sql.strip().rstrip(";").strip()

    # Cheap keyword check first: it catches statements whose parse tree is a
    # generic Command node that the structural pass below cannot classify.
    if _FORBIDDEN_KEYWORDS.match(original):
        keyword = _FORBIDDEN_KEYWORDS.match(original).group(1).upper()
        raise UnsafeSQLError(
            f"{keyword} is not allowed. This connection is read-only; "
            "use a SELECT statement."
        )

    try:
        statements = sqlglot.parse(original, dialect=DIALECT)
    except Exception as exc:
        raise UnsafeSQLError(f"Could not parse SQL: {exc}") from exc

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise UnsafeSQLError(
            f"Expected exactly one statement, found {len(statements)}. "
            "Send one SELECT at a time."
        )

    statement = statements[0]
    if not isinstance(statement, (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)):
        raise UnsafeSQLError(
            f"Only SELECT statements are allowed, got {type(statement).__name__.upper()}."
        )

    for node in statement.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeSQLError(
                f"{type(node).__name__.upper()} is not allowed anywhere in the "
                "query, including inside CTEs and subqueries."
            )
        name = _function_name(node)
        if name and name in _FORBIDDEN_FUNCTIONS:
            raise UnsafeSQLError(
                f"The function {name}() is not allowed: it reads or writes "
                "files outside the database."
            )

    limit_applied = False
    if isinstance(statement, exp.Select) and statement.args.get("limit") is None:
        statement = statement.limit(max_rows)
        limit_applied = True
    elif not isinstance(statement, exp.Select) and statement.args.get("limit") is None:
        # UNION and friends take a LIMIT on the whole set operation.
        statement = exp.Subquery(this=statement).limit(max_rows)
        limit_applied = True

    return SafeQuery(
        sql=statement.sql(dialect=DIALECT),
        limit_applied=limit_applied,
        original_sql=original,
    )
