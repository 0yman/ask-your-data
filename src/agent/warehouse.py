"""Read-only access to the DuckDB warehouse.

Two independent layers keep the agent from doing damage:

1. `guardrails.validate` parses the SQL and rejects anything but a single
   SELECT.
2. The connection itself is opened ``read_only=True``, so even a statement
   that somehow slipped past the parser cannot write.

Belt and braces on purpose: layer 1 knows *why* a query is wrong and can tell
the agent, but it is code I wrote and could have a gap; layer 2 is enforced by
the database and has no opinions.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from .guardrails import UnsafeSQLError, validate

logger = logging.getLogger(__name__)


class QueryTimeout(RuntimeError):
    pass


# Values that name a group rather than one entity: a "World" row among the
# countries, a "Total" row under the items. The prompt tells the model to look
# for them; on the CO2 data it did not, and added World to the countries or
# said the data had no continents. So the schema says so when a column holds
# them - and says nothing, leaving the prompt as it was, when none does.
# Whole values only: a product called "WORLD WAR 2 GLIDERS" is not a group.
_GROUP_VALUE = re.compile(
    r"^(world|total|grand total|all|overall)$"
    r"|^(africa|asia|europe|north america|south america|latin america|oceania"
    r"|middle east|european union)( \(.*\))?$"
    r"|income (countries|economies)|\(excl\. |\boecd\b|least developed",
    re.IGNORECASE,
)
_WHOLE = ("world", "total", "grand total", "all", "overall")
# All of them, up to this many: shown six and "22 more", the model guessed
# the rest and put the Democratic Republic of Congo among the groups.
_SHOW_GROUPS = 40


# Every connection in the process shares one configuration: DuckDB refuses a
# second connection to a file that is already open with different settings.
_CONFIG: dict[str, str] = {}


def set_memory_limit(limit: str) -> None:
    """Cap the memory of each database opened from now on ("" = DuckDB's own
    default). Called once at startup."""
    _CONFIG.pop("memory_limit", None)
    if limit:
        _CONFIG["memory_limit"] = limit


def connect(path: Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=read_only, config=dict(_CONFIG))


@dataclass(slots=True)
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    row_count: int
    truncated: bool
    elapsed_ms: float
    limit_applied: bool = False

    def to_markdown(self, max_rows: int = 30, max_cell: int = 40) -> str:
        """Render as a markdown table.

        The agent reads this, so it is capped: a thousand rows pasted into the
        context window crowds out the reasoning that should follow them.
        """
        if not self.columns:
            return "(no columns)"
        if not self.rows:
            return "(0 rows)"

        def cell(value: Any) -> str:
            if value is None:
                return "NULL"
            if isinstance(value, float):
                text = f"{value:,.4f}".rstrip("0").rstrip(".")
            else:
                text = str(value)
            return text if len(text) <= max_cell else text[: max_cell - 1] + "…"

        shown = self.rows[:max_rows]
        lines = [
            "| " + " | ".join(self.columns) + " |",
            "| " + " | ".join("---" for _ in self.columns) + " |",
        ]
        lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in shown]
        if len(self.rows) > max_rows:
            lines.append(f"\n({len(self.rows) - max_rows} more rows not shown)")
        return "\n".join(lines)


@dataclass(slots=True)
class ColumnInfo:
    name: str
    type: str
    nullable: bool


@dataclass(slots=True)
class TableInfo:
    name: str
    kind: str          # "fact" or "dimension", inferred from the name prefix
    row_count: int
    columns: list[ColumnInfo] = field(default_factory=list)


class Warehouse:
    def __init__(
        self,
        db_path: Path,
        max_rows: int = 1000,
        timeout_seconds: float = 20.0,
    ) -> None:
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(
                f"No warehouse at {db_path}. Run `python scripts/build_warehouse.py` first."
            )
        self.db_path = db_path
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        self._connection = connect(db_path, read_only=True)
        # Re-entrant: schema_summary holds it while calling describe_table.
        # Parallel agent runs (vote_runs) share this connection.
        self._lock = threading.RLock()

    def close(self) -> None:
        self._connection.close()

    # --- introspection ---------------------------------------------------

    def list_tables(self) -> list[TableInfo]:
        with self._lock:
            return self._list_tables()

    def _list_tables(self) -> list[TableInfo]:
        names = [
            row[0]
            for row in self._connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
        ]
        tables = []
        for name in names:
            count = self._connection.execute(
                f'SELECT COUNT(*) FROM "{name}"'  # noqa: S608 - name comes from the catalogue
            ).fetchone()[0]
            tables.append(
                TableInfo(
                    name=name,
                    kind="fact" if name.startswith("fact_") else "dimension",
                    row_count=count,
                )
            )
        return tables

    def describe_table(self, table: str) -> TableInfo:
        known = {t.name: t for t in self.list_tables()}
        if table not in known:
            raise ValueError(
                f"No table named {table!r}. Available tables: {', '.join(sorted(known))}."
            )
        info = known[table]
        with self._lock:
            rows = self._connection.execute(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
        info.columns = [
            ColumnInfo(name=name, type=data_type, nullable=(nullable == "YES"))
            for name, data_type, nullable in rows
        ]
        return info

    def schema_summary(self) -> str:
        """A compact schema description for the system prompt.

        Sent once, up front. Making the agent discover the schema tool-call by
        tool-call wastes turns on something that fits in a few hundred tokens.
        """
        lines = []
        for table in self.list_tables():
            info = self.describe_table(table.name)
            columns = ", ".join(f"{c.name} {c.type}" for c in info.columns)
            lines.append(f"{info.name} ({info.row_count:,} rows): {columns}")
            for column in info.columns:
                if column.type.upper() != "VARCHAR":
                    continue
                groups = self._group_values(info.name, column.name)
                if groups:
                    shown = ", ".join(f"'{g}'" for g in groups[:_SHOW_GROUPS])
                    more = f" and {len(groups) - _SHOW_GROUPS} more" if len(groups) > _SHOW_GROUPS else ""
                    lines.append(
                        f"  - {column.name} also holds {len(groups)} group values, not single "
                        f"entities: {shown}{more}. Leave them out when ranking or adding up "
                        "single entities; use them when the question is about the group."
                    )
        return "\n".join(lines)

    def _group_values(self, table: str, column: str) -> list[str]:
        """Values of a text column that name groups ("World", "Asia",
        "High-income countries"): the whole-data ones first, then plain names
        before variants such as "Asia (GCP)"."""
        quoted_table, quoted_column = (f'"{name.replace(chr(34), chr(34) * 2)}"' for name in (table, column))
        with self._lock:
            values = self._connection.execute(
                f"SELECT DISTINCT {quoted_column} FROM {quoted_table} "  # noqa: S608 - names from the catalogue
                f"WHERE {quoted_column} IS NOT NULL LIMIT 50000"
            ).fetchall()
        groups = [v for (v,) in values if isinstance(v, str) and _GROUP_VALUE.search(v.strip())]
        return sorted(groups, key=lambda v: (v.strip().lower() not in _WHOLE, "(" in v, v))

    # --- querying --------------------------------------------------------

    def run_sql(self, sql: str) -> QueryResult:
        """Validate then execute. Raises UnsafeSQLError or QueryTimeout."""
        safe = validate(sql, max_rows=self.max_rows)
        started = time.perf_counter()

        # DuckDB has no statement timeout, so a watchdog interrupts a query
        # that overruns. Without it one accidental cross join hangs the API.
        timer = threading.Timer(self.timeout_seconds, self._connection.interrupt)
        timer.daemon = True

        with self._lock:  # a DuckDB connection is not safe for concurrent use
            timer.start()
            try:
                cursor = self._connection.execute(safe.sql)
                columns = [d[0] for d in cursor.description or []]
                rows = cursor.fetchall()
            except duckdb.InterruptException as exc:
                raise QueryTimeout(
                    f"Query exceeded {self.timeout_seconds:.0f}s and was cancelled. "
                    "Add a filter or an aggregation to reduce the work."
                ) from exc
            finally:
                timer.cancel()

        elapsed = (time.perf_counter() - started) * 1000
        return QueryResult(
            sql=safe.sql,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=len(rows) >= self.max_rows,
            elapsed_ms=elapsed,
            limit_applied=safe.limit_applied,
        )

    def sample_rows(self, table: str, limit: int = 5) -> QueryResult:
        self.describe_table(table)  # validates the name against the catalogue
        return self.run_sql(f'SELECT * FROM "{table}" LIMIT {int(limit)}')  # noqa: S608


__all__ = [
    "ColumnInfo",
    "QueryResult",
    "QueryTimeout",
    "TableInfo",
    "UnsafeSQLError",
    "Warehouse",
    "connect",
    "set_memory_limit",
]
