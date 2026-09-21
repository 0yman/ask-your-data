"""Command-line interface.

    python -m agent.cli "Which berth is the least productive per crane?"
    python -m agent.cli --trace "Compare dwell time by quarter"
    python -m agent.cli --schema
"""

from __future__ import annotations

import argparse
import logging
import sys

from .agent import build_agent
from .config import get_settings
from .warehouse import Warehouse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="port-analyst", description=__doc__)
    parser.add_argument("question", nargs="?", help="The question to answer")
    parser.add_argument("--trace", action="store_true", help="Print every agent step")
    parser.add_argument("--schema", action="store_true", help="Print the warehouse schema and exit")
    parser.add_argument("--sql", type=str, help="Run one SELECT through the guardrails and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()

    if args.schema:
        print(Warehouse(settings.db_path).schema_summary())
        return 0

    if args.sql:
        result = Warehouse(settings.db_path).run_sql(args.sql)
        print(result.to_markdown())
        print(f"\n{result.row_count} row(s) in {result.elapsed_ms:.0f}ms")
        return 0

    if not args.question:
        parser.error("a question is required unless --schema or --sql is given")

    result = build_agent(settings).ask(args.question)

    if args.trace:
        print(result.format_trace())
        print()
    else:
        print(result.answer)

    if result.sql_queries:
        print(f"\nSQL ({len(result.sql_queries)} quer{'y' if len(result.sql_queries) == 1 else 'ies'}):")
        print(f"  {result.sql_queries[-1]}")

    print(
        f"\n[{len(result.steps)} steps, {result.failed_attempts} self-corrections, "
        f"{result.total_latency_ms / 1000:.1f}s, "
        f"{result.usage.get('prompt_tokens', 0) + result.usage.get('output_tokens', 0):,} tokens]",
        file=sys.stderr,
    )
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
