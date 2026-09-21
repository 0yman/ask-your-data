"""Evaluate the agent against eval/questions.jsonl.

**Execution accuracy, not string matching.** There are many correct SQL
statements for any question, so comparing the agent's SQL to a reference query
as text measures formatting, not correctness. Each question instead carries a
hand-verified `gold_sql`; both are executed and their *result sets* compared,
normalised for row order and float precision.

Three things are scored:

* `execution_match` - did a query the agent ran reproduce the gold result set?
* `answer_has_figures` - do the gold numbers actually appear in the prose the
  user is shown? A run can match on SQL and still report the wrong figure.
* `declined` - on the deliberately unanswerable questions, did the agent say
  the warehouse cannot answer, instead of inventing a plausible number?

    python eval/run_eval.py                # all questions
    python eval/run_eval.py --limit 5      # quick pass, saves free-tier quota
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent.agent import build_agent  # noqa: E402
from agent.config import get_settings  # noqa: E402
from agent.warehouse import QueryResult, Warehouse  # noqa: E402

QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.jsonl"

DECLINE_MARKERS = (
    "cannot", "can't", "unable", "does not contain", "doesn't contain",
    "no data", "not available", "not recorded", "no column", "not present",
    "no information", "not stored", "isn't recorded", "is not tracked",
    "no crew", "no revenue",
)


def normalise_rows(rows: list[tuple], places: int = 2) -> list[tuple]:
    """Make two result sets comparable.

    Row order is ignored unless the question asked for a ranking - and even
    then, a correct answer that sorts ascending instead of descending is
    still the same set of facts. Floats are rounded because
    `ROUND(AVG(x), 2)` and `AVG(x)` are the same answer at the precision a
    human reads.
    """
    normalised = []
    for row in rows:
        cells = []
        for value in row:
            if isinstance(value, bool):
                # Booleans become 0/1 deliberately, not by relying on Python
                # treating True == 1. A gold query returning a BOOLEAN and an
                # agent query that cast it to INTEGER state the same fact, and
                # the explicit conversion also keeps the sort key below
                # consistent between the two.
                cells.append(float(value))
            elif isinstance(value, (int, float)):
                cells.append(round(float(value), places))
            elif value is None:
                cells.append(None)
            else:
                cells.append(str(value).strip())
        normalised.append(tuple(cells))
    return sorted(normalised, key=lambda r: tuple(str(c) for c in r))


def results_match(gold: QueryResult, candidate: QueryResult) -> bool:
    """Strict set equality: same rows, same columns, same number of them."""
    if gold.row_count != candidate.row_count:
        return False
    return normalise_rows(gold.rows) == normalise_rows(candidate.rows)


def covers_gold(gold: QueryResult, candidate: QueryResult) -> bool:
    """Does the candidate contain every fact the gold query returned?

    Strict equality turns out to measure the *shape* of a query as much as its
    correctness. Three patterns fail it while answering the question perfectly:

    * returning the whole ranking where the gold query used `LIMIT 1`;
    * carrying the intermediate columns (`total_calls`, `long_wait_calls`)
      alongside the percentage the question asked for;
    * computing a derived figure - a year-over-year delta - in prose instead
      of with a window function.

    This relaxation asks the question that actually matters of a result set:
    for every gold row, is there a candidate row containing all of its values?
    It is deliberately reported *next to* strict equality rather than instead
    of it, because it is the weaker claim: a candidate that returns everything
    trivially covers everything.
    """
    gold_rows = normalise_rows(gold.rows)
    candidate_rows = [set(row) for row in normalise_rows(candidate.rows)]
    if not gold_rows:
        return not candidate_rows
    return all(
        any(set(gold_row) <= candidate_row for candidate_row in candidate_rows)
        for gold_row in gold_rows
    )


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def gold_figures(rows: list[tuple]) -> list[float]:
    figures = []
    for row in rows:
        for value in row:
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                figures.append(float(value))
    return figures


def answer_mentions(answer: str, figures: list[float], tolerance: float = 0.02) -> float:
    """Fraction of the gold figures that appear in the answer text.

    Thousands separators, currency symbols and rounding all vary, so the
    answer's numbers are extracted and compared numerically rather than by
    substring.
    """
    if not figures:
        return 1.0
    found_numbers = [float(m) for m in _NUMBER.findall(answer.replace(",", ""))]
    if not found_numbers:
        return 0.0

    hits = 0
    for figure in figures:
        for found in found_numbers:
            scale = max(abs(figure), 1.0)
            if abs(found - figure) <= tolerance * scale:
                hits += 1
                break
    return hits / len(figures)


def looks_like_decline(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in DECLINE_MARKERS)


def evaluate(records: list[dict[str, Any]], settings, verbose: bool) -> dict[str, Any]:
    agent = build_agent(settings)
    gold_warehouse = Warehouse(settings.db_path)
    rows: list[dict[str, Any]] = []

    for record in records:
        print(f"  {record['id']} [{record['difficulty']}] {record['question'][:64]}...", flush=True)
        started = time.perf_counter()
        result = agent.ask(record["question"])
        elapsed = time.perf_counter() - started

        row: dict[str, Any] = {
            "id": record["id"],
            "difficulty": record["difficulty"],
            "question": record["question"],
            "answer": result.answer,
            "stop_reason": result.stop_reason,
            "steps": len(result.steps),
            "queries_run": len(result.sql_queries),
            "failed_attempts": result.failed_attempts,
            "latency_s": round(elapsed, 2),
            "usage": result.usage,
            "final_sql": result.final_sql,
        }

        if record.get("gold_sql"):
            gold = gold_warehouse.run_sql(record["gold_sql"])
            figures = gold_figures(gold.rows)

            any_match = False
            final_match = False
            covered = False
            for i, sql in enumerate(result.sql_queries):
                try:
                    candidate = gold_warehouse.run_sql(sql)
                except Exception:
                    continue
                if covers_gold(gold, candidate):
                    covered = True
                if results_match(gold, candidate):
                    any_match = True
                    if i == len(result.sql_queries) - 1:
                        final_match = True

            row.update(
                {
                    "execution_match": any_match,
                    "final_query_match": final_match,
                    "covers_gold": covered,
                    "answer_has_figures": round(answer_mentions(result.answer, figures), 3),
                    "declined": None,
                }
            )
        else:
            row.update(
                {
                    "execution_match": None,
                    "final_query_match": None,
                    "covers_gold": None,
                    "answer_has_figures": None,
                    "declined": looks_like_decline(result.answer),
                }
            )

        rows.append(row)
        if verbose:
            print(result.format_trace())
            print()

    return {"rows": rows}


def rescore(report: dict[str, Any], settings) -> dict[str, Any]:
    """Recompute the metrics from an earlier run's stored SQL.

    Every scoring rule here is a judgement call, and refining one should not
    cost another 100 model calls against a rate-limited free tier. The agent
    transcript is the expensive artefact; the scoring over it is cheap and
    worth being able to iterate on.
    """
    warehouse = Warehouse(settings.db_path)
    questions = {
        json.loads(line)["id"]: json.loads(line)
        for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    for row in report["rows"]:
        record = questions.get(row["id"])
        if record is None or not record.get("gold_sql"):
            row["declined"] = looks_like_decline(row["answer"])
            row["execution_match"] = row["final_query_match"] = row["covers_gold"] = None
            row["answer_has_figures"] = None
            continue

        gold = warehouse.run_sql(record["gold_sql"])
        row["answer_has_figures"] = round(answer_mentions(row["answer"], gold_figures(gold.rows)), 3)
        row["execution_match"] = row["final_query_match"] = row["covers_gold"] = False
        row["declined"] = None

        if row.get("final_sql"):
            try:
                candidate = warehouse.run_sql(row["final_sql"])
            except Exception:
                continue
            row["covers_gold"] = covers_gold(gold, candidate)
            if results_match(gold, candidate):
                row["execution_match"] = row["final_query_match"] = True

    report["summary"] = summarise(report["rows"])
    return report


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["execution_match"] is not None]
    traps = [r for r in rows if r["declined"] is not None]

    def mean(values):
        values = list(values)
        return round(sum(values) / len(values), 3) if values else 0.0

    by_difficulty = defaultdict(list)
    for row in answerable:
        by_difficulty[row["difficulty"]].append(row)

    return {
        "questions": len(rows),
        "answerable": len(answerable),
        "traps": len(traps),
        "execution_accuracy": mean(r["execution_match"] for r in answerable),
        "final_query_accuracy": mean(r["final_query_match"] for r in answerable),
        "gold_coverage": mean(r["covers_gold"] for r in answerable),
        "answer_figure_coverage": mean(r["answer_has_figures"] for r in answerable),
        "decline_accuracy": mean(r["declined"] for r in traps),
        "execution_accuracy_by_difficulty": {
            level: mean(r["execution_match"] for r in group)
            for level, group in sorted(by_difficulty.items())
        },
        "mean_steps": mean(r["steps"] for r in rows),
        "mean_queries": mean(r["queries_run"] for r in rows),
        "self_corrections": sum(r["failed_attempts"] for r in rows),
        "mean_latency_s": mean(r["latency_s"] for r in rows),
        "total_prompt_tokens": sum(r["usage"].get("prompt_tokens", 0) for r in rows),
        "total_output_tokens": sum(r["usage"].get("output_tokens", 0) for r in rows),
    }


def format_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Agent evaluation",
        "",
        f"- Model: `{report['model']}`",
        f"- Schema in system prompt: {report.get('schema_in_prompt', True)}",
        f"- Questions: **{summary['questions']}** "
        f"({summary['answerable']} answerable, {summary['traps']} unanswerable by design)",
        f"- Generated: {report['generated_at']}",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Answer figure coverage (end to end) | **{summary['answer_figure_coverage']:.3f}** |",
        f"| Execution accuracy (strict set equality) | {summary['execution_accuracy']:.3f} |",
        f"| Gold coverage (relaxed) | {summary['gold_coverage']:.3f} |",
        f"| Correctly declined (unanswerable) | {summary['decline_accuracy']:.3f} |",
        "",
        "## By difficulty",
        "",
        "| Difficulty | Execution accuracy |",
        "|---|---|",
    ]
    for level, value in summary["execution_accuracy_by_difficulty"].items():
        lines.append(f"| {level} | {value:.3f} |")

    lines += [
        "",
        "## Cost and behaviour",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean steps per question | {summary['mean_steps']:.2f} |",
        f"| Mean queries per question | {summary['mean_queries']:.2f} |",
        f"| Self-corrections triggered | {summary['self_corrections']} |",
        f"| Mean latency | {summary['mean_latency_s']:.1f}s |",
        f"| Prompt tokens | {summary['total_prompt_tokens']:,} |",
        f"| Output tokens | {summary['total_output_tokens']:,} |",
        "",
        "## Per question",
        "",
        "| id | difficulty | exec | figures | steps | retries | stop reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        if row["execution_match"] is None:
            exec_cell = "declined ok" if row["declined"] else "**invented**"
            figures = "-"
        else:
            exec_cell = "yes" if row["execution_match"] else "**no**"
            if not row["execution_match"] and row.get("covers_gold"):
                exec_cell = "covers"
            figures = f"{row['answer_has_figures']:.2f}"
        lines.append(
            f"| {row['id']} | {row['difficulty']} | {exec_cell} | {figures} | "
            f"{row['steps']} | {row['failed_attempts']} | {row['stop_reason']} |"
        )

    failures = [r for r in report["rows"] if r["execution_match"] is False]
    if failures:
        lines += ["", "## Failures", ""]
        for row in failures:
            lines += [
                f"**{row['id']}** — {row['question']}",
                "",
                "```sql",
                (row["final_sql"] or "(no query produced)"),
                "```",
                "",
                f"> {row['answer'][:400]}",
                "",
            ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions")
    parser.add_argument("--difficulty", type=str, default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "results.md")
    parser.add_argument("--json-out", type=Path, default=REPO_ROOT / "eval" / "results.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print each trace")
    parser.add_argument(
        "--no-schema-prompt",
        action="store_true",
        help="Withhold the schema from the system prompt, forcing the agent to discover it",
    )
    parser.add_argument("--model", type=str, default=None, help="Override the Gemini model")
    parser.add_argument(
        "--rescore",
        type=Path,
        default=None,
        help="Recompute metrics from a previous results.json without calling the model",
    )
    args = parser.parse_args()

    if args.rescore:
        settings = get_settings()
        report = rescore(json.loads(args.rescore.read_text(encoding="utf-8")), settings)
        args.out.write_text(format_markdown(report), encoding="utf-8")
        args.json_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        summary = report["summary"]
        print(
            f"rescored: execution={summary['execution_accuracy']:.3f}  "
            f"gold_coverage={summary['gold_coverage']:.3f}  "
            f"figures={summary['answer_figure_coverage']:.3f}  "
            f"declined={summary['decline_accuracy']:.3f}"
        )
        print(f"Wrote {args.out} and {args.json_out}")
        return 0

    records = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.difficulty:
        records = [r for r in records if r["difficulty"] == args.difficulty]
    if args.limit:
        records = records[: args.limit]

    settings = get_settings()
    print(f"Evaluating {len(records)} questions with {settings.gemini_model}\n")

    report = evaluate(records, settings, args.verbose)
    report["summary"] = summarise(report["rows"])
    report["model"] = settings.gemini_model
    report["schema_in_prompt"] = settings.include_schema_in_prompt
    report["generated_at"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    args.out.write_text(format_markdown(report), encoding="utf-8")
    args.json_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    summary = report["summary"]
    print(
        f"\nexecution_accuracy={summary['execution_accuracy']:.3f}  "
        f"figures={summary['answer_figure_coverage']:.3f}  "
        f"declined={summary['decline_accuracy']:.3f}  "
        f"steps={summary['mean_steps']:.2f}  "
        f"self_corrections={summary['self_corrections']}"
    )
    print(f"Wrote {args.out} and {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
