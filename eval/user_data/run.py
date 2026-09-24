"""Evaluate the agent on real, public data it has never seen - the "My files" path.

The main evaluation (eval/run_eval.py) runs on the synthetic port warehouse,
with a system prompt that carries its business rules. This one does what a
user does instead: uploads real spreadsheets, gets the general prompt with no
domain hints, and asks questions - including ones where the obvious query is
wrong, and ones the data cannot answer.

Datasets (both CC BY 4.0), downloaded on first run:

* UCI Online Retail - 541,909 transactions from a UK online shop, as Excel.
  Returns are negative rows, cancellations are 'C'-prefixed invoices, a
  quarter of rows have no customer.
* Our World in Data CO2 - 79 columns of emissions data per country and year.
  'World', continents and income groups share the `country` column with real
  countries.

Grading is on the figures and names in the answer, checked against values
computed directly from the data. Trap questions also record whether the
answer contains the specific wrong value the trap produces.

    python eval/user_data/run.py
    python eval/user_data/run.py --only r4,c1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from agent.agent import build_agent  # noqa: E402
from agent.config import get_settings  # noqa: E402
from agent.datasets import import_file  # noqa: E402
from run_eval import looks_like_decline  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "user_eval"
SOURCES = {
    "retail": ("https://archive.ics.uci.edu/static/public/352/online+retail.zip", "Online Retail.xlsx"),
    "co2": ("https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv", "owid-co2-data.csv"),
}
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def fetch(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (url, filename) in SOURCES.items():
        target = data_dir / filename
        if not target.exists():
            print(f"Downloading {name} ...", flush=True)
            if url.endswith(".zip"):
                archive = data_dir / "download.zip"
                urllib.request.urlretrieve(url, archive)
                with zipfile.ZipFile(archive) as z:
                    z.extract(filename, data_dir)
                archive.unlink()
            else:
                urllib.request.urlretrieve(url, target)
        paths[name] = target
    return paths


def numbers_in(text: str) -> list[float]:
    return [float(m) for m in _NUMBER.findall(text.replace(",", ""))]


def mentions_number(text: str, target: float, tolerance: float) -> bool:
    for found in numbers_in(text):
        if tolerance == 0:
            if found == target:
                return True
        elif abs(found - target) <= tolerance * max(abs(target), 1.0):
            return True
    return False


def meets(condition: dict[str, Any], answer: str) -> bool:
    """One part of a multi-part answer: a name (any of several spellings)
    or a figure within a relative tolerance. `abs` accepts either sign, for
    "fell by 967.8" as well as "-967.8"."""
    if "text" in condition:
        return any(t in answer.lower() for t in condition["text"])
    target, tolerance = condition["number"], condition.get("tolerance", 0.01)
    if condition.get("abs"):
        return any(
            abs(abs(found) - abs(target)) <= tolerance * max(abs(target), 1.0) if tolerance else abs(found) == abs(target)
            for found in numbers_in(answer)
        )
    return mentions_number(answer, target, tolerance)


def grade(record: dict[str, Any], answer: str) -> dict[str, Any]:
    text = answer.lower()
    if "all_of" in record:
        # A multi-part question is right only when every part is.
        parts = [meets(condition, answer) for condition in record["all_of"]]
        return {"correct": all(parts), "fell_for_trap": False, "declined": looks_like_decline(answer),
                "parts": f"{sum(parts)}/{len(parts)}"}
    if record["kind"] == "unanswerable":
        declined = looks_like_decline(answer)
        return {"correct": declined, "fell_for_trap": False, "declined": declined}

    tolerance = record.get("tolerance", 0.01)
    correct = any(t in text for t in record.get("accept_text", [])) or any(
        mentions_number(answer, n, tolerance) for n in record.get("accept_numbers", [])
    )
    fell = any(t in text for t in record.get("trap_text", [])) or any(
        mentions_number(answer, n, tolerance) for n in record.get("trap_numbers", [])
    )
    # Naming the trap value while also giving the right answer ("World is
    # higher, but among countries China leads") is not falling for it.
    return {"correct": correct, "fell_for_trap": fell and not correct, "declined": looks_like_decline(answer)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="Comma-separated question ids")
    parser.add_argument("--questions", type=Path, default=HERE / "questions.jsonl",
                        help="Question set: questions.jsonl (17) or hard_questions.jsonl (12)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=HERE / "results.md")
    parser.add_argument("--json-out", type=Path, default=HERE / "results.json")
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]

    paths = fetch(args.data_dir)
    rows: list[dict[str, Any]] = []
    for dataset in ("retail", "co2"):
        subset = [q for q in questions if q["dataset"] == dataset]
        if not subset:
            continue
        # One dataset per agent, the way the app shows one upload set at a time.
        db = args.data_dir / f"{dataset}.duckdb"
        if not db.exists():
            started = time.perf_counter()
            imported = import_file(paths[dataset], db)
            print(f"Imported {imported.table}: {imported.rows:,} rows in {time.perf_counter() - started:.0f}s")
        settings = get_settings(user_db_path=db)
        agent = build_agent(settings, dataset="mine")
        for record in subset:
            print(f"  {record['id']} [{record['kind']}] {record['question']}", flush=True)
            started = time.perf_counter()
            try:
                result = agent.ask(record["question"])
                answer, error = result.answer, None
            except Exception as exc:  # a model outage should not end the run
                result, answer, error = None, "", str(exc)[:200]
            row = {
                "id": record["id"], "dataset": dataset, "kind": record["kind"],
                "question": record["question"], "trap": record.get("trap"),
                "answer": answer, "error": error,
                "latency_s": round(time.perf_counter() - started, 1),
                "steps": len(result.steps) if result else 0,
                "failed_attempts": result.failed_attempts if result else 0,
                "stop_reason": result.stop_reason if result else "error",
                "final_sql": result.final_sql if result else None,
                **(grade(record, answer) if not error else {"correct": False, "fell_for_trap": False, "declined": False}),
            }
            print(f"      -> {'OK ' if row['correct'] else 'MISS'}{' (trap)' if row['fell_for_trap'] else ''}  {answer[:140]!r}")
            rows.append(row)
        agent.warehouse.close()

    report = summarise(rows)
    args.json_out.write_text(json.dumps({"summary": report, "rows": rows}, indent=2), encoding="utf-8")
    model = settings.openai_model if settings.llm_backend == "openai" else settings.gemini_model
    args.out.write_text(markdown(report, rows, model), encoding="utf-8")
    print(f"\n{report['correct']}/{report['total']} correct  |  plain {report['by_kind'].get('plain','-')}  "
          f"trap {report['by_kind'].get('trap','-')}  unanswerable {report['by_kind'].get('unanswerable','-')}")
    return 0


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, str] = {}
    for kind in ("plain", "trap", "multi", "puzzle", "trick", "unanswerable"):
        group = [r for r in rows if r["kind"] == kind]
        if group:
            by_kind[kind] = f"{sum(r['correct'] for r in group)}/{len(group)}"
    return {
        "total": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "by_kind": by_kind,
        "fell_for_traps": [r["id"] for r in rows if r["fell_for_trap"]],
        "errors": [r["id"] for r in rows if r["error"]],
        "self_corrections": sum(r["failed_attempts"] for r in rows),
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / max(len(rows), 1), 1),
    }


def markdown(report: dict[str, Any], rows: list[dict[str, Any]], model: str) -> str:
    lines = [
        "# Evaluation on real, unseen data",
        "",
        f"Model: `{model}` · general prompt, no domain rules · {time.strftime('%Y-%m-%d')}",
        "",
        f"**{report['correct']} / {report['total']} correct** — "
        + " · ".join(f"{k}: {v}" for k, v in report["by_kind"].items()),
        "",
        "| id | kind | correct | fell for trap | steps | question | answer |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        answer = (r["error"] and f"ERROR: {r['error']}") or r["answer"]
        answer = answer.replace("|", "\\|").replace("\n", " ")[:220]
        lines.append(
            f"| {r['id']} | {r['kind']} | {'yes' if r['correct'] else '**no**'} | "
            f"{'**yes**' if r['fell_for_trap'] else ''} | {r['steps']} | {r['question']} | {answer} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
