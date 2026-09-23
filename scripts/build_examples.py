"""Build the real, public datasets the app offers next to the sample port data.

    python scripts/build_examples.py
    python scripts/build_examples.py --source-dir ~/Downloads   # files already downloaded
    python scripts/build_examples.py --allow-missing            # a failed download is not an error

Both are CC BY 4.0, and both are the datasets of the evaluation on unseen data
(eval/user_data), picked because the obvious query on each is wrong:

* retail - UCI Online Retail: 541,909 transactions of a UK online gift shop,
  Dec 2010 - Dec 2011. Returns and cancellations are negative rows.
  https://archive.ics.uci.edu/dataset/352/online+retail
* co2 - Our World in Data CO2 and greenhouse-gas emissions, by country and
  year. 'World', continents and income groups share the country column.
  https://github.com/owid/co2-data

Each becomes one read-only DuckDB file in data/examples/, imported the same
way a user's upload is.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.config import DATA_DIR  # noqa: E402
from agent.datasets import import_file  # noqa: E402

SOURCES = {
    "retail": ("https://archive.ics.uci.edu/static/public/352/online+retail.zip", "Online Retail.xlsx"),
    "co2": ("https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv", "owid-co2-data.csv"),
}


def fetch(name: str, into: Path) -> Path:
    url, filename = SOURCES[name]
    target = into / filename
    if target.exists():
        return target
    print(f"  downloading {name} ...", flush=True)
    if url.endswith(".zip"):
        archive = into / f"{name}.zip"
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as z:
            z.extract(filename, into)
        archive.unlink()
    else:
        urllib.request.urlretrieve(url, target)
    return target


def build(examples_dir: Path, source_dir: Path | None = None, allow_missing: bool = False) -> list[str]:
    examples_dir.mkdir(parents=True, exist_ok=True)
    built = []
    with tempfile.TemporaryDirectory() as tmp:
        downloads = source_dir or Path(tmp)
        for name in SOURCES:
            target = examples_dir / f"{name}.duckdb"
            if target.exists():
                print(f"  {name}: already built")
                built.append(name)
                continue
            try:
                source = fetch(name, downloads)
                started = time.perf_counter()
                # Built beside the target and moved into place, so an
                # interrupted build never leaves a half-written dataset
                # that the app would offer.
                partial = examples_dir / f"{name}.partial.duckdb"
                partial.unlink(missing_ok=True)
                result = import_file(source, partial)
                shutil.move(partial, target)
            except Exception as exc:
                if not allow_missing:
                    raise
                print(f"  {name}: skipped ({exc})", flush=True)
                continue
            print(f"  {name}: table {result.table}, {result.rows:,} rows, "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
            built.append(name)
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "examples")
    parser.add_argument("--source-dir", type=Path, help="Use source files already downloaded here")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Skip a dataset whose download fails instead of stopping")
    args = parser.parse_args()
    built = build(args.out, args.source_dir, args.allow_missing)
    print(f"Ready: {', '.join(built) or 'none'}")


if __name__ == "__main__":
    main()
