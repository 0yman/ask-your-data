"""Publish the app as a public demo on Hugging Face Spaces.

    python scripts/deploy_space.py                  # Space: <your user>/ask-your-data
    python scripts/deploy_space.py --space me/demo

Needs a Hugging Face write token (`hf auth login`, or HF_TOKEN). What it does:

1. Creates the Space (Docker) if it does not exist.
2. Stores GOOGLE_API_KEY from .env as a Space *secret*: it never enters the
   Space's files, and nobody visiting can read it.
3. Turns on public mode: a private workspace per visitor, a rationed key,
   and a memory cap per database.
4. Uploads the committed files - exactly what `git archive` sees, so .env,
   data/ and anything uncommitted cannot leak - with a README carrying the
   metadata Spaces reads.

The Space then builds the Dockerfile itself; the first build takes a few
minutes, most of it importing the two example datasets.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/0yman/ask-your-data"

FRONT_MATTER = """---
title: Ask Your Data
emoji: 📊
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 8000
pinned: true
license: mit
short_description: Ask a spreadsheet a question; an agent writes the SQL
---

> Live demo of [{github}]({github}). Source code, tests and the full
> evaluation live there; this Space is built from it.

"""

PUBLIC_SETTINGS = {
    "AGENT_PUBLIC_MODE": "true",
    "AGENT_DUCKDB_MEMORY_LIMIT": "2GB",
}


def read_env_key(name: str) -> str | None:
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == name and value.strip():
            return value.strip().strip('"').strip("'")
    return None


def committed_tree(into: Path) -> None:
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        print("  note: uncommitted changes are not deployed - commit them first to include them")
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, capture_output=True, check=True)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(into, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", help="owner/name (default: <your user>/ask-your-data)")
    parser.add_argument("--skip-secret", action="store_true", help="Leave the Space's key as it is")
    args = parser.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    user = api.whoami()["name"]
    space = args.space or f"{user}/ask-your-data"

    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"Space: https://huggingface.co/spaces/{space}")

    if not args.skip_secret:
        key = os.environ.get("GOOGLE_API_KEY") or read_env_key("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("No GOOGLE_API_KEY in the environment or .env - nothing to give the Space.")
        api.add_space_secret(space, "GOOGLE_API_KEY", key)
        print("  key stored as a Space secret")
    for name, value in PUBLIC_SETTINGS.items():
        api.add_space_variable(space, name, value)
    print("  public mode on")

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        committed_tree(folder)
        readme = folder / "README.md"
        readme.write_text(FRONT_MATTER.format(github=GITHUB) + readme.read_text(encoding="utf-8"), encoding="utf-8")
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        api.upload_folder(
            repo_id=space, repo_type="space", folder_path=folder,
            commit_message=f"Deploy {commit} from {GITHUB}",
            delete_patterns=["*"],  # the Space mirrors the commit: files removed there go here too
        )
    host = space.replace("/", "-").replace("_", "-").lower()
    print(f"  uploaded {commit}; building now. When it is up: https://{host}.hf.space")


if __name__ == "__main__":
    main()
