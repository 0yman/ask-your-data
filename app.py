"""Start the app and open it in your browser.

    python app.py              # http://localhost:8000
    python app.py --port 9000
    python app.py --no-browser

Everything a first-time user can get wrong is checked here, before the server
starts, so the failure is a sentence that says what to do rather than a stack
trace.
"""

from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 11)
REQUIRED = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "duckdb": "duckdb",
    "sqlglot": "sqlglot",
    "pydantic_settings": "pydantic-settings",
    "multipart": "python-multipart",
    "openpyxl": "openpyxl",
}


def fail(message: str) -> None:
    print(f"\n  {message}\n", file=sys.stderr)
    sys.exit(1)


def check_environment() -> None:
    if sys.version_info < MIN_PYTHON:
        fail(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is needed; this is "
            f"{sys.version_info.major}.{sys.version_info.minor}. "
            "Install a newer Python from https://www.python.org/downloads/"
        )

    missing = [pkg for module, pkg in REQUIRED.items() if importlib.util.find_spec(module) is None]
    if missing:
        fail(
            "Some packages are missing: " + ", ".join(missing) + "\n"
            "  Install everything with:\n\n"
            "      pip install -r requirements.txt"
        )



def ensure_sample_data() -> None:
    """Build the sample port database on first run. It is generated from a
    fixed seed rather than shipped, which keeps the download small."""
    if (ROOT / "data" / "port.duckdb").exists():
        return
    print("  First run: building the sample database (about 20 seconds)...", flush=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_warehouse import DEFAULT_DB, build

    build(DEFAULT_DB)


def free_port(preferred: int) -> int:
    """The preferred port if it is free, otherwise the next one that is."""
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    fail(f"Ports {preferred}-{preferred + 19} are all in use. Pass another with --port.")
    return preferred  # unreachable


def announce_when_ready(url: str, open_browser: bool) -> None:
    """Wait for the server to answer, then say so - and open the browser.

    Opening it straight away gives a 'connection refused' page, because the
    first start builds the sample database, which takes a little while.
    """
    for _ in range(600):  # up to five minutes on a slow connection
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1):
                print(f"  Ready: {url}\n", flush=True)
                if open_browser:
                    webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.5)
    print("  Still not answering after five minutes - check the messages above.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about your data in plain English.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Use 0.0.0.0 to reach it from other devices on your network")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed logs")
    args = parser.parse_args()

    check_environment()
    sys.path.insert(0, str(ROOT / "src"))

    ensure_sample_data()
    port = free_port(args.port)
    url = f"http://localhost:{port}"

    say = lambda text="": print(text, flush=True)  # noqa: E731
    say()
    say("  Ask your data")
    say(f"  Starting on {url} ...  (Ctrl+C here to stop)")
    if not (ROOT / ".env").exists():
        say("  No AI key yet: the page will show you how to add a free one.")
    say()

    threading.Thread(
        target=announce_when_ready, args=(url, not args.no_browser), daemon=True
    ).start()

    import logging

    import uvicorn

    level = "info" if args.verbose else "warning"
    # The app's own logs are for developers; someone who just wants to ask
    # questions should see the few lines above, not a dict of index stats.
    logging.getLogger("agent").setLevel(logging.INFO if args.verbose else logging.WARNING)
    uvicorn.run("agent.api:app", host=args.host, port=port, log_level=level)


if __name__ == "__main__":
    main()
