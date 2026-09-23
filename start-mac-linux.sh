#!/usr/bin/env bash
# Start the app. On a Mac you can double-click this in Finder after running
#   chmod +x start-mac-linux.sh
# once; otherwise run it from a terminal:  ./start-mac-linux.sh
#
# The first run installs what it needs (about two minutes); after that it
# starts in a few seconds.

set -e
cd "$(dirname "$0")"

PY=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 \
     && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    PY="$candidate"; break
  fi
done

if [ -z "$PY" ]; then
  echo
  echo "  Python 3.11 or newer is needed and was not found."
  echo "  Install it from https://www.python.org/downloads/ and run this again."
  echo
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo
  echo "  First run: setting things up. This takes about two minutes."
  echo
  if ! "$PY" -m venv .venv \
     || ! .venv/bin/python -m pip install --quiet --upgrade pip \
     || ! .venv/bin/python -m pip install --quiet -r requirements.txt; then
    echo
    echo "  Setup did not finish. Check your internet connection and run this again."
    echo "  If it keeps failing, delete the .venv folder and try once more."
    rm -rf .venv
    exit 1
  fi
fi

exec .venv/bin/python app.py
