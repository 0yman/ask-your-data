"""Write one setting into the .env file, leaving everything else as it was.

Lets someone paste an API key into the web page instead of finding, creating
and editing a hidden file - the step most likely to lose a non-developer.
"""

from __future__ import annotations

import re
from pathlib import Path

_VALID_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def set_env_value(path: Path, name: str, value: str) -> None:
    if not _VALID_NAME.match(name):
        raise ValueError(f"Not a valid setting name: {name!r}")
    # A newline in the value would let it write a second, unintended setting.
    if any(ch in value for ch in "\r\n"):
        raise ValueError("The value must be a single line.")

    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    replaced = False
    for i, line in enumerate(lines):
        if re.match(rf"^\s*{name}\s*=", line):
            lines[i] = f"{name}={value}"
            replaced = True
    if not replaced:
        lines.append(f"{name}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
