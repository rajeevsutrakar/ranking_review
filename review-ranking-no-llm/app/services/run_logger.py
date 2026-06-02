from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "runtime_no_llm.log"


def _write(level: str, message: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] [{level}] {message}"
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_info(message: str) -> None:
    _write("INFO", message)


def log_warn(message: str) -> None:
    _write("WARN", message)


def log_error(message: str) -> None:
    _write("ERROR", message)
