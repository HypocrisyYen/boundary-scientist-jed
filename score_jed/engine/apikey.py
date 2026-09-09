"""Strategist API-key loader — keeps the key OUT of source files.

Precedence: environment variable first (OPENAI_API_KEY / OPENROUTER_API_KEY /
MOONSHOT_API_KEY), then a local untracked file `.secrets/agnes_key.txt`. Returns
"" if none is found, so callers can fail loudly with a clear message instead of
shipping a hardcoded credential. Rotate the key if it was ever committed.
"""

from __future__ import annotations

import os
from pathlib import Path

_KEY_FILES = (
    Path(".secrets/agnes_key.txt"),
    Path(__file__).resolve().parent.parent / ".secrets" / "agnes_key.txt",
    Path.home() / ".secrets" / "agnes_key.txt",
)


def load_api_key() -> str:
    for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "MOONSHOT_API_KEY"):
        v = os.environ.get(var)
        if v:
            return v
    for p in _KEY_FILES:
        try:
            if p.is_file():
                v = p.read_text(encoding="utf-8").strip()
                if v:
                    return v
        except Exception:
            continue
    return ""
