"""Pre-run / pre-bundle secret scanner (P0-5 defense-in-depth).

Scans a directory tree for likely credentials (API keys, tokens) and exits non-zero
if any are found, so a run or a bundle can be gated on a clean scan. The out-of-tree
key store (~/.secrets) and the read-only legacy corpus are excluded by design.

Usage:  python tools/secret_scan.py [root]      # default root = score_jed dir
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PATTERNS = [
    ("openai/generic sk-", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),
    ("slack token", re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}\b")),
    ("bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}\b")),
]
_SKIP_DIRS = {".git", "__pycache__", ".secrets", "results_disc_legacy_buggy", ".venv", "node_modules"}
_SCAN_SUFFIXES = {".py", ".json", ".jsonl", ".txt", ".ipynb", ".md", ".yaml", ".yml", ".sh", ".cfg", ".ini", ".env"}


def scan(root: str | Path) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for p in Path(root).rglob("*"):
        if not p.is_file() or p.suffix.lower() not in _SCAN_SUFFIXES:
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for name, pat in _PATTERNS:
            if pat.search(txt):
                hits.append((str(p), name))
    return hits


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent)
    found = scan(root)
    if found:
        for path, kind in found:
            print(f"SECRET FOUND [{kind}]: {path}")
        print(f"FAIL: {len(found)} potential secret(s) in tree.")
        sys.exit(1)
    print("secret scan clean")
    sys.exit(0)
