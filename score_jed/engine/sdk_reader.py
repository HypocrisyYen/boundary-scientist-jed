"""On-demand whole-SDK reading capability for the boundary scientist.

NOT a digest: in ``source_informed`` knowledge mode the scientist pulls ANY file
under the allowed roots IN FULL (paginating only very large fixtures), so it can
hunt mechanisms that live inside implementation details — the predicate decode
pipeline, cell_signature, the guardrail's arg-scan order, the agent's request
builder / chat-template handling — none of which a pre-computed summary can
anticipate. Nothing is injected per-prompt except a small top-level tree; the
model reads what it judges relevant, when it judges it relevant.

Safety: read-only, confined to explicit roots (the competition SDK tree + the
scientist's own per-scope results dir). Path traversal (``..``), absolute paths,
``.secrets`` and anything outside the roots are refused. Blind mode never gets
a reader instance, so this capability cannot leak into the API-only condition.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

_SKIP_DIRS = {".git", "__pycache__", ".secrets", ".venv", "node_modules"}
_BINARY_CHECK = 4096
# The reader is for reasoning from the guardrail/predicate/agent CODE — NOT for reading
# the seed ANSWER KEY. Deny the real secret/credential fixtures so the scientist can't
# hardcode the live secret into a (public-only, non-transferable) exfil, and so those
# values can't leak into the external strategist prompt/logs/checkpoints (P0-E).
_DENY_NAMES = {"secret.txt", "api_keys.txt", "credentials.json", "compromised.txt"}
_DENY_PAT = __import__("re").compile(r"(secret|credential|api[_-]?key|password|token)", __import__("re").I)
_DENY_EXT = {".pem", ".key", ".p12", ".pfx", ".env"}


def _looks_binary(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            chunk = f.read(_BINARY_CHECK)
        return b"\x00" in chunk
    except Exception:
        return True


class SdkReader:
    """Read-only, root-confined file reader with pagination and a total budget."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        total_budget_chars: int | None = None,
        per_file_chars: int = 40_000,
    ) -> None:
        self.roots: list[Path] = []
        for r in roots:
            try:
                rp = Path(r).resolve()
            except Exception:
                continue
            if rp.is_dir():
                self.roots.append(rp)
        self.total_budget_chars = int(
            total_budget_chars or os.environ.get("SCI_READ_BUDGET_CHARS", "100000")
        )
        self.per_file_chars = int(per_file_chars)

    # -- path resolution / guards -------------------------------------------

    def _resolve(self, rel: str) -> tuple[Path | None, str | None]:
        """Resolve ``rel`` against the roots. Returns (path, error)."""
        rel = str(rel or "").strip().replace("\\", "/")
        if not rel:
            return None, "empty path"
        if rel.startswith(("/", "~")) or (len(rel) > 1 and rel[1] == ":"):
            return None, f"REFUSED: absolute paths are not allowed ({rel!r})"
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            return None, f"REFUSED: '..' traversal is not allowed ({rel!r})"
        if any(p == ".secrets" for p in parts):
            return None, f"REFUSED: .secrets is off-limits ({rel!r})"
        base = parts[-1].lower() if parts else ""
        if base in _DENY_NAMES or Path(base).suffix in _DENY_EXT or _DENY_PAT.search(base):
            return None, (f"REFUSED: {rel!r} is a secret/credential fixture (the answer key) — "
                          "read the guardrail/predicate/agent CODE, not the seed secret.")
        cand_rel = Path(*parts)
        for root in self.roots:
            cand = (root / cand_rel).resolve()
            try:
                cand.relative_to(root)
            except ValueError:
                continue
            if cand.exists():
                return cand, None
        return None, f"not found under any allowed root: {rel!r}"

    # -- listing --------------------------------------------------------------

    def list_tree(self, rel: str = "", depth: int = 2, *, max_entries: int = 400) -> str:
        """Indented directory listing (dirs first, sizes for files)."""
        if rel:
            base, err = self._resolve(rel)
            if err:
                return err
            if not base.is_dir():
                return f"not a directory: {rel!r}"
            root_label = rel
        else:
            # virtual view: list every root's top level
            base = None
            root_label = ""
        lines: list[str] = []
        entries = 0
        truncated = False

        def walk(d: Path, prefix: str, lvl: int) -> None:
            nonlocal entries, truncated
            if lvl > depth or truncated:
                return
            try:
                kids = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            except Exception:
                return
            for k in kids:
                if k.name in _SKIP_DIRS or k.name.startswith("."):
                    continue
                if entries >= max_entries:
                    truncated = True
                    return
                entries += 1
                if k.is_dir():
                    lines.append(f"{prefix}{k.name}/")
                    walk(k, prefix + "  ", lvl + 1)
                else:
                    lines.append(f"{prefix}{k.name} ({k.stat().st_size:,}B)")

        if base is None:
            for root in self.roots:
                lines.append(f"{root.name}/")
                walk(root, "  ", 1)
        else:
            lines.append(f"{root_label.rstrip('/')}/")
            walk(base, "  ", 1)
        if truncated:
            lines.append(f"... (truncated at {max_entries} entries; list a subdirectory)")
        return "\n".join(lines)

    # -- reading ----------------------------------------------------------------

    def read_file(self, rel: str, *, offset: int = 1, limit: int | None = None) -> str:
        """Read one file (1-based line offset, optional line limit)."""
        p, err = self._resolve(rel)
        if err:
            return err
        if p.is_dir():
            return f"{rel!r} is a directory — use action=\"list\" instead"
        if _looks_binary(p):
            return f"REFUSED: {rel!r} looks binary; only text files are readable"
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"read error: {type(exc).__name__}: {exc}"
        lines = text.splitlines()
        total = len(lines)
        offset = max(1, int(offset or 1))
        limit = int(limit) if limit else total
        chunk = lines[offset - 1: offset - 1 + limit]
        body = "\n".join(chunk)
        if len(body) > self.per_file_chars:
            body = body[: self.per_file_chars] + "\n... (per-file char cap — paginate with offset/limit)"
        header = f"=== {rel} (lines {offset}-{offset - 1 + len(chunk)} of {total}) ==="
        if offset - 1 + len(chunk) < total:
            header += f"  [more: read again with offset={offset + len(chunk)}]"
        return f"{header}\n{body}"

    def read_many(self, requests: Iterable[dict | str]) -> str:
        """Batch read: each item is ``"path"`` or ``{"path": ..., "offset": ..., "limit": ...}``.
        Stops when the total budget is exhausted (reported, not silent)."""
        out: list[str] = []
        used = 0
        items = list(requests)[:12]
        for i, item in enumerate(items):
            if isinstance(item, str):
                item = {"path": item}
            rel = str(item.get("path", ""))
            chunk = self.read_file(rel, offset=int(item.get("offset", 1) or 1),
                                   limit=item.get("limit"))
            remaining = self.total_budget_chars - used
            if remaining <= 0:
                out.append(f"... (read budget exhausted; {len(items) - i} request(s) dropped — ask again next step)")
                break
            if len(chunk) > remaining:
                chunk = chunk[:remaining] + "\n... (read budget exhausted this step)"
            out.append(chunk)
            used += len(chunk)
        return "\n\n".join(out) if out else "(no readable files requested)"
