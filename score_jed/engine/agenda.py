"""Research agenda — the scientist's persistent research PLAN across episodes.

The LabNotebook stores atomic facts; the HypothesisGraph stores falsifiable
claims; the FailureIndex stores dead ends. None of them stores *what the
scientist intends to do next* — so every episode used to restart its reasoning
from a blank plan. This module is the missing carrier: an explicit, persistent,
LLM-writable research agenda with four sections (open questions, active plans,
next experiments, dead ends), rendered into every prompt and updated through
structured ops in the scientist's JSON output ("agenda": [...]).

Discipline: items are short (240 chars), deduped by normalized text, capped per
section (oldest open dropped first), and the file is rewritten atomically per
mutation — a crash can lose at most the last op.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

SECTIONS = ("open_questions", "active_plans", "next_experiments", "dead_ends")
_PREFIX = {"open_questions": "q", "active_plans": "p", "next_experiments": "e", "dead_ends": "d"}
_CAP = 12
_WORD = re.compile(r"[a-z0-9_]+")


def _norm(text: str) -> str:
    return " ".join(sorted(_WORD.findall((text or "").lower())))[:160]


@dataclass
class AgendaItem:
    iid: str
    section: str
    text: str
    status: str = "open"            # open | done | dropped
    evidence: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


class ResearchAgenda:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._items: dict[str, AgendaItem] = {}
        self._ctr = 0
        if self.path.is_file():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                for raw in d.get("items", []):
                    it = AgendaItem(**raw)
                    self._items[it.iid] = it
                self._ctr = int(d.get("ctr", 0))
            except Exception:
                self._items, self._ctr = {}, 0

    # -- queries ------------------------------------------------------------

    def items(self, section: str, *, open_only: bool = True) -> list[AgendaItem]:
        out = [it for it in self._items.values() if it.section == section]
        if open_only:
            out = [it for it in out if it.status == "open"]
        return sorted(out, key=lambda it: it.ts)

    # -- mutation -------------------------------------------------------------

    def add(self, section: str, text: str, *, evidence: Iterable[str] = ()) -> str | None:
        text = (text or "").strip()[:240]
        if section not in SECTIONS or len(text) < 8:
            return None
        norm = _norm(text)
        for it in self._items.values():
            if it.section == section and it.status == "open" and _norm(it.text) == norm:
                return it.iid                       # near-duplicate: keep the original
        self._ctr += 1
        iid = f"{_PREFIX[section]}-{self._ctr:04d}"
        self._items[iid] = AgendaItem(iid=iid, section=section, text=text,
                                      evidence=[str(e) for e in evidence])
        # cap: drop the oldest OPEN items first (never silently drop done history)
        open_items = self.items(section)
        for it in open_items[: max(0, len(open_items) - _CAP)]:
            it.status = "dropped"
        self.save()
        return iid

    def _set(self, iid: str, status: str | None = None, text: str | None = None) -> bool:
        it = self._items.get(str(iid or ""))
        if it is None:
            return False
        if status is not None:
            it.status = status
        if text:
            it.text = text.strip()[:240]
        it.ts = time.time()
        self.save()
        return True

    def apply_ops(self, ops: Iterable[dict]) -> list[str]:
        """Apply the LLM's structured agenda ops; return human-readable feedback notes."""
        notes: list[str] = []
        for op in list(ops or [])[:6]:
            if not isinstance(op, dict):
                continue
            kind = str(op.get("op", "")).lower()
            if kind == "add":
                evidence = op.get("evidence", ())
                if isinstance(evidence, str):
                    evidence = [evidence]
                iid = self.add(str(op.get("section", "")), str(op.get("text", "")),
                               evidence=evidence)
                notes.append(f"added {iid}" if iid else
                             f"rejected add (unknown section {op.get('section')!r} or too short)")
            elif kind in ("done", "drop"):
                ok = self._set(op.get("id"), status="done" if kind == "done" else "dropped")
                notes.append(f"{kind} {op.get('id')}" if ok else f"unknown id {op.get('id')!r}")
            elif kind == "update":
                ok = self._set(op.get("id"), text=str(op.get("text", "")))
                notes.append(f"updated {op.get('id')}" if ok else f"unknown id {op.get('id')!r}")
            else:
                notes.append(f"unknown op {kind!r}")
        return notes

    # -- render / persist -------------------------------------------------------

    def render(self, budget_chars: int = 3000) -> str:
        if not any(it.status == "open" for it in self._items.values()):
            return "(empty — build it as you learn: open questions, plans, next experiments, dead ends)"
        lines = ["YOUR RESEARCH AGENDA (persistent across episodes — a PLAN/observations, not evidence; "
                 "only trial-backed experiments confirm or refute. Update with the \"agenda\" ops; "
                 "start from YOUR OWN plan, not from scratch):"]
        for section in SECTIONS:
            open_items = self.items(section)
            if not open_items:
                continue
            lines.append(f"  {section.replace('_', ' ').upper()}:")
            for it in open_items:
                lines.append(f"    {it.iid}: {it.text}")
        out = "\n".join(lines)
        if len(out) > budget_chars:
            out = out[:budget_chars] + "\n  ... (agenda truncated)"
        return out

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"ctr": self._ctr,
                                       "items": [asdict(it) for it in self._items.values()]},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:
            pass

    @classmethod
    def load(cls, path: str | Path) -> "ResearchAgenda":
        return cls(path)
