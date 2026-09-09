"""Experience replay buffer + meta-learner (P2/P3).

The FailureIndex stores dead ends, the HypothesisGraph stores claims, the strategy
bank stores generalized successes — but NONE of them stores the raw *trajectories*
of past episodes. So the scientist forgets what it actually DID after ~10 compressed
history steps ("失憶"). This module is the missing replay buffer:

  * :func:`ReplayBuffer.record_episode` — append ONE immutable episode summary (the
    actions taken, whether each scored, families found, outcome) per scope;
  * :func:`ReplayBuffer.render` — retrieve the episodes most relevant to the current
    query (token overlap + scope hard filter) and surface them in the prompt, so a
    pattern found 3 hours ago is NOT re-discovered from scratch;
  * :func:`action_productivity` — the META-LEARNER: per-action-type fire rate from
    the buffered episodes, so the scientist sees WHICH exploration moves actually
    produce scoring recipes and can allocate its next decision accordingly.

Append-only JSONL per (model, guardrail) scope; a crash loses at most one episode.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

_WORD = re.compile(r"[a-z0-9_]+")
_STOP = {"the", "a", "an", "is", "it", "to", "of", "and", "or", "in", "on", "for", "with",
         "that", "this", "was", "its", "at", "by", "be", "as", "if", "then", "now", "using",
         "then", "use", "send", "post", "http", "data", "url"}


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 1}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / (1.0 + len(ta | tb))


class ReplayBuffer:
    def __init__(self, path: str | Path, *, max_episodes: int = 2000) -> None:
        self.path = Path(path)
        self.max_episodes = int(max_episodes)
        self._episodes: list[dict[str, Any]] = []
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    self._episodes.append(json.loads(line))
                except Exception:
                    continue
        self._episodes = self._episodes[-self.max_episodes:]

    # -- write (append-only) -------------------------------------------------

    def record_episode(self, *, scope: str, actions: Iterable[dict], families: Iterable[str] = (),
                       outcome: str = "no_score", note: str = "") -> None:
        ep = {
            "ts": time.time(),
            "scope": str(scope),
            "actions": [dict(a) for a in actions][:40],
            "families": sorted(families),
            "outcome": str(outcome),
            "note": str(note)[:240],
        }
        self._episodes.append(ep)
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes:]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(ep, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def count(self) -> int:
        return len(self._episodes)

    # -- retrieve (relevance-ranked, scope-filtered) -------------------------

    def render(self, query: str, scope: str = "", *, k: int = 3, budget_chars: int = 2500) -> str:
        pool = [e for e in self._episodes
                if not scope or e.get("scope") == scope]
        pool.sort(key=lambda e: (_overlap(query, e.get("note", "")),
                                 sum(1 for a in e.get("actions", []) if a.get("fired")),
                                 float(e.get("ts", 0))), reverse=True)
        picked = pool[:k]
        if not picked:
            return "(no past episodes recorded in this scope yet)"
        lines = ["PAST EXPERIENCES (this scope — what you did before, so you build on it not repeat it):"]
        for e in picked:
            acted = "; ".join(
                f"{a.get('action')}({str(a.get('msg') or a.get('query') or '')[:40]})"
                + ("✓" if a.get("fired") else "") for a in e.get("actions", [])[:6])
            fams = ",".join(e.get("families", [])) or "-"
            lines.append(f"  [{e['outcome']} fam={fams}] {acted}")
        out = "\n".join(lines)
        if len(out) > budget_chars:
            out = out[:budget_chars] + "\n  ... (truncated)"
        return out


def action_productivity(episodes: Iterable[dict]) -> dict[str, dict[str, float]]:
    """META-LEARNER: per-action-type fire rate across buffered episodes.
    Returns {action_type: {"n", "fired", "rate"}} sorted by rate desc — the scientist's
    "what exploration actually works" table. Accepts either a list of EPISODES (each with
    an ``actions`` list) or a flat list of action dicts."""
    agg: dict[str, dict[str, float]] = {}
    for item in episodes:
        acts = item.get("actions", []) if isinstance(item, dict) and "actions" in item else [item]
        for a in acts:
            t = str(a.get("action") or "?")
            s = agg.setdefault(t, {"n": 0.0, "fired": 0.0})
            s["n"] += 1.0
            if a.get("fired"):
                s["fired"] += 1.0
    for s in agg.values():
        s["rate"] = s["fired"] / max(1.0, s["n"])
    return dict(sorted(agg.items(), key=lambda kv: (kv[1]["rate"], kv[1]["n"]), reverse=True))


def render_productivity(prod: dict[str, dict[str, float]], *, budget: int = 6) -> str:
    """Human-readable meta-learning block for the prompt."""
    if not prod:
        return "(not enough episodes yet to learn which actions pay off)"
    lines = ["EXPLORATION EFFECTIVENESS (meta-learning — which of your moves actually SCORE, "
             "from past episodes; invest your next decisions accordingly):"]
    for t, s in list(prod.items())[:budget]:
        lines.append(f"  {t:12s} fired {s['fired']:.0f}/{s['n']:.0f} = {s['rate']*100:.0f}%")
    return "\n".join(lines)
