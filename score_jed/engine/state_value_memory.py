"""StateValueMemory — cross-episode game-state -> action value memory (MEMORY_CRITIQUE.md P1).

The existing 6 memory components model GUARDRAIL-RULE inference (JED-specific: which arg gets blocked).
Online network attack (CAGE4/CyberWheel, the B mode) needs a different memory: "in game-states like
THIS one, which action advanced the killchain / degraded THIS blue defender?" — a state-indexed value
memory so a LONG episode reuses cross-episode experience instead of re-deriving each turn (the whole
point of distillation: keep long episodes cheap).

Design (matches the project's memory principles):
  * APPEND-ONLY on disk (scope-keyed JSONL under SCI_MEMORY_DIR) — a bad write pollutes one row, never
    the accumulated whole; robust + auditable. This is also the PERSISTENCE layer (F4): unlike the
    per-cycle StructuredMemory and the tempdir components, this survives across runs/episodes.
  * BOUNDED at RENDER — render(state_sig, budget) is a read-time view (top-k actions by value for the
    current/similar state), never a mutation.
  * BENCHMARK-AGNOSTIC — a state signature is just a string the caller builds (host-state histogram for
    CAGE4, ATT&CK-stage set for CyberWheel, or a guardrail-map digest for JED). The value is a running
    mean reward-delta per (state_sig, action).

Not wired into the JED submission path (that path has no per-turn scalar reward); it is fed by the
online B loop (CybORGSandbox) — one update() per StepResult — and its render() is added to the scientist
prompt when the component is present, exactly like the other optional components.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


class _Stat:
    __slots__ = ("n", "mean", "last")
    def __init__(self, n: int = 0, mean: float = 0.0, last: float = 0.0):
        self.n = n; self.mean = mean; self.last = last

    def update(self, reward_delta: float) -> None:
        self.n += 1
        self.mean += (reward_delta - self.mean) / self.n   # running mean
        self.last = reward_delta


class StateValueMemory:
    """(state_sig, action) -> running mean reward-delta, persisted append-only per scope.

    scope: a stable string identifying the arena (e.g. "cage4/gnn", "cyberwheel", "jed/optimal") so
    experience from different defenders/guardrails does not cross-contaminate. path defaults to
    SCI_MEMORY_DIR/state_value/<scope>.jsonl; None => in-memory only (no persistence)."""

    def __init__(self, scope: str = "default", *, path: str | None = None) -> None:
        self.scope = str(scope or "default")
        if path is None:
            base = os.environ.get("SCI_MEMORY_DIR", "")
            path = os.path.join(base, "state_value", self.scope.replace("/", "_") + ".jsonl") if base else None
        self.path = Path(path) if path else None
        self._stats: dict[tuple[str, str], _Stat] = {}
        if self.path and self.path.is_file():
            self._load()

    # -- persistence (append-only) -----------------------------------------------------------------
    def _load(self) -> None:
        try:
            for ln in self.path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                # replay each recorded observation into the running stats (order-independent for mean)
                k = (r.get("s", ""), r.get("a", ""))
                st = self._stats.get(k) or _Stat()
                st.update(float(r.get("d", 0.0)))
                self._stats[k] = st
        except Exception:
            pass   # a corrupt tail loses at most recent rows; never blocks the run

    def update(self, state_sig: str, action: str, reward_delta: float) -> None:
        """Record one observation: taking `action` in a state like `state_sig` yielded `reward_delta`
        (e.g. red_reward gained this turn / blue-performance drop). Appends to disk; updates memory."""
        s = str(state_sig or "")[:200]; a = str(action or "")[:120]; d = float(reward_delta)
        k = (s, a)
        st = self._stats.get(k) or _Stat()
        st.update(d)
        self._stats[k] = st
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"s": s, "a": a, "d": d, "ts": time.time()}, ensure_ascii=False) + "\n")
            except Exception:
                pass   # persistence best-effort; memory still works in-process

    # -- bounded read-time view --------------------------------------------------------------------
    def render(self, state_sig: str = "", *, k: int = 6, budget_chars: int = 1600) -> str:
        """Top actions BY VALUE for the current state (exact match first, then any state — bounded).
        A read-time view over the append-only store; never mutates. Empty string when nothing learned."""
        if not self._stats:
            return ""
        s = str(state_sig or "")[:200]
        exact = [(a, st) for (ss, a), st in self._stats.items() if ss == s]
        # rank exact-state actions by mean value; if none, fall back to globally best actions (transfer)
        pool = exact if exact else [(a, st) for (_, a), st in self._stats.items()]
        pool.sort(key=lambda x: (x[1].mean, x[1].n), reverse=True)
        header = ("STATE-VALUE MEMORY (cross-episode: which actions paid off in states like this one — "
                  "exploit the proven ones, and note low/negative-value actions to avoid):")
        lines = [header]
        used = len(header)
        tag = "for THIS state" if exact else "(no exact match — best actions seen in ANY state, unverified here)"
        lines.append("  " + tag); used += len(lines[-1])
        for a, st in pool[:k]:
            line = f"  {a[:80]!r}: mean_reward={st.mean:+.2f} (n={st.n}, last={st.last:+.2f})"
            if used + len(line) + 1 > budget_chars:
                break
            lines.append(line); used += len(line) + 1
        return "\n".join(lines) if len(lines) > 2 else ""

    def best_action(self, state_sig: str) -> tuple[str, float] | None:
        """The highest-mean-value action recorded for this exact state (or None)."""
        cand = [(a, st.mean) for (ss, a), st in self._stats.items() if ss == str(state_sig or "")[:200]]
        return max(cand, key=lambda x: x[1]) if cand else None

    def __len__(self) -> int:
        return len(self._stats)


# ---- benchmark-agnostic state-signature helpers (callers build a compact, stable string) ----------
def cage4_state_sig(obs: dict) -> str:
    """Compact CAGE4 state signature: a histogram of host knowledge-states (K/S/U/R + decoy) — the
    thing that determines which red action advances the killchain. Order-stable, bounded."""
    from collections import Counter
    c: Counter = Counter()
    hosts = (obs or {}).get("host_states") or (obs or {}).get("hosts") or {}
    try:
        for h in (hosts.values() if isinstance(hosts, dict) else hosts):
            st = (h.get("state") if isinstance(h, dict) else str(h)) or "?"
            c[str(st)] += 1
    except Exception:
        pass
    return "cage4:" + ",".join(f"{k}={c[k]}" for k in sorted(c))


def attack_stage_sig(stages) -> str:
    """CyberWheel/ATT&CK: the SET of stages reached so far (recon/access/exfil/impact...)."""
    try:
        return "attck:" + ",".join(sorted({str(x) for x in (stages or [])}))
    except Exception:
        return "attck:?"
