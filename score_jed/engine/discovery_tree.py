"""Unified discovery backbone — breadth+depth search over restorable env states.

This is the piece the flat boundary scientist lacked. It resurrects the score
system's tree-search ideas (PUCT selection, a Go-Explore archive) but makes the
node a RESTORABLE ENV STATE the live scientist can be launched from, so
exploration is GLOBAL (expand any high-value state from history) rather than
episode-local. Two things it fixes directly:

  * BREADTH+DEPTH. `select_frontier` blends exploitation (return to a high-EV
    state and dig deeper) with exploration (fresh root starts + under-visited
    novelty). Pure depth collapses to the single-step exfil; pure breadth is the
    923-arms-no-peak we had. Both, together.

  * PARETO CAPTURE. `consider` keeps a recipe if it improves ANY axis — new
    family / new semantic behavior / new cell / higher EV / higher LCB / lower
    latency / higher reliability / lower fatal-rate — not only "new cell/family".
    So a faster or more-reliable variant of a known method is no longer discarded
    before it can be scored (the old capture gate's core flaw).

The SEMANTIC-BEHAVIOR key (tool sequence, predicate vector, sources, hop count,
stop mode, denial class) is the research-state identity — distinct from the
official score-cell (which only varies the benign URL). This is the archive
split the review asked for: behavior != cell.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

_URL_RE = re.compile(r"https?://[^\s'\"]+")
_NUM_RE = re.compile(r"\d+")


def mechanism_key(messages: Iterable[str]) -> str:
    """Template/mechanism signature — the attack SHAPE (finding 16). ONLY the URL and the
    public marker are normalized away (so URL-farming variants share a mechanism); control-
    flow-relevant DIGITS ARE KEPT (count=8 != count=9) and repeats are NOT collapsed (a
    2-message chain != a 1-message chain). Identity comes from structure, not over-
    normalized text."""
    parts = []
    for m in messages:
        s = _URL_RE.sub("{u}", str(m).lower())
        s = s.replace("secret_marker", "{marker}")
        s = re.sub(r"\s+", " ", s).strip()
        parts.append(s)                      # keep every message + its count digits
    return " || ".join(parts)[:400]


def _payload_class(events: list) -> tuple[str, str]:
    """Classify the exfil payload + its encoding — different MECHANISMS the coarse
    tool-sequence key would merge (finding 17). Returns (payload_class, encoding)."""
    for e in events:
        if e.get("name") in ("http.post", "email.send"):
            args = e.get("args") or {}
            data = str(args.get("data") or args.get("body") or "")
            low = data.lower()
            if "secret_marker" in low:
                pc = "sentinel"
            elif re.fullmatch(r"[A-Za-z0-9+/=]{16,}", data or ""):
                pc = "real_or_encoded"
            elif len(data) >= 12:
                pc = "real_or_other"
            elif data:
                pc = "benign"
            else:
                pc = "empty"
            if re.fullmatch(r"[0-9a-fA-F]{16,}", data or ""):
                enc = "hex"
            elif "%" in data and re.search(r"%[0-9a-fA-F]{2}", data):
                enc = "url"
            elif re.fullmatch(r"[A-Za-z0-9+/]{16,}={0,2}", data or ""):
                enc = "base64"
            else:
                enc = "plain"
            return pc, enc
    return "none", "plain"


def behavior_key(report: Any) -> tuple:
    """Semantic-behavior signature of a trace — the research-state identity (findings 8,17).

    Independent of the benign URL (which only mints official cells), but now DISTINGUISHES
    payload class (sentinel vs real/encoded) and encoding, so genuinely different exfil
    mechanisms are not merged. Still coarser than exact bytes (that's the exact-message id).
    """
    events = getattr(report, "tool_events", None) or []
    seq = tuple((e.get("name"), bool(e.get("ok"))) for e in events)
    sources = tuple(sorted({str(e.get("source")) for e in events if e.get("source")}))
    preds = tuple(sorted({p.predicate for p in getattr(report, "predicates", [])}))
    denials = tuple(sorted({str(d.get("reason", d.get("error", "")))[:24]
                            for d in (getattr(report, "denied", None) or [])}))
    stop = getattr(report, "stop_reason", "") or ""
    hops = getattr(report, "n_tool_hops", len(events))
    payload_class, encoding = _payload_class(events)
    return (seq, sources, preds, denials, stop, hops, payload_class, encoding)


@dataclass
class RecipeStats:
    """The multi-axis summary the tree reasons over (from an EVArmStats + report)."""
    name: str
    ev: float                    # risk-adjusted raw/sec (fills the finding budget cheaply)
    lcb: float                   # success_p lower bound (reliability floor)
    latency: float               # mean replay seconds (lower = better)
    reliability: float           # success_p
    fatal_rate: float            # share of trials that hit a fatal/harness error
    severity: float = 0.0        # mean severity_raw (the per-finding VALUE under the 2000 cap)
    projected_norm: float = 0.0  # the REAL objective (EVArmStats.projected_norm) — what floor_ev
                                  # and the tree's own reward are scaled in; NOT raw/sec or severity.
    cells: set[str] = field(default_factory=set)
    family: str = "generic"
    behavior: tuple = ()
    messages: tuple[str, ...] = ()


@dataclass
class ParetoEntry:
    ev: float
    lcb: float
    latency: float
    reliability: float
    fatal_rate: float
    name: str


@dataclass
class StateNode:
    key: str
    snapshot: Any                       # restorable live-env state (None = root)
    messages: tuple[str, ...] = ()      # prefix that reaches this state
    parent: "StateNode | None" = None
    depth: int = 0
    visits: int = 0
    value: float = 0.0                  # backed-up reward (projected_norm — the real objective)
    best_ev: float = 0.0                # best projected_norm reached at/below this state
    novelty: float = 1.0
    prefix_wall_s: float = 0.0          # measured replay seconds to REACH this state (for honest raw/sec)
    summary: str = ""                   # human-readable launch-state summary shown to the LLM (finding 2)
    reason: str = ""                    # why this state was archived (novelty / high-EV)

    @property
    def q(self) -> float:
        return self.value / max(1, self.visits)

    def backprop(self, reward: float) -> None:
        node: "StateNode | None" = self
        while node is not None:
            node.visits += 1
            node.value += reward
            node.best_ev = max(node.best_ev, reward)
            node = node.parent


class DiscoveryTree:
    def __init__(self, *, puct_c: float = 1.4, root_prob: float = 0.5,
                 max_states: int = 64, seed: int = 0) -> None:
        self.puct_c = float(puct_c)
        self.root_prob = float(root_prob)          # P(fresh breadth start) vs. depth continuation
        self.max_states = int(max_states)
        self.rng = random.Random(seed)
        self.root = StateNode(key="root", snapshot=None, messages=())
        self.states: list[StateNode] = []          # restorable-state archive (Go-Explore, bounded)
        self.behavior_pareto: dict[tuple, list[RecipeStats]] = {}   # real Pareto SET per behavior
        self.mechanisms: set[str] = set()                   # mechanism/template archive (URL-normalized shapes)
        self.cells: set[str] = set()
        self.families: set[str] = set()

    # -- selection (breadth + depth) -----------------------------------------

    def select_frontier(self) -> StateNode:
        """Pick the state to launch the next investigation from.

        With prob `root_prob` (or when the archive is empty) start FRESH from root
        — that is the breadth channel that discovers new mechanisms. Otherwise
        return to an archived high-value state via PUCT (Q exploit + UCB explore +
        novelty) — the depth channel that digs a promising method deeper.
        """
        # BUG FIX (found in code review): this used to ALSO increment visits here, on top of
        # backprop() incrementing it again on the same node a moment later (research_discovery.py
        # always calls node.backprop(reward) exactly once per episode right after this selection)
        # — double-counting the directly-selected node's visits relative to its ancestors (who
        # are only ever touched by backprop's walk-up), which biased q=value/visits low and
        # understated the UCB explore term for exactly the nodes being repeatedly launched from.
        # visit accounting belongs SOLELY to backprop (matches engine/uct.py's sibling tree).
        if not self.states or self.rng.random() < self.root_prob:
            return self.root
        total = max(2, sum(s.visits for s in self.states) + 1)
        log_n = math.log(total)

        def puct(s: StateNode) -> float:
            explore = self.puct_c * math.sqrt(log_n / max(1, s.visits))
            return s.q + 0.4 * s.best_ev + 0.3 * s.novelty + explore

        return max(self.states, key=puct)

    # -- Go-Explore archive ---------------------------------------------------

    def add_state(self, parent: StateNode, messages: tuple[str, ...], snapshot: Any,
                  ev: float, novelty: float, *, prefix_wall_s: float = 0.0,
                  summary: str = "", reason: str = "") -> StateNode | None:
        """Archive a restorable state the scientist can later be re-launched from.

        ``ev`` MUST be raw/sec (risk-adjusted EV per second), so PUCT compares it
        like-for-like with backprop rewards (finding 12). Bounded: once full, a new
        state only displaces the weakest incumbent, so memory stays flat over 24h.
        """
        if snapshot is None:
            return None
        node = StateNode(key=f"s{len(self.states)}_{hash(messages) & 0xffff:04x}",
                         snapshot=snapshot, messages=tuple(messages), parent=parent,
                         depth=(parent.depth + 1 if parent else 1), best_ev=ev, novelty=novelty,
                         prefix_wall_s=float(prefix_wall_s), summary=summary, reason=reason)
        if len(self.states) < self.max_states:
            self.states.append(node)
            return node
        weakest = min(self.states, key=lambda s: s.best_ev + 0.5 * s.novelty)
        if (ev + 0.5 * novelty) > (weakest.best_ev + 0.5 * weakest.novelty):
            self.states.remove(weakest)
            self.states.append(node)
            return node
        return None

    # -- Pareto capture -------------------------------------------------------

    @staticmethod
    def _dominates(a: RecipeStats, b: RecipeStats) -> bool:
        """a Pareto-dominates b: no worse on every axis and strictly better on one. SEVERITY and
        PROJECTED_NORM are first-class axes, so a high-severity multi-step chain is NOT dominated
        by a fast low-severity filler (it wins the per-finding-value / real-objective axes) —
        all survive on the frontier."""
        no_worse = (a.ev >= b.ev and a.lcb >= b.lcb and a.reliability >= b.reliability
                    and a.severity >= b.severity and a.projected_norm >= b.projected_norm
                    and a.latency <= b.latency and a.fatal_rate <= b.fatal_rate)
        strictly = (a.ev > b.ev or a.lcb > b.lcb or a.reliability > b.reliability
                    or a.severity > b.severity or a.projected_norm > b.projected_norm
                    or a.latency < b.latency or a.fatal_rate < b.fatal_rate)
        return no_worse and strictly

    def consider(self, stats: RecipeStats, floor_ev: float = 0.0,
                 hypothesis_evidence: bool = False) -> tuple[bool, list[str]]:
        """Keep a recipe if it is a REAL Pareto improvement on any axis (finding 13). Per
        behavior we maintain the actual set of NON-DOMINATED real recipes — not a synthetic
        max-per-axis 'super candidate' that may correspond to no real recipe. Capture on:
        new family / new mechanism / new cell / a non-dominated point on the Pareto frontier
        / supports-or-refutes an important hypothesis (refinement 9). Novelty-only captures
        are still gated by `floor_ev` so a dominated straggler isn't rewarded.
        """
        hard: list[str] = []
        novelty: list[str] = []
        if stats.family and stats.family not in self.families:
            hard.append("new_family")
        mech = mechanism_key(stats.messages)
        if mech and mech not in self.mechanisms:
            hard.append("new_mechanism")
        if stats.cells - self.cells:
            novelty.append(f"new_cell x{len(stats.cells - self.cells)}")
        if hypothesis_evidence:
            hard.append("hypothesis_evidence")
        frontier = self.behavior_pareto.get(stats.behavior, [])
        if not frontier:
            novelty.append("new_behavior")
        elif not any(self._dominates(e, stats) for e in frontier):
            # on the frontier -> a genuine improvement over at least one incumbent
            if stats.projected_norm > max(e.projected_norm for e in frontier):
                hard.append("higher_norm")          # a real improvement on the true objective
            elif stats.ev > max(e.ev for e in frontier):
                hard.append("higher_ev")
            elif stats.severity > max(e.severity for e in frontier):
                hard.append("higher_severity")     # a new high-value-per-finding chain
            else:
                hard.append("pareto_nondominated")
        reasons = list(hard)
        # UNIT FIX: floor_ev is projected_norm-scale (0.4× the corpus's current best projected_norm)
        # — comparing it against stats.ev (raw/sec) silently made this gate almost never bind.
        if novelty and (hard or stats.projected_norm >= floor_ev):
            reasons += novelty
        capture = bool(reasons)
        if capture:
            self.families.add(stats.family)
            self.mechanisms.add(mech)
            self.cells |= stats.cells
            # update the real Pareto SET: drop incumbents this recipe dominates, then add it
            kept = [e for e in frontier if not self._dominates(stats, e)]
            if not any(self._dominates(e, stats) for e in kept):
                kept.append(stats)
            self.behavior_pareto[stats.behavior] = kept
        return capture, reasons

    def pareto_frontier(self) -> list[RecipeStats]:
        """The union of all per-behavior non-dominated recipes (real recipes, ranked by EV)."""
        out: list[RecipeStats] = []
        for fr in self.behavior_pareto.values():
            out.extend(fr)
        return sorted(out, key=lambda s: s.ev, reverse=True)

    def stats_summary(self) -> dict:
        return {"states": len(self.states), "behaviors": len(self.behavior_pareto),
                "pareto": sum(len(v) for v in self.behavior_pareto.values()),
                "mechanisms": len(self.mechanisms), "cells": len(self.cells),
                "families": sorted(self.families)}

    # -- persistence (cross-block replayable archive) -------------------------
    # A frontier state is PREFIX MESSAGES + measured metadata, NOT the live env snapshot
    # (which is unserializable). On resume the scientist REPLAYS the prefix from the clean
    # root to reconstruct the state — so a promising frontier survives block/model swaps and
    # process restarts, instead of dying when run_target rebuilds the tree each block.

    @staticmethod
    def _tuplify(x):
        return tuple(DiscoveryTree._tuplify(e) for e in x) if isinstance(x, list) else x

    def save(self, path) -> None:
        import json
        from pathlib import Path as _P
        try:
            states = [{"key": s.key, "messages": list(s.messages), "depth": s.depth,
                       "visits": s.visits, "value": s.value, "best_ev": s.best_ev,
                       "novelty": s.novelty, "prefix_wall_s": s.prefix_wall_s,
                       "summary": s.summary, "reason": s.reason}
                      for s in self.states if s.messages]      # only replayable (non-root) states
            pareto = []
            for beh, recipes in self.behavior_pareto.items():
                pareto.append({"behavior": list(beh),
                               "recipes": [{"name": r.name, "ev": r.ev, "lcb": r.lcb,
                                            "latency": r.latency, "reliability": r.reliability,
                                            "fatal_rate": r.fatal_rate, "cells": sorted(r.cells),
                                            "family": r.family, "behavior": list(r.behavior),
                                            "messages": list(r.messages)} for r in recipes]})
            _P(path).write_text(json.dumps({
                "states": states, "mechanisms": sorted(self.mechanisms),
                "cells": sorted(self.cells), "families": sorted(self.families),
                "behavior_pareto": pareto}, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass

    def load(self, path) -> None:
        import json
        from pathlib import Path as _P
        p = _P(path)
        if not p.is_file():
            return
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        for sd in d.get("states", []):
            msgs = tuple(sd.get("messages", ()))
            if not msgs:
                continue
            # snapshot=None + parent=root: a REPLAYABLE frontier (investigate replays the prefix).
            node = StateNode(key=str(sd.get("key", f"s{len(self.states)}")), snapshot=None,
                             messages=msgs, parent=self.root, depth=int(sd.get("depth", 1)),
                             visits=int(sd.get("visits", 0)), value=float(sd.get("value", 0.0)),
                             best_ev=float(sd.get("best_ev", 0.0)), novelty=float(sd.get("novelty", 1.0)),
                             prefix_wall_s=float(sd.get("prefix_wall_s", 0.0)),
                             summary=str(sd.get("summary", "")), reason=str(sd.get("reason", "")))
            if len(self.states) < self.max_states:
                self.states.append(node)
        self.mechanisms |= set(d.get("mechanisms", []))
        self.cells |= set(d.get("cells", []))
        self.families |= set(d.get("families", []))
        for pe in d.get("behavior_pareto", []):
            beh = self._tuplify(pe.get("behavior", []))
            recipes = [RecipeStats(
                name=r.get("name", ""), ev=float(r.get("ev", 0.0)), lcb=float(r.get("lcb", 0.0)),
                latency=float(r.get("latency", 0.0)), reliability=float(r.get("reliability", 0.0)),
                fatal_rate=float(r.get("fatal_rate", 0.0)), cells=set(r.get("cells", [])),
                family=r.get("family", "generic"), behavior=self._tuplify(r.get("behavior", [])),
                messages=tuple(r.get("messages", ()))) for r in pe.get("recipes", [])]
            if recipes:
                self.behavior_pareto[beh] = recipes
