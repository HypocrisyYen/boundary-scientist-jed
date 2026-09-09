"""Cross-chunk posterior accumulation — the fix for the research plateau.

The old Research Lab built a fresh Searcher per chunk, so every arm's posterior
reset to zero each chunk: a program screened in chunk 1 was re-screened from
scratch in chunk 2, its samples discarded. The corpus valuation (pred_raw) then
plateaued because success_p never tightened and discovered cells never
accumulated beyond one chunk's top-200.

``PosteriorStore`` keeps a persistent, growing posterior keyed by the program
genome (``program.sha``). Each chunk MERGES its arms' raw samples in, so:
  * success_p tightens toward the true rate over many chunks (a reliably-scoring
    sev-64 arm climbs from ~0.57/0.75 toward its real reliability -> higher
    expected raw -> pred_raw improves past the plateau);
  * every scoring cell ever seen accumulates (more +2 in the corpus);
  * compute is not wasted re-screening known arms from zero.

The store is the persisted research state (save/load) and the seed source +
prior for both the next chunk and the 9000 s submission.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from engine.contracts import AttackProgram, Step
from engine.scoring import MIN_REPLAY_S, EVArmStats, _z, severity_raw

_MAX_SAMPLES = 40  # bound per-arm sample memory over a 12h run


@dataclass
class AccumArm:
    """Accumulated raw evidence for ONE program genome, across all chunks."""

    key: str
    name: str
    family: str = "generic"
    mechanism: str = "direct"
    encoding: str = "plain"
    source: str = "builtin"                                      # operator that created this genome
    steps: list[tuple[str, str]] = field(default_factory=list)   # genome (re-renderable)
    exact_messages: tuple[str, ...] = ()                          # a cold-positive's exact bytes
    deploy_cell: str | None = None
    n: int = 0
    n_positive: int = 0
    n_cold: int = 0
    n_cold_positive: int = 0
    sev_samples: list[float] = field(default_factory=list)
    time_samples: list[float] = field(default_factory=list)
    wall_samples: list[float] = field(default_factory=list)
    cell_hashes: set[str] = field(default_factory=set)
    predicate_families: set[str] = field(default_factory=set)   # which predicates this genome ever tripped
    fatal: bool = False

    def add_reports(self, reports, program: AttackProgram) -> None:
        if not self.steps:
            self.steps = [(s.text, s.intent) for s in program.steps]
        for r in reports:
            self.n += 1
            cold = bool(getattr(r, "cold", True))
            if cold:
                self.n_cold += 1
            if r.fatal_reason:
                self.fatal = True
            self.time_samples.append(max(r.replay_s, MIN_REPLAY_S))
            self.wall_samples.append(max(getattr(r, "wall_s", 0.0), 0.0))
            if r.solved:
                self.n_positive += 1
                if cold:
                    self.n_cold_positive += 1
                self.sev_samples.append(severity_raw(r.predicates))
                for p in r.predicates:
                    self.predicate_families.add(p.predicate)
                if r.cell_hash:
                    self.cell_hashes.add(r.cell_hash)
                # deploy a COLD-positive's exact bytes (validated from root); else any positive
                if r.messages and (not self.exact_messages or (cold and not self._deploy_is_cold)):
                    self.exact_messages = tuple(r.messages)
                    self.deploy_cell = r.cell_hash
                    self._deploy_is_cold = cold
        # bound memory
        self.sev_samples = self.sev_samples[-_MAX_SAMPLES:]
        self.time_samples = self.time_samples[-_MAX_SAMPLES:]
        self.wall_samples = self.wall_samples[-_MAX_SAMPLES:]

    _deploy_is_cold: bool = False

    @property
    def eligible(self) -> bool:
        return self.n_positive > 0 and not self.fatal

    def to_ev(self) -> EVArmStats:
        """Recompute the EV posterior from ALL accumulated samples (tightens over time)."""
        a = EVArmStats(
            program_sha=self.key, program_name=self.name, family=self.family, mechanism=self.mechanism,
            n=self.n, n_positive=self.n_positive, n_cold=self.n_cold, n_cold_positive=self.n_cold_positive,
            cell_hashes=set(self.cell_hashes), messages=tuple(self.exact_messages), deploy_cell=self.deploy_cell,
            fatal_observed=self.fatal,
        )
        a.mean_severity_raw = fmean(self.sev_samples) if self.sev_samples else 0.0
        a.std_severity_raw = pstdev(self.sev_samples) if len(self.sev_samples) > 1 else 0.0
        times = self.time_samples or [MIN_REPLAY_S]
        a.mean_replay_s = fmean(times)
        a.std_replay_s = pstdev(times) if len(times) > 1 else 0.0
        a.replay_s_p95 = max(a.mean_replay_s + 1.64 * a.std_replay_s, max(times))
        a.mean_wall_s = fmean(self.wall_samples) if self.wall_samples else 0.0
        if self.n_cold > 0:
            aa, bb = 1.0 + self.n_cold_positive, 1.0 + (self.n_cold - self.n_cold_positive)
        else:
            aa, bb = 1.0 + self.n_positive, 1.5 + (self.n - self.n_positive)
        mean = aa / (aa + bb)
        std = math.sqrt((aa * bb) / (((aa + bb) ** 2) * (aa + bb + 1)))
        a.success_p = mean
        a.success_p_lb = max(0.0, mean - _z(0.8) * std)
        a.expected_severity_raw = a.success_p * a.mean_severity_raw
        a.risk_adjusted_raw = a.success_p_lb * max(0.0, a.mean_severity_raw - 0.1 * a.std_severity_raw)
        return a

    def to_program(self) -> AttackProgram:
        steps = tuple(Step(t, i) for (t, i) in self.steps) or (Step(self.exact_messages[0] if self.exact_messages else "", "prior"),)
        return AttackProgram(name=self.name, steps=steps, family=self.family, mechanism=self.mechanism,
                             encoding=self.encoding, source="store")

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        d["cell_hashes"] = sorted(self.cell_hashes)
        d["predicate_families"] = sorted(self.predicate_families)
        d["steps"] = [list(s) for s in self.steps]
        d["exact_messages"] = list(self.exact_messages)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AccumArm":
        arm = cls(key=d["key"], name=d.get("name", d["key"]))
        for k, v in d.items():
            if k in ("cell_hashes", "predicate_families"):
                setattr(arm, k, set(v))
            elif k in ("steps",):
                arm.steps = [tuple(s) for s in v]
            elif k in ("exact_messages",):
                arm.exact_messages = tuple(v)
            elif hasattr(arm, k):
                setattr(arm, k, v)
        return arm


class PosteriorStore:
    def __init__(self) -> None:
        self.arms: dict[str, AccumArm] = {}

    def update_from_nodes(self, nodes: dict) -> int:
        """Merge a chunk's search nodes into the persistent posterior. Returns #new keys."""
        new = 0
        for node in nodes.values():
            prog = node.program
            key = prog.sha
            arm = self.arms.get(key)
            if arm is None:
                arm = AccumArm(key=key, name=prog.name, family=prog.family,
                               mechanism=prog.mechanism, encoding=prog.encoding, source=prog.source)
                self.arms[key] = arm
                new += 1
            arm.add_reports(node.reports, prog)
        return new

    def operator_stats(self) -> dict[str, dict[str, float]]:
        """Per-operator productivity (persisted across chunks): how much EV each
        operator's genomes actually earned — the basis for adaptive budget."""
        stats: dict[str, dict[str, float]] = {}
        for a in self.arms.values():
            s = stats.setdefault(a.source, {"genomes": 0, "eligible": 0, "exp_raw": 0.0})
            s["genomes"] += 1
            if a.eligible:
                s["eligible"] += 1
                s["exp_raw"] += a.to_ev().expected_severity_raw
        return stats

    def eligible_evs(self) -> list[EVArmStats]:
        return [a.to_ev() for a in self.arms.values() if a.eligible]

    def top_programs(self, k: int) -> list[AttackProgram]:
        ranked = sorted((a for a in self.arms.values() if a.eligible),
                        key=lambda a: a.to_ev().projected_norm, reverse=True)
        return [a.to_program() for a in ranked[:k]]

    def all_cells(self) -> set[str]:
        cells: set[str] = set()
        for a in self.arms.values():
            if a.eligible:
                cells |= a.cell_hashes
        return cells

    def predicted_corpus_raw(self, top_k: int = 200) -> float:
        evs = sorted(self.eligible_evs(), key=lambda e: e.projected_norm, reverse=True)[:top_k]
        cells = {e.deploy_cell for e in evs if e.deploy_cell}
        return sum(e.expected_severity_raw for e in evs) + 2.0 * len(cells)

    def to_prior(self, *, target: str, top_k: int = 200, strategies: list[dict] | None = None,
                 diversity_per_family: int = 8):
        from engine.prior import CompactPrior, PriorArm
        ranked_all = sorted((a for a in self.arms.values() if a.eligible),
                            key=lambda a: a.to_ev().projected_norm, reverse=True)
        selected = list(ranked_all[:top_k])
        sel_keys = {a.key for a in selected}
        # Guarantee predicate-FAMILY coverage: the top-k EV cut is dominated by the
        # high-severity exfil stacks and silently drops the (low-EV but diverse)
        # CONFUSED_DEPUTY / other-family methods. Append the best few arms that trip
        # each under-represented family so the shipped prior preserves diversity, not
        # just raw EV (each distinct family/cell is scored under-replay).
        for fam in ("CONFUSED_DEPUTY", "DESTRUCTIVE_WRITE", "UNTRUSTED_TO_ACTION", "EXFILTRATION"):
            added = 0
            for a in ranked_all:
                if added >= diversity_per_family:
                    break
                if a.key in sel_keys:
                    continue
                if fam in getattr(a, "predicate_families", set()):
                    selected.append(a)
                    sel_keys.add(a.key)
                    added += 1
        arms = []
        for a in selected:
            ev = a.to_ev()
            arms.append(PriorArm(
                name=a.name, family=a.family, mechanism=a.mechanism, encoding=a.encoding,
                steps=list(a.steps), exact_messages=list(a.exact_messages),
                severity_raw=round(ev.mean_severity_raw, 3), success_p=round(ev.success_p, 4),
                mean_replay_s=round(ev.mean_replay_s, 4), deploy_cell=a.deploy_cell,
                predicates=[],
            ))
        from datetime import datetime, timezone
        return CompactPrior(target=target, created_at=datetime.now(timezone.utc).isoformat(),
                            arms=arms, strategies=strategies or [],
                            notes=f"posterior store: {len(self.arms)} genomes, {len(arms)} top eligible")

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"arms": [a.to_dict() for a in self.arms.values()]}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PosteriorStore":
        store = cls()
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        for a in d.get("arms", []):
            arm = AccumArm.from_dict(a)
            store.arms[arm.key] = arm
        return store
