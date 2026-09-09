"""Compact prior: the distilled bridge from Research Lab -> 9000 s submission.

The research mode explores for a long time and writes a small, self-contained
``CompactPrior`` — the highest-EV attack *genomes* plus the EXACT messages that
actually scored and their measured stats. The submission mode loads it and starts
adaptation from these proven candidates instead of from a cold seed bank.

Design goals:
  * self-contained + tiny (bakeable into the Kaggle submission, no external deps);
  * carries BOTH the genome (re-renderable / mutatable) and the exact tested bytes
    (deployable as-is), so submission can either re-validate exact or adapt;
  * carries per-arm EV stats measured in research as an informative prior for the
    submission's Beta-Bernoulli posterior (research is a warm start, not truth —
    the live target may differ, so submission re-measures).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.contracts import AttackProgram, Step


@dataclass
class PriorArm:
    name: str
    family: str
    mechanism: str
    encoding: str
    steps: list[tuple[str, str]]        # genome step templates (text, intent)
    exact_messages: list[str]           # the exact bytes that scored in research
    severity_raw: float
    success_p: float
    mean_replay_s: float
    deploy_cell: str | None
    predicates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "family": self.family, "mechanism": self.mechanism,
            "encoding": self.encoding, "steps": [list(s) for s in self.steps],
            "exact_messages": self.exact_messages, "severity_raw": self.severity_raw,
            "success_p": self.success_p, "mean_replay_s": self.mean_replay_s,
            "deploy_cell": self.deploy_cell, "predicates": self.predicates,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PriorArm":
        return cls(
            name=d["name"], family=d.get("family", "generic"), mechanism=d.get("mechanism", "direct"),
            encoding=d.get("encoding", "plain"),
            steps=[tuple(s) for s in d.get("steps", [])],
            exact_messages=list(d.get("exact_messages", [])),
            severity_raw=float(d.get("severity_raw", 0.0)),
            success_p=float(d.get("success_p", 0.0)),
            mean_replay_s=float(d.get("mean_replay_s", 1.0)),
            deploy_cell=d.get("deploy_cell"),
            predicates=list(d.get("predicates", [])),
        )

    def to_program(self) -> AttackProgram:
        """Re-renderable genome (mutatable in the submission's adaptation phase)."""
        steps = tuple(Step(t, i) for (t, i) in self.steps) or (Step(self.exact_messages[0] if self.exact_messages else "", "prior"),)
        return AttackProgram(
            name=self.name, steps=steps, family=self.family, mechanism=self.mechanism,
            encoding=self.encoding, source="prior",
        )

    def to_exact_program(self) -> AttackProgram | None:
        """Program whose steps ARE the exact tested bytes (no placeholders).

        Rendering this reproduces the SAME messages that scored in research, so the
        submission can re-validate the exact bytes it will deploy — not a fresh-nonce
        variant of the template (which may score differently). This is the arm the
        submission actually ships.
        """
        if not self.exact_messages:
            return None
        steps = tuple(Step(m, f"exact_{i}") for i, m in enumerate(self.exact_messages))
        return AttackProgram(
            name=self.name + "#exact", steps=steps, family=self.family, mechanism=self.mechanism,
            encoding=self.encoding, source="prior_exact",
        )


@dataclass
class CompactPrior:
    version: str = "1"
    target: str = ""
    created_at: str = ""
    arms: list[PriorArm] = field(default_factory=list)
    strategies: list[dict] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "target": self.target, "created_at": self.created_at,
            "arms": [a.to_dict() for a in self.arms], "strategies": self.strategies, "notes": self.notes,
        }

    def programs(self) -> list[AttackProgram]:
        return [a.to_program() for a in self.arms]

    def predicted_raw(self) -> float:
        """Rough official raw if each arm is deployed once: Σ p·severity + 2·unique cells."""
        cells = {a.deploy_cell for a in self.arms if a.deploy_cell}
        return sum(a.success_p * a.severity_raw for a in self.arms) + 2.0 * len(cells)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CompactPrior":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        pri = cls(version=d.get("version", "1"), target=d.get("target", ""),
                  created_at=d.get("created_at", ""), strategies=d.get("strategies", []),
                  notes=d.get("notes", ""))
        pri.arms = [PriorArm.from_dict(a) for a in d.get("arms", [])]
        return pri


def load_external(path: str | Path) -> CompactPrior:
    """Tolerantly load a prior from ANY JSON (our format or e.g. a V28 prior).

    Tries the native format first; otherwise scans for a list of arm-like objects
    (under ``arms``/``programs``/``top_ev_arms``/``candidates``) and extracts a
    name + the exact messages + whatever EV stats are present. Unknown fields are
    ignored. This lets the submission warm-start from an externally distilled
    prior (e.g. the notebook's v28_3_research_prior_compact.json) without a schema
    lock-in. Untested against a specific V28 file here — mapping is best-effort.
    """
    import json as _json

    d = _json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, dict) and isinstance(d.get("arms"), list) and d.get("version"):
        try:
            return CompactPrior.load(path)
        except Exception:
            pass

    def _messages(o: dict) -> list[str]:
        for k in ("exact_messages", "messages", "user_messages", "prompt_chain", "chain"):
            v = o.get(k)
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                return v
        return []

    rows: list = []
    if isinstance(d, dict):
        for k in ("arms", "programs", "top_ev_arms", "top_arms", "candidates"):
            if isinstance(d.get(k), list):
                rows = d[k]
                break
    elif isinstance(d, list):
        rows = d

    arms: list[PriorArm] = []
    for i, o in enumerate(rows):
        if not isinstance(o, dict):
            continue
        msgs = _messages(o)
        if not msgs:
            continue
        arms.append(PriorArm(
            name=str(o.get("name") or o.get("message_hash") or f"ext_{i}")[:60],
            family=str(o.get("family", "generic")), mechanism=str(o.get("mechanism", "direct")),
            encoding=str(o.get("encoding", "plain")),
            steps=[(m, f"ext{j}") for j, m in enumerate(msgs)], exact_messages=msgs,
            severity_raw=float(o.get("severity_raw", o.get("expected_severity_raw", 0.0)) or 0.0),
            success_p=float(o.get("success_p", o.get("success_probability", 0.5)) or 0.5),
            mean_replay_s=float(o.get("mean_replay_s", o.get("replay_mean_s", 1.0)) or 1.0),
            deploy_cell=o.get("deploy_cell") or o.get("cell_hash"),
            predicates=list(o.get("predicates", [])),
        ))
    return CompactPrior(target=str(d.get("target", "external")) if isinstance(d, dict) else "external",
                        arms=arms, notes=f"imported {len(arms)} arms from {Path(path).name}")


def append_archive(result: dict, path: str | Path, *, chunk: int = 0) -> int:
    """Append every eligible arm found this chunk to a persistent JSONL archive.

    This is the long research run's accumulating record of high-EV methods (the
    full library, not just the top-K prior). Append-only + deduped by (name,
    deploy_cell) so a 12h run builds a rich, resumable corpus.
    """
    import json as _json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple] = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                o = _json.loads(line)
                seen.add((o.get("name"), o.get("deploy_cell")))
            except Exception:
                continue
    written = 0
    with p.open("a", encoding="utf-8") as f:
        for n in result.get("nodes", {}).values():
            if not n.arm.eligible:
                continue
            key = (n.program.name, n.arm.deploy_cell)
            if key in seen:
                continue
            seen.add(key)
            f.write(_json.dumps({
                "chunk": chunk, "name": n.program.name, "family": n.program.family,
                "mechanism": n.program.mechanism, "encoding": n.program.encoding,
                "exact_messages": list(n.arm.messages), "severity_raw": round(n.arm.mean_severity_raw, 3),
                "success_p": round(n.arm.success_p, 4), "n_cold": n.arm.n_cold,
                "mean_replay_s": round(n.arm.mean_replay_s, 4), "deploy_cell": n.arm.deploy_cell,
                "predicates": sorted({pr.predicate for r in n.reports for pr in r.predicates}),
            }, ensure_ascii=False) + "\n")
            written += 1
    return written


def merge_priors(*priors: CompactPrior, top_k: int = 60) -> CompactPrior:
    """Merge priors (checkpoint/resume): dedupe by name, keep the higher-EV arm."""
    best: dict[str, PriorArm] = {}
    for pri in priors:
        for a in pri.arms:
            cur = best.get(a.name)
            score = a.success_p * a.severity_raw
            if cur is None or score > cur.success_p * cur.severity_raw:
                best[a.name] = a
    arms = sorted(best.values(), key=lambda a: a.success_p * a.severity_raw, reverse=True)[:top_k]
    tgt = next((p.target for p in priors if p.target), "merged")
    return CompactPrior(target=tgt, arms=arms, notes=f"merged {len(priors)} priors -> {len(arms)} arms")


def distill(result: dict, *, target: str, top_k: int = 40, strategies: list[dict] | None = None) -> CompactPrior:
    """Build a CompactPrior from a search result (keeps only eligible, high-EV arms)."""
    nodes = list(result.get("nodes", {}).values())
    scored = [n for n in nodes if n.arm.eligible]
    scored.sort(key=lambda n: (n.arm.projected_norm, n.arm.risk_adjusted_ev_per_s), reverse=True)
    arms: list[PriorArm] = []
    for n in scored[:top_k]:
        preds = sorted({p.predicate for r in n.reports for p in r.predicates})
        arms.append(PriorArm(
            name=n.program.name, family=n.program.family, mechanism=n.program.mechanism,
            encoding=n.program.encoding,
            steps=[(s.text, s.intent) for s in n.program.steps],
            exact_messages=list(n.arm.messages),
            severity_raw=round(n.arm.mean_severity_raw, 3),
            success_p=round(n.arm.success_p, 4),
            mean_replay_s=round(n.arm.mean_replay_s, 4),
            deploy_cell=n.arm.deploy_cell,
            predicates=preds,
        ))
    return CompactPrior(
        target=target, created_at=datetime.now(timezone.utc).isoformat(),
        arms=arms, strategies=strategies or [],
        notes=f"distilled from {len(nodes)} nodes; {len(arms)} eligible high-EV arms kept",
    )
