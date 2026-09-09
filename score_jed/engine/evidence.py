"""Immutable, append-only evidence store — the single source of truth for research.

Principle: *hardcode HOW to do reliable science, never WHICH attack is best.* The
LLM strategist may propose claims, explain traces, and design experiments, but it
NEVER sets a verdict. Every executed replay is written once as an immutable
``Trial``; every controlled comparison is an ``Experiment`` that references its
trials; and the posterior, hypothesis verdicts, Pareto frontier and leaderboard
are DERIVED VIEWS recomputable from the raw trials. Nothing is overwritten.

Two scopes/profiles are kept strictly separate (keyed by ``Scope.key``):
  * ``official`` — 8192 ctx / 256 tok / greedy, matching the evaluator; ONLY these
    trials feed the formal posterior / Pareto / knowledge base;
  * ``fast_screen`` — cheap exploration; never enters the formal posterior.

A trial also records its sequence ``regime`` (env/agent/backend/process-cold, hot,
interleaved), because a greedy model is NOT fully deterministic across cache /
backend / request-history state.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

MIN_LATENCY_S = 0.05
REGIMES = ("env_cold", "agent_cold", "backend_cold", "process_cold", "hot", "interleaved")
PROFILES = ("official", "fast_screen")

# severity weight, mirroring aicomp_sdk.scoring.SEVERITY_W
_SEV_W = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _z(conf: float) -> float:
    return {0.8: 0.8416, 0.9: 1.2816, 0.95: 1.6449}.get(conf, 0.8416)


# --------------------------------------------------------------------------- Scope

@dataclass(frozen=True)
class Scope:
    """The full environment scope every trial is bound to. Evidence from one scope
    must NOT be treated as a fact in another (findings 7,10).

    The FULL block is a hard filter: a greedy model is still not bit-reproducible
    across a different context window, token budget, tool-hop cap, backend, chat
    template, model file, SDK build or guardrail build — so each is part of the
    identity, not an annotation. Populate them with :func:`build_scope`."""
    model: str = ""
    guardrail: str = ""
    profile: str = "official"            # official | fast_screen
    n_ctx: int = 0
    max_new_tokens: int = 0
    max_tool_hops: int = 0
    backend: str = "llama.cpp"
    chat_template: str = ""               # chat-template fingerprint (decoding boundary)
    seed: int = 0
    model_file_hash: str = ""             # cheap gguf fingerprint (path|size|mtime)
    sdk_version: str = ""                 # aicomp_sdk scoring-contract source hash
    guardrail_version: str = ""           # the specific guardrail module's source hash
    code_hash: str = ""                   # score_jed engine source hash

    @property
    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf-8")).hexdigest()[:16]

    @property
    def is_official(self) -> bool:
        return self.profile == "official"

    def matches(self, other: "Scope", *, keys: tuple[str, ...] = ("model", "guardrail", "profile")) -> bool:
        """Hard scope filter over the given fields (default: model+guardrail+profile)."""
        return all(getattr(self, k) == getattr(other, k) for k in keys)


def file_fingerprint(path: Any) -> str:
    """CHEAP identity of a large binary (e.g. a 12GB .gguf) — sha of name|size|mtime,
    NEVER a full read. Enough to notice the weights changed under us; not a content hash."""
    try:
        p = Path(path)
        st = p.stat()
        return hashlib.sha256(f"{p.name}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def source_hash(paths: Iterable[Any]) -> str:
    """Content hash of a set of (small) source files — the version of a code contract
    (the scoring/predicate SDK, a guardrail module). Missing files are skipped."""
    h = hashlib.sha256()
    got = False
    for path in paths:
        try:
            h.update(Path(path).read_bytes())
            got = True
        except Exception:
            continue
    return h.hexdigest()[:16] if got else ""


def build_scope(*, model: str, guardrail: str, profile: str, n_ctx: int, max_new_tokens: int,
                max_tool_hops: int, seed: int, backend: str = "llama.cpp", code_hash: str = "",
                sdk_root: Any = None, model_path: Any = None, chat_template: str = "",
                agent_kind: str = "") -> Scope:
    """Assemble the FULL scope block, computing the provenance fingerprints (model file,
    SDK scoring contract, the specific guardrail module, the chat-template/request builder)
    from the on-disk sources so every trial is bound to the exact build it was measured under
    (finding 7/10; Scope completeness). All fingerprinting is cheap (metadata for the weights,
    small-file content hash for code)."""
    sdk_ver = guard_ver = ""
    if sdk_root is not None:
        root = Path(sdk_root)
        # SDK version = the SCORING CONTRACT sources (predicates + cell signature + scoring),
        # so a change to how a breach is measured shows up as a new scope, not silently.
        sdk_ver = source_hash([
            root / "aicomp_sdk" / "core" / "predicates.py",
            root / "aicomp_sdk" / "core" / "cells.py",
            root / "aicomp_sdk" / "core" / "scoring.py",
        ])
        # guardrail_version = the exact module the run used (optimal | rules | allow).
        gmap = {"optimal": root / "aicomp_sdk" / "guardrails" / "optimal.py",
                "rules": root / "aicomp_sdk" / "attacks" / "baselines" / "guardrail_rules.py",
                "allow": root / "aicomp_sdk" / "attacks" / "baselines" / "guardrail_allow.py"}
        guard_ver = source_hash([gmap.get(guardrail, gmap["optimal"])])
        # chat-template fingerprint: the request/template builder module for this agent kind
        # (harmony for gpt_oss, gemma template for gemma) — a decode-boundary the exploit
        # depends on. Cheap source hash, no model load.
        if not chat_template and agent_kind:
            amap = {"gpt_oss": root / "aicomp_sdk" / "agents" / "gpt_oss_agent.py",
                    "gemma": root / "aicomp_sdk" / "agents" / "gemma4_agent.py",
                    "gemma_4": root / "aicomp_sdk" / "agents" / "gemma4_agent.py"}
            am = amap.get(agent_kind)
            if am is not None:
                chat_template = source_hash([am])
    return Scope(
        model=model, guardrail=guardrail, profile=profile, n_ctx=int(n_ctx),
        max_new_tokens=int(max_new_tokens), max_tool_hops=int(max_tool_hops), backend=backend,
        chat_template=str(chat_template or ""), seed=int(seed),
        model_file_hash=file_fingerprint(model_path) if model_path else "",
        sdk_version=sdk_ver, guardrail_version=guard_ver, code_hash=str(code_hash or ""),
    )


# --------------------------------------------------------------------------- Trial / Experiment

@dataclass
class Trial:
    trial_id: str
    program_sha: str          # template-level program sha (unrendered; groups template families)
    template_id: str          # the parameterized mechanism/template it instantiates
    nonce: int
    scope_key: str
    regime: str
    profile: str
    fired: bool               # a predicate fired on a successful call
    severity: float           # Σ severity weight for this trace
    latency_s: float          # full-chain replay seconds
    tool_count: int
    malformed: bool
    stop_reason: str
    cell_hash: str | None
    fatal: bool               # candidate-caused fatal (hard-exclude from EV)
    families: tuple[str, ...] = ()
    experiment_id: str | None = None
    exact_sha: str = ""       # SHA256 of rendered exact messages (for exact-message posterior)
    infra_fatal: bool = False # infrastructure-caused fatal (timeout/OOM — not the candidate's fault)
    is_diagnostic: bool = False  # sweep / paired-experiment probe — NEVER enters the deploy posterior
    ts: float = field(default_factory=time.time)


@dataclass
class Experiment:
    experiment_id: str
    hypothesis_id: str
    changed_variable: str
    controlled_vars: list[str]
    variants: list[str]
    trial_ids: list[str]
    primary_metric: str       # "fired" | "severity" | "latency_s" | ...
    effect_size: float
    verdict: str              # supports | contradicts | inconclusive
    scope_key: str
    ts: float = field(default_factory=time.time)


def _pred_get(p: Any, k: str, default: Any = None) -> Any:
    """SDK eval_predicates returns dicts; older fakes use attribute objects — support both."""
    if isinstance(p, dict):
        return p.get(k, default)
    return getattr(p, k, default)


def trial_from_report(report: Any, *, program_sha: str, template_id: str, nonce: int,
                      scope: Scope, regime: str, experiment_id: str | None = None,
                      exact_messages: tuple[str, ...] | None = None, infra_fatal: bool = False,
                      is_diagnostic: bool | None = None) -> Trial:
    """Build an immutable Trial from an oracle AttackReport + context.
    
    ``exact_messages``, if given, is used to compute the EXACT rendered message
    hash (``exact_sha``) for exact-message posterior grouping. Without it the
    caller must provide ``exact_sha`` via another mechanism."""
    preds = getattr(report, "predicates", None) or []
    ok = bool(getattr(report, "ok", False))
    fired = bool(preds) and ok
    sev = float(getattr(report, "severity_raw", 0.0) or
                sum(_SEV_W.get(int(_pred_get(p, "severity", 1)), 1) for p in preds))
    fatal_reason = getattr(report, "fatal_reason", None)
    stop = str(getattr(report, "stop_reason", "") or "")
    malformed = ("malform" in stop.lower()) or (fatal_reason is not None and "malform" in str(fatal_reason).lower())
    fams = tuple(sorted({_pred_get(p, "predicate", "") for p in preds}))
    exact_sha = ""
    if exact_messages is not None:
        import hashlib
        exact_sha = hashlib.sha256("␟".join(exact_messages).encode("utf-8")).hexdigest()[:16]
    # sweep / paired-experiment probes are DIAGNOSTIC (never ship): auto-detect from the
    # template id convention unless the caller states it explicitly.
    if is_diagnostic is None:
        tid = str(template_id or "")
        is_diagnostic = tid == "sweep" or tid.startswith("exp:")
    return Trial(
        trial_id=new_id("t"), program_sha=program_sha, template_id=template_id, nonce=int(nonce),
        scope_key=scope.key, regime=regime, profile=scope.profile,
        fired=fired, severity=sev,
        latency_s=float(getattr(report, "replay_s", 0.0) or getattr(report, "wall_s", 0.0) or 0.0),
        tool_count=int(getattr(report, "n_tool_hops", 0) or 0),
        malformed=bool(malformed), stop_reason=stop,
        cell_hash=getattr(report, "cell_hash", None),
        fatal=fatal_reason is not None, families=fams, experiment_id=experiment_id,
        exact_sha=exact_sha, infra_fatal=bool(infra_fatal), is_diagnostic=bool(is_diagnostic),
    )


# --------------------------------------------------------------------------- program registry

@dataclass
class ProgramMeta:
    """Deployable metadata for one EXACT message chain, keyed by exact_sha. This is what
    the trial log lacks (message TEXT + name/family/mechanism/deploy_cell) and what the
    prior / leaderboard need — so the EventStore can be the single source of truth."""
    exact_sha: str
    template_sha: str
    name: str
    family: str
    mechanism: str
    encoding: str
    exact_messages: tuple[str, ...]
    deploy_cell: str | None = None
    is_diagnostic: bool = False
    source: str = "unknown"          # P0-7: who produced this arm (boundary_sci / sweep / amp_url / nb_*)


def program_meta(program: Any, exact_messages: tuple[str, ...], *,
                 deploy_cell: str | None = None, is_diagnostic: bool = False) -> ProgramMeta:
    """Build a ProgramMeta from an AttackProgram + its rendered exact messages."""
    exact_sha = hashlib.sha256("␟".join(exact_messages).encode("utf-8")).hexdigest()[:16] if exact_messages else ""
    return ProgramMeta(
        exact_sha=exact_sha, template_sha=str(getattr(program, "sha", "")),
        name=str(getattr(program, "name", "")), family=str(getattr(program, "family", "generic")),
        mechanism=str(getattr(program, "mechanism", "direct")),
        encoding=str(getattr(program, "encoding", "plain")),
        exact_messages=tuple(exact_messages), deploy_cell=deploy_cell,
        is_diagnostic=bool(is_diagnostic),
        source=str(getattr(program, "source", "unknown")),
    )


# --------------------------------------------------------------------------- append-only store

class EventStore:
    """Append-only log of trials + experiments. Views recompute from these — the log
    is never rewritten (finding: 'immutable, recomputable evidence')."""

    def __init__(self, rd: str | Path) -> None:
        self.rd = Path(rd)
        self.rd.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.rd / "trials.jsonl"
        self.experiments_path = self.rd / "experiments.jsonl"
        self.programs_path = self.rd / "programs.jsonl"
        self.scopes_path = self.rd / "scopes.json"
        self._scopes: dict[str, dict] = {}
        if self.scopes_path.is_file():
            try:
                self._scopes = json.loads(self.scopes_path.read_text(encoding="utf-8"))
            except Exception:
                self._scopes = {}
        # in-memory set of already-registered exact_sha (append-once dedup)
        self._prog_seen: set[str] = set()
        if self.programs_path.is_file():
            for line in self.programs_path.read_text(encoding="utf-8").splitlines():
                try:
                    self._prog_seen.add(json.loads(line)["exact_sha"])
                except Exception:
                    continue

    def register_scope(self, scope: Scope) -> str:
        k = scope.key
        if k not in self._scopes:
            self._scopes[k] = asdict(scope)
            self.scopes_path.write_text(json.dumps(self._scopes, ensure_ascii=False, indent=1), encoding="utf-8")
        return k

    def append_trial(self, t: Trial) -> None:
        with self.trials_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")

    def append_experiment(self, e: Experiment) -> None:
        with self.experiments_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    def register_program(self, meta: ProgramMeta) -> None:
        """Append the deployable metadata for an exact message chain ONCE (keyed by
        exact_sha). This is what makes the log self-contained for the prior/leaderboard."""
        if not meta.exact_sha or meta.exact_sha in self._prog_seen:
            return
        self._prog_seen.add(meta.exact_sha)
        with self.programs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")

    def load_programs(self) -> dict[str, dict]:
        """exact_sha -> deployable metadata dict (last write wins on a re-registration)."""
        out: dict[str, dict] = {}
        if self.programs_path.is_file():
            for line in self.programs_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                out[d.get("exact_sha", "")] = d
        return out

    def load_trials(self, *, official_only: bool = False, scope_key: str | None = None) -> list[dict]:
        out: list[dict] = []
        if self.trials_path.is_file():
            for line in self.trials_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if official_only and d.get("profile") != "official":
                    continue
                if scope_key is not None and d.get("scope_key") != scope_key:
                    continue
                out.append(d)
        return out

    def load_experiments(self, *, hypothesis_id: str | None = None) -> list[dict]:
        out: list[dict] = []
        if self.experiments_path.is_file():
            for line in self.experiments_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if hypothesis_id is not None and d.get("hypothesis_id") != hypothesis_id:
                    continue
                out.append(d)
        return out


# --------------------------------------------------------------------------- materialized posterior

@dataclass
class ArmEV:
    """A materialized EV, recomputed from trials — for one exact message OR one template."""
    key: str
    kind: str                 # "exact" | "template"
    n: int
    n_pos: int
    success_p: float
    success_p_lb: float
    mean_severity: float
    mean_latency: float
    latency_p95: float
    cells: set
    risk_adjusted_raw: float

    @property
    def risk_adjusted_ev_per_s(self) -> float:
        return self.risk_adjusted_raw / max(self.mean_latency, MIN_LATENCY_S)

    @property
    def eligible(self) -> bool:
        return self.n_pos > 0 and self.mean_severity > 0


def _ev_from_trials(key: str, kind: str, trials: list[dict]) -> ArmEV:
    """CANDIDATE-CAUSED fatals make the arm permanently ineligible (never silent drop).
    P0-6: infrastructure fatals are dropped from EV too (not counted as negative samples)."""
    candidate_fatal = any(t.get("fatal") and not t.get("infra_fatal") for t in trials)
    usable = [t for t in trials if not t.get("fatal")]
    lats = [max(float(t.get("latency_s", 0.0)), MIN_LATENCY_S) for t in usable] or [MIN_LATENCY_S]
    if candidate_fatal:
        return ArmEV(key=key, kind=kind, n=len(usable), n_pos=0,
                     success_p=0.0, success_p_lb=0.0, mean_severity=0.0,
                     mean_latency=MIN_LATENCY_S, latency_p95=max(lats),
                     cells=set(), risk_adjusted_raw=0.0)
    n = len(usable)
    pos = [t for t in usable if t.get("fired")]
    n_pos = len(pos)
    sevs = [float(t["severity"]) for t in pos] or [0.0]
    lats = [max(float(t["latency_s"]), MIN_LATENCY_S) for t in usable] or [MIN_LATENCY_S]
    cells = {t["cell_hash"] for t in pos if t.get("cell_hash")}
    # Beta(1,1) posterior mean + normal-approx lower bound (mirrors engine.scoring)
    a, b = n_pos + 1, (n - n_pos) + 1
    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    p_lb = max(0.0, mean - _z(0.8) * math.sqrt(var))
    mean_sev = fmean(sevs)
    std_sev = pstdev(sevs) if len(sevs) > 1 else 0.0
    mean_lat = fmean(lats)
    lat_p95 = max(mean_lat + 1.64 * (pstdev(lats) if len(lats) > 1 else 0.0), max(lats))
    risk_adjusted = p_lb * max(0.0, mean_sev - 0.1 * std_sev)
    return ArmEV(key=key, kind=kind, n=n, n_pos=n_pos, success_p=mean, success_p_lb=p_lb,
                 mean_severity=mean_sev, mean_latency=mean_lat, latency_p95=lat_p95,
                 cells=cells, risk_adjusted_raw=risk_adjusted)


def materialize_posterior(trials: list[dict], *, official_only: bool = True) -> dict[str, ArmEV]:
    """EXACT-message deployment posterior — what actually ships is judged by a
    byte-identical message's OWN trials (finding: template != exact)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        if official_only and t.get("profile") != "official":
            continue
        groups[t["program_sha"]].append(t)
    return {sha: _ev_from_trials(sha, "exact", ts) for sha, ts in groups.items()}


def materialize_exact_posterior(trials: list[dict], *, official_only: bool = True) -> dict[str, ArmEV]:
    """EXACT-MESSAGE deployment posterior — grouped by rendered bytes (``exact_sha``),
    not by template ``program_sha``. This is what actually ships: two URL-variant
    candidates of the same template are SEPARATE exact arms with their OWN trials."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        if official_only and t.get("profile") != "official":
            continue
        sha = str(t.get("exact_sha") or t.get("program_sha", ""))
        groups[sha].append(t)
    return {sha: _ev_from_trials(sha, "exact", ts) for sha, ts in groups.items()}


def materialize_template_posterior(trials: list[dict], *, official_only: bool = True) -> dict[str, ArmEV]:
    """TEMPLATE screening posterior — 'is this class worth exploring', across its
    exact variants. Never used for ship/deploy decisions."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        if official_only and t.get("profile") != "official":
            continue
        groups[t.get("template_id") or t["program_sha"]].append(t)
    return {tid: _ev_from_trials(tid, "template", ts) for tid, ts in groups.items()}


def _is_cold(regime: str) -> bool:
    return str(regime or "").endswith("_cold")


def _evarm_from_trials(key: str, kind: str, trials: list[dict], meta: dict, max_tool_hops: int):
    """Build ONE EVArmStats (drop-in for PosteriorStore arms) from a group of trials + its
    registry metadata. ``kind`` is "exact" (key=exact_sha) or "template" (key=program_sha).
    Candidate-caused fatals make the arm permanently ineligible; infrastructure fatals are
    excluded from EV without poisoning eligibility."""
    from engine.scoring import EVArmStats
    candidate_fatal = any(t.get("fatal") and not t.get("infra_fatal") for t in trials)
    # P0-6: BOTH candidate-caused and infrastructure fatals are excluded from attack EV (they are
    # not observations of the candidate's real behavior). Candidate fatals additionally make the
    # arm PERMANENTLY ineligible; infra fatals only drop the sample (tracked separately upstream).
    usable = [t for t in trials if not t.get("fatal")]
    n = len(usable)
    pos = [t for t in usable if t.get("fired")]
    n_pos = len(pos)
    n_cold = sum(1 for t in usable if _is_cold(t.get("regime")))
    n_cold_pos = sum(1 for t in pos if _is_cold(t.get("regime")))
    sevs = [float(t.get("severity", 0.0)) for t in pos] or [0.0]
    lats = [max(float(t.get("latency_s", 0.0)), MIN_LATENCY_S) for t in usable] or [MIN_LATENCY_S]
    cells = {t.get("cell_hash") for t in pos if t.get("cell_hash")}
    a, b = n_pos + 1, (n - n_pos) + 1
    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    p_lb = max(0.0, mean - _z(0.8) * math.sqrt(var))
    mean_sev = fmean(sevs)
    std_sev = pstdev(sevs) if len(sevs) > 1 else 0.0
    mean_lat = fmean(lats)
    std_lat = pstdev(lats) if len(lats) > 1 else 0.0
    p95 = max(mean_lat + 1.64 * std_lat, max(lats))
    risk_adjusted = p_lb * max(0.0, mean_sev - 0.1 * std_sev)
    zero = candidate_fatal
    arm = EVArmStats(
        program_sha=key,                            # identity = the grouping key (exact_sha or template sha)
        program_name=str(meta.get("name") or f"{kind}:{key}"),
        family=str(meta.get("family", "generic")), mechanism=str(meta.get("mechanism", "direct")),
        n=n, n_positive=n_pos, n_cold=n_cold, n_cold_positive=n_cold_pos,
        fatal_observed=candidate_fatal, fatal_reason=("candidate_fatal" if candidate_fatal else None),
        mean_severity_raw=(0.0 if zero else mean_sev), std_severity_raw=std_sev,
        mean_replay_s=mean_lat, std_replay_s=std_lat, replay_s_p95=p95, mean_wall_s=mean_lat,
        cell_hashes=cells, messages=tuple(meta.get("exact_messages", ()) or ()),
        deploy_cell=meta.get("deploy_cell"), max_tool_hops=int(max_tool_hops),
        success_p=(0.0 if zero else mean), success_p_lb=(0.0 if zero else p_lb),
        expected_severity_raw=(0.0 if zero else mean * mean_sev),
        risk_adjusted_raw=(0.0 if zero else risk_adjusted),
    )
    # predicate families this arm actually tripped (for prior family-diversity + focus).
    pfams: set[str] = set()
    for t in pos:
        pfams |= {f for f in (t.get("families") or []) if f}
    arm.predicate_families = pfams
    arm.encoding = str(meta.get("encoding", "plain"))
    return arm


def materialize_evarms(trials: list[dict], programs: dict[str, dict], *,
                       official_only: bool = True, include_diagnostic: bool = False,
                       max_tool_hops: int = 8) -> list:
    """MATERIALIZED DEPLOY POSTERIOR as EVArmStats — the drop-in replacement for
    PosteriorStore.eligible_evs() (P0-A). Recomputed from the immutable trial log:
      * grouped by exact_sha (byte-identical messages), so two URL-variants are SEPARATE arms;
      * DIAGNOSTIC (sweep / paired-experiment) trials are excluded — they never ship;
      * deployable metadata (name/family/mechanism/messages/deploy_cell) comes from the
        program registry, making the log self-contained for the leaderboard AND the prior.
    Returns arms sorted by risk-adjusted raw/sec (eligible ones first)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        if official_only and t.get("profile") != "official":
            continue
        if t.get("is_diagnostic") and not include_diagnostic:
            continue
        sha = str(t.get("exact_sha") or t.get("program_sha", ""))
        if not sha:
            continue
        groups[sha].append(t)
    arms = [_evarm_from_trials(sha, "exact", ts, programs.get(sha, {}), max_tool_hops)
            for sha, ts in groups.items()]
    arms.sort(key=lambda a: (a.eligible, a.projected_norm), reverse=True)
    return arms


# --------------------------------------------------------------------------- live materialized view

class _ArmWrap:
    """Duck-types PosteriorStore's AccumArm for the scoring loop: `.to_ev()` + `.eligible`."""
    __slots__ = ("_ev",)

    def __init__(self, ev):
        self._ev = ev

    def to_ev(self):
        return self._ev

    @property
    def eligible(self):
        return self._ev.eligible


class _TemplateArms:
    """Duck-types PosteriorStore.arms: `.get(program_sha)`, `.values()`, `len()`."""

    def __init__(self, mp: "MaterializedPosterior"):
        self._mp = mp

    def get(self, program_sha: str):
        self._mp._refresh()
        ev = self._mp._tmpl_cache.get(program_sha)
        return _ArmWrap(ev) if ev is not None else None

    def values(self):
        self._mp._refresh()
        return [_ArmWrap(ev) for ev in self._mp._tmpl_cache.values()]

    def __len__(self):
        return len(self._mp._tmpl_trials)


class MaterializedPosterior:
    """Trust-authoritative, INCREMENTAL replacement for PosteriorStore — backed by the
    immutable trial log (P0-A Steps 3-5). Nothing is overwritten; both views recompute
    from trials, and only DIRTY arms are recomputed per read (O(arms), not O(trials)):

      * ``.arms[program_sha].to_ev()`` -> TEMPLATE-aggregated EVArmStats (the scoring loop's
        per-recipe EV — matches the old store's per-program semantics);
      * ``.eligible_evs()`` / ``.all_cells()`` / ``.to_prior()`` -> EXACT-message DEPLOY arms
        (what actually ships — two URL-variants are SEPARATE arms).

    Diagnostic (sweep/experiment) and non-official trials never enter either view. If a
    ``scope_key`` is given, ONLY trials bound to that exact full scope are ingested —
    evidence from an older code hash, a different ctx/token profile, or another backend is
    HARD-REJECTED (P0-1), so the materialized posterior never mixes scopes."""

    def __init__(self, max_tool_hops: int = 8, *, scope_key: str = "") -> None:
        self.max_tool_hops = int(max_tool_hops)
        self.scope_key = str(scope_key or "")
        self._exact_trials: dict[str, list[dict]] = defaultdict(list)
        self._tmpl_trials: dict[str, list[dict]] = defaultdict(list)
        self._programs: dict[str, dict] = {}
        self._exact_cache: dict[str, Any] = {}
        self._tmpl_cache: dict[str, Any] = {}
        self._exact_dirty: set[str] = set()
        self._tmpl_dirty: set[str] = set()
        self.arms = _TemplateArms(self)

    # -- ingest (incremental, O(1) per trial) --------------------------------
    def ingest_trial(self, t: dict) -> None:
        if self.scope_key and str(t.get("scope_key") or "") != self.scope_key:
            return                                       # P0-1: hard reject other-scope evidence
        if t.get("profile") != "official" or t.get("is_diagnostic"):
            return
        es = str(t.get("exact_sha") or t.get("program_sha") or "")
        ps = str(t.get("program_sha") or "")
        if es:
            self._exact_trials[es].append(t); self._exact_dirty.add(es)
        if ps:
            self._tmpl_trials[ps].append(t); self._tmpl_dirty.add(ps)

    def ingest_program(self, meta: dict) -> None:
        sha = str(meta.get("exact_sha") or "")
        if sha:
            self._programs[sha] = meta
            self._exact_dirty.add(sha)

    def _refresh(self) -> None:
        for sha in self._exact_dirty:
            self._exact_cache[sha] = _evarm_from_trials(
                sha, "exact", self._exact_trials[sha], self._programs.get(sha, {}), self.max_tool_hops)
        self._exact_dirty.clear()
        for sha in self._tmpl_dirty:
            meta = next((m for m in self._programs.values() if m.get("template_sha") == sha), {})
            self._tmpl_cache[sha] = _evarm_from_trials(
                sha, "template", self._tmpl_trials[sha], {**meta, "exact_messages": ()}, self.max_tool_hops)
        self._tmpl_dirty.clear()

    # -- PosteriorStore-compatible reads (EXACT deploy view) -----------------
    def eligible_evs(self) -> list:
        self._refresh()
        return [a for a in self._exact_cache.values() if a.eligible]

    def all_cells(self) -> set:
        self._refresh()
        cells: set = set()
        for a in self._exact_cache.values():
            if a.eligible:
                cells |= a.cell_hashes
        return cells

    def operator_stats(self) -> dict:
        """Per-source productivity from the PROGRAM REGISTRY (P0-7): how many exact arms each
        operator (boundary_sci / boundary_sci_sweep / amp_url / nb_harmony / ...) produced and
        their total expected severity — the input for a real VOI budget allocator. Derived from
        the materialized exact arms (never the LLM's say-so)."""
        self._refresh()
        stats: dict[str, dict[str, float]] = {}
        for sha, ev in self._exact_cache.items():
            meta = self._programs.get(sha, {})
            src = str(meta.get("source") or "unknown")
            s = stats.setdefault(src, {"genomes": 0.0, "eligible": 0.0, "exp_raw": 0.0})
            s["genomes"] += 1.0
            if ev.eligible:
                s["eligible"] += 1.0
                s["exp_raw"] += float(ev.expected_severity_raw if hasattr(ev, "expected_severity_raw") else ev.mean_severity_raw)
        return stats

    def save(self, path) -> None:
        """The trial log is the source of truth; persist only a small human-readable snapshot."""
        try:
            self._refresh()
            top = sorted(self._exact_cache.values(), key=lambda a: a.projected_norm, reverse=True)[:50]
            Path(path).write_text(json.dumps({
                "materialized": True, "n_exact_arms": len(self._exact_cache),
                "n_templates": len(self._tmpl_trials),
                "top": [{"name": a.program_name, "norm": round(a.projected_norm, 1),
                         "rs": round(a.risk_adjusted_ev_per_s, 3),
                         "p": round(a.success_p, 3), "cells": len(a.cell_hashes)} for a in top],
            }, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass

    def to_prior(self, *, target: str, top_k: int = 200, strategies=None, diversity_per_family: int = 8):
        from engine.prior import CompactPrior, PriorArm
        self._refresh()
        ranked = sorted([a for a in self._exact_cache.values() if a.eligible],
                        key=lambda a: a.projected_norm, reverse=True)
        selected = list(ranked[:top_k])
        sel = {a.program_sha for a in selected}
        for fam in ("CONFUSED_DEPUTY", "DESTRUCTIVE_WRITE", "UNTRUSTED_TO_ACTION", "EXFILTRATION"):
            added = 0
            for a in ranked:
                if added >= diversity_per_family:
                    break
                if a.program_sha in sel:
                    continue
                if fam in getattr(a, "predicate_families", set()):
                    selected.append(a); sel.add(a.program_sha); added += 1
        arms = []
        for a in selected:
            msgs = list(a.messages)
            arms.append(PriorArm(
                name=a.program_name, family=a.family, mechanism=a.mechanism,
                encoding=getattr(a, "encoding", "plain"),
                steps=[(m, "prior") for m in msgs], exact_messages=msgs,
                severity_raw=round(a.mean_severity_raw, 3), success_p=round(a.success_p, 4),
                mean_replay_s=round(a.mean_replay_s, 4), deploy_cell=a.deploy_cell, predicates=[]))
        from datetime import datetime, timezone
        return CompactPrior(target=target, created_at=datetime.now(timezone.utc).isoformat(),
                            arms=arms, strategies=strategies or [],
                            notes=f"materialized deploy posterior: {len(self._exact_cache)} exact arms, {len(arms)} shipped")

    @classmethod
    def from_store(cls, event_store, *, max_tool_hops: int = 8, scope_key: str = "") -> "MaterializedPosterior":
        """Full rebuild from the immutable log (resume / integrity). With ``scope_key`` set,
        only trials of that exact scope are loaded — older code-hash / config evidence is
        excluded instead of silently contaminating the posterior (P0-1)."""
        mp = cls(max_tool_hops, scope_key=scope_key)
        for m in event_store.load_programs().values():
            mp.ingest_program(m)
        for t in event_store.load_trials():
            mp.ingest_trial(t)
        return mp


# --------------------------------------------------------------------------- deterministic analysis

def analyze_paired(metric_a: list[float], metric_b: list[float], *,
                   min_effect: float = 0.34) -> tuple[float, str]:
    """Deterministic verdict for a paired A/B on the pre-specified primary metric.

    ``effect`` = mean(A) - mean(B). Verdict is 'supports' if the effect magnitude
    clears ``min_effect`` (A better), 'contradicts' if it clears it the other way,
    else 'inconclusive'. This is a function of REAL measured metrics, never of LLM
    text (fixes findings 7,8)."""
    if not metric_a or not metric_b:
        return 0.0, "inconclusive"
    effect = fmean(metric_a) - fmean(metric_b)
    if effect >= min_effect:
        return effect, "supports"
    if effect <= -min_effect:
        return effect, "contradicts"
    return effect, "inconclusive"


def hypothesis_verdict(experiments: list[dict]) -> tuple[str, float]:
    """Deterministic hypothesis status from its experiments' verdicts (fixes 8).

    supported/contradicted from the balance; confirmed/refuted only with repeated,
    one-sided evidence. The LLM cannot move this — only real experiments can."""
    s = sum(1 for e in experiments if e.get("verdict") == "supports")
    c = sum(1 for e in experiments if e.get("verdict") == "contradicts")
    conf = (s + 1.0) / (s + c + 2.0)
    if s >= 2 and c == 0:
        return "confirmed", conf
    if c >= 2 and s == 0:
        return "refuted", conf
    if c > s:
        return "contradicted", conf
    if s > c:
        return "supported", conf
    return "proposed", 0.5
