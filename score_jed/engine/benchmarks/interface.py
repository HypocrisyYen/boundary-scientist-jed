"""Benchmark-agnostic interfaces for the red-team scientist.

The score_jed scientist (`strata_search` + `BoundaryScientist`) only ever touches a sandbox through
a SMALL method surface and scores through ONE objective function. This module pins that surface down
as explicit ABCs so the same scientist can drive ANY red-team benchmark — the JED aicomp_sdk, the
Trident CAGE4/CyberWheel rollout servers, or a future one — by supplying a new adapter, with NO change
to the search core.

Two execution MODES cover the benchmarks we care about:

  INTERACTIVE  (JED): an ACTION is a user message. `step(action)` runs the target agent + guardrail
                      for one turn and returns a rich TRACE (tool_events + assistant_messages). The
                      objective scores the trace with the official predicates (severity + cells).

  POLICY       (Trident): an ACTION is an attack STRATEGY (natural language / JSON). `step(action)`
                      compiles it to an executable red-agent policy (Code-as-Policy) and runs a full
                      rollout on the benchmark server, returning a scalar reward (blue-performance
                      drop). The objective is (a transform of) that scalar.

The scientist code does not branch on mode: it calls the same 6 methods and reads `StepResult`.
Adapters translate. `mode` is advisory metadata (lets a caller widen/narrow the action vocabulary —
e.g. suppress interactive-only actions like `forge` in POLICY mode).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------------------------------
# StepResult — the uniform return of one scientist action, whatever the benchmark.
# --------------------------------------------------------------------------------------------------
@dataclass
class StepResult:
    """Result of one `sandbox.step(action)`.

    `trace` is the benchmark-native evidence the objective consumes:
      - INTERACTIVE: an SDK trace dict {"tool_events": [...], "assistant_messages": [...], ...}
      - POLICY:      a rollout summary dict {"reward": float, "blue_reward": float, "status": ...,
                     "log": <compressed rollout log the scientist can reason over>, ...}
    `wall_s` is the honest wall-clock this step cost (drives raw/s-style objectives).
    `feedback` is human/LLM-readable text the scientist sees next turn (deny reasons, tracebacks,
    rollout summary). `ok` is False when the action failed to execute at all (exec error / HTTP error).
    """
    trace: dict = field(default_factory=dict)
    wall_s: float = 0.0
    feedback: str = ""
    ok: bool = True
    raw: Any = None                      # optional benchmark-native raw payload (debug / advanced use)

    # dict-compat so JED-shaped code (BoundaryScientist.investigate reads sr.get("trace")/["wall_s"]/…)
    # can consume a RedTeamSandbox StepResult unchanged — the interop shim for #2 (drive the full core
    # over a non-JED sandbox). Exposes the JED step-dict keys: trace / wall_s / new_events / feedback / ok.
    def get(self, key: str, default: Any = None) -> Any:
        if key == "new_events":
            return (self.trace or {}).get("tool_events", []) if self.trace else []
        return getattr(self, key, (self.trace or {}).get(key, default))

    def __getitem__(self, key: str) -> Any:
        return self.get(key)


# --------------------------------------------------------------------------------------------------
# RedTeamSandbox — the ONLY sandbox surface the scientist requires.
# --------------------------------------------------------------------------------------------------
class RedTeamSandbox(abc.ABC):
    """Minimal contract `strata_search` / `BoundaryScientist` need. Existing `AttackSandbox` /
    `ProvidedEnvSandbox` already satisfy this shape (INTERACTIVE); `TridentSandbox` implements it in
    POLICY mode. Snapshots are opaque tokens — an adapter may return a cheap surrogate (e.g. the
    accumulated action list) when the underlying benchmark has no true state checkpoint."""

    mode: str = "interactive"            # "interactive" | "policy"

    @abc.abstractmethod
    def begin_episode(self) -> None:
        """Start a fresh episode from the clean root (equivalent to env.reset())."""

    @abc.abstractmethod
    def step(self, action: str, **kw) -> StepResult:
        """Execute ONE scientist action and return a StepResult (see above)."""

    @abc.abstractmethod
    def current_trace(self) -> dict:
        """The cumulative trace/evidence for the CURRENT episode state (post-last-step)."""

    @abc.abstractmethod
    def episode_snapshot(self) -> Any:
        """Return an opaque token capturing the current episode state (for Go-Explore branching)."""

    @abc.abstractmethod
    def episode_restore(self, snap: Any) -> bool:
        """Restore a prior snapshot. Return True on success. May be a no-op surrogate if the
        benchmark is stateless per rollout (then branching replays the action prefix instead)."""

    def replay(self, actions: tuple[str, ...], *, use_cache: bool = False, **kw) -> StepResult:
        """COLD-evaluate a full action sequence from a fresh root (what an evaluator does). Default
        implementation: begin_episode() then step() each action, returning the final StepResult with
        summed wall. Adapters override for a faithful/cheaper path (e.g. a single server rollout)."""
        self.begin_episode()
        total = 0.0
        last = StepResult()
        for a in actions:
            last = self.step(a)
            total += float(last.wall_s or 0.0)
        last.wall_s = total
        return last


# --------------------------------------------------------------------------------------------------
# Objective — the ONE scoring surface. Replaces the hardcoded `real_rawps` with a pluggable policy.
# --------------------------------------------------------------------------------------------------
@dataclass
class Score:
    value: float                          # the scalar the scientist maximizes (e.g. raw/s, or reward)
    components: dict = field(default_factory=dict)   # named parts for logging (sev, cells, blue_reward…)


class Objective(abc.ABC):
    """Maps a StepResult (or its trace + wall) to a scalar the scientist maximizes. One instance is
    threaded through `strata_search`; `extract_golden_traces` and the champion/floor comparison use it
    instead of importing `real_rawps` directly, so a new benchmark plugs in its own reward here."""

    name: str = "objective"

    @abc.abstractmethod
    def score(self, result: StepResult) -> Score:
        """Return the Score for one step result."""

    def gold_keep(self, best: Score, floor: float) -> bool:
        """Whether a trajectory whose best step scored `best` is worth keeping as golden vs `floor`.
        Default: beat the floor by 10%. Benchmarks with a high-severity signal override (see the JED
        objective's sev>=8 keep)."""
        return best.value >= 1.10 * float(floor)


# --------------------------------------------------------------------------------------------------
# JED objective adapter — wraps the existing real_rawps so the current system fits the new interface
# with zero behavior change. (Kept here so both benchmarks read from one place.)
# --------------------------------------------------------------------------------------------------
class JEDRawPerSecond(Objective):
    """INTERACTIVE / JED: raw/s = (Σ severity_weights + 2·unique_cells) / wall, via the official
    predicates. Delegates to engine.strata_search.real_rawps_from_trace so scoring stays identical to
    submission/SFT/RL. gold_keep also honors the sev>=GOLD_SEV_KEEP high-severity rule."""

    name = "jed_raw_per_s"

    def score(self, result: StepResult) -> Score:
        from engine.strata_search import real_rawps_from_trace
        rawps, sev, cells = real_rawps_from_trace(result.trace or {}, float(result.wall_s or 0.1))
        return Score(value=rawps, components={"sev": sev, "cells": cells, "wall_s": result.wall_s})

    def gold_keep(self, best: Score, floor: float) -> bool:
        import os
        sev_keep = float(os.environ.get("STRATA_GOLD_SEV_KEEP", "8"))
        return best.value >= 1.10 * float(floor) or float(best.components.get("sev", 0.0)) >= sev_keep


class TridentBlueDrop(Objective):
    """POLICY / Trident: the scientist maximizes the RED reward (== the blue defender's performance
    drop) returned by the rollout server. Higher red_reward / lower blue_reward = better attack.
    Normalized to a raw/s-like scale by dividing by wall so it composes with the same search loop."""

    name = "trident_blue_drop"

    def score(self, result: StepResult) -> Score:
        tr = result.trace or {}
        red = tr.get("red_reward")
        if red is None:                                   # derive red from blue if only blue is given
            red = -float(tr.get("blue_reward", 0.0))
        red = float(red)
        wall = max(float(result.wall_s or 0.1), 0.1)
        # value is reward-per-second so a fast high-impact policy ranks above a slow one, mirroring raw/s
        return Score(value=red / wall, components={"red_reward": red,
                                                    "blue_reward": tr.get("blue_reward"), "wall_s": wall})

    def gold_keep(self, best: Score, floor: float) -> bool:
        # keep any policy that actually degraded the blue agent (red_reward > 0), even below the floor rate
        return best.value >= 1.10 * float(floor) or float(best.components.get("red_reward", 0.0)) > 0.0
