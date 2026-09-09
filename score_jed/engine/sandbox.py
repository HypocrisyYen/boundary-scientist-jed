"""Sandbox: build a JED environment and replay a message chain in it.

score_jed analog of score_golf's subprocess sandbox. Instead of running Python
on stdin/stdout, it runs an attack (a user-message chain) against a tool-using
agent behind a guardrail and returns the resulting trace + the replay cost.

Snapshot prefix cache
---------------------
The competition env exposes ``snapshot()`` / ``restore(handle)``, and snapshots
round-trip and even restore into a fresh env. We exploit this like Go-Explore:
after each message we snapshot and cache it keyed by the exact rendered *prefix*.
To evaluate a chain we restore the longest cached prefix and replay only the
divergent suffix — skipping every tool-loop model call of the shared prefix
(a large saving on the GGUF targets, where each hop is seconds).

Crucially, EV stays honest: each prefix entry stores the cumulative wall time and
hop count measured when it was first run, so the reported ``replay_s`` is always
the **full-chain** cost the real evaluator would pay, never just the suffix.

Semantics note: reusing a prefix snapshot branches every suffix from ONE
realization of that prefix. For the deterministic target that is exact; for a
stochastic model it is the standard Go-Explore assumption (branch from a fixed
frontier state). Pass ``use_cache=False`` for an independent cold replay.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def add_sdk_path(sdk_root: str | Path | None = None) -> Path:
    root = Path(sdk_root) if sdk_root else _default_sdk_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root / "aicomp_sdk" / "fixtures"


def _default_sdk_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "ai-agent-security-multi-step-tool-attacks"
        if (cand / "aicomp_sdk").is_dir():
            return cand
    raise FileNotFoundError("Could not locate ai-agent-security-multi-step-tool-attacks/aicomp_sdk")


@dataclass
class TargetSpec:
    agent: str = "deterministic"     # deterministic | openai | gpt_oss | gemma | gemma_4
    guardrail: str = "optimal"       # optimal | allow | rules
    seed: int = 123
    max_tool_hops: int = 8
    # cost model (used when the agent replay is ~free, e.g. deterministic). For a
    # real GGUF target these are forced to 0 so EV uses the *measured* decode time.
    per_hop_s: float = 0.35
    per_message_s: float = 0.05
    agent_factory: Callable[[], Any] | None = None
    # GGUF wiring (used when agent in {gpt_oss, gemma, gemma_4} and no explicit factory)
    gguf_path: str | None = None
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    max_new_tokens: int = 256
    share_backend: bool = True
    # snapshot prefix cache
    use_prefix_cache: bool = True
    max_cache_entries: int = 4000
    # custom guardrail OBJECT (private-sim variants); when set, wins over `guardrail` kind
    guardrail_obj: Any | None = None


@dataclass
class _PrefixEntry:
    snapshot: Any
    cum_wall_s: float      # cumulative measured wall to reach this prefix (cold, first run)
    cum_hops: int          # cumulative tool events at this prefix
    n_messages: int


@dataclass
class ReplayResult:
    trace: dict[str, Any]
    wall_s: float
    replay_s: float = 0.0
    refused: bool = False
    fatal: str | None = None
    reused_prefix_len: int = 0
    cold: bool = True
    first_bad_message: int | None = None   # first message index that was denied/refused (repair anchor)


def _build_guardrail(kind: str):
    if kind == "allow":
        from aicomp_sdk.attacks.baselines.guardrail_allow import Guardrail
    elif kind == "rules":
        from aicomp_sdk.attacks.baselines.guardrail_rules import Guardrail
    else:
        from aicomp_sdk.guardrails.optimal import Guardrail
    return Guardrail()


class AttackSandbox:
    """Replays message chains against a fixed (agent, guardrail) target.

    Holds one persistent env (built lazily) plus a snapshot prefix cache. The
    searcher is single-threaded, so the persistent env + restore pattern is safe
    and matches the SDK's own Go-Explore baseline.
    """

    def __init__(self, spec: TargetSpec, *, sdk_root: str | Path | None = None) -> None:
        self.spec = spec
        self.fixtures_dir = add_sdk_path(sdk_root)
        from aicomp_sdk.core.env.sandbox import SandboxEnv

        self._SandboxEnv = SandboxEnv
        self._agent_factory = spec.agent_factory or self._default_agent_factory(spec)
        self._env = None
        self._root_snapshot = None
        self._gen_backend = None   # set by _ensure_env once the agent is built (P5) — the raw
                                    # backend object, not its stats list (see step()'s comment
                                    # on why a delta against _gen_total survives ring-buffer eviction)
        self._cache: dict[tuple[str, ...], _PrefixEntry] = {}
        # simple insertion-order eviction to bound snapshot memory
        self._cache_order: list[tuple[str, ...]] = []
        self.reuse_events = 0
        self.reuse_saved_messages = 0

    # -- agent / env plumbing -------------------------------------------------

    @staticmethod
    def _default_agent_factory(spec: "TargetSpec") -> Callable[[], Any]:
        from engine.gguf_agent import GGUF_KINDS, build_gguf_agent_factory

        if spec.agent in GGUF_KINDS:
            spec.per_hop_s = 0.0
            spec.per_message_s = 0.0
            return build_gguf_agent_factory(
                spec.agent,
                gguf_path=spec.gguf_path,
                n_ctx=spec.n_ctx,
                n_gpu_layers=spec.n_gpu_layers,
                max_new_tokens=spec.max_new_tokens,
                share_backend=spec.share_backend,
            )
        from aicomp_sdk.agents import build_agent

        return lambda: build_agent(spec.agent)

    def _ensure_env(self):
        if self._env is None:
            agent = self._agent_factory()
            # token-level diagnostics (P5): a GGUF agent factory stashes its backend as a plain
            # attribute (engine/gguf_agent.py); non-GGUF/deterministic agents simply lack it, so
            # this degrades to no gen-stats gracefully rather than erroring.
            self._gen_backend = getattr(agent, "_sj_backend", None)
            self._env = self._SandboxEnv(
                seed=self.spec.seed,
                fixtures_dir=self.fixtures_dir,
                agent=agent,
                guardrail=(self.spec.guardrail_obj
                           if self.spec.guardrail_obj is not None
                           else _build_guardrail(self.spec.guardrail)),
            )
            self._env.reset()
            self._root_snapshot = self._env.snapshot()
        return self._env

    # -- cache ----------------------------------------------------------------

    def _cache_put(self, key: tuple[str, ...], entry: _PrefixEntry) -> None:
        if key not in self._cache:
            self._cache_order.append(key)
            if len(self._cache_order) > max(0, self.spec.max_cache_entries):
                oldest = self._cache_order.pop(0)
                self._cache.pop(oldest, None)
        self._cache[key] = entry

    def _longest_cached_prefix(self, messages: tuple[str, ...]) -> tuple[int, _PrefixEntry | None]:
        for p in range(len(messages) - 1, 0, -1):
            entry = self._cache.get(messages[:p])
            if entry is not None:
                return p, entry
        return 0, None

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_order.clear()

    # -- replay ---------------------------------------------------------------

    def _cost(self, total_hops: int, n_messages: int, measured: float) -> float:
        modeled = self.spec.per_hop_s * total_hops + self.spec.per_message_s * n_messages
        return max(measured, modeled)

    def replay(
        self,
        messages: tuple[str, ...],
        *,
        max_tool_hops: int | None = None,
        use_cache: bool | None = None,
    ) -> ReplayResult:
        """Replay a chain; restore the longest cached prefix when possible.

        ``replay_s`` is always the reconstructed FULL-chain cost, so EV is honest
        regardless of how much of the prefix was served from cache.
        """
        hops = int(max_tool_hops or self.spec.max_tool_hops)
        cache_on = self.spec.use_prefix_cache if use_cache is None else bool(use_cache)
        env = self._ensure_env()

        # choose starting point
        start = 0
        cum_wall = 0.0
        cum_hops = 0
        if cache_on:
            start, entry = self._longest_cached_prefix(messages)
            if entry is not None:
                try:
                    env.restore(entry.snapshot)
                    cum_wall = entry.cum_wall_s
                    cum_hops = entry.cum_hops
                    self.reuse_events += 1
                    self.reuse_saved_messages += start
                except Exception:
                    start = 0
        if start == 0:
            try:
                if self._root_snapshot is not None:
                    env.restore(self._root_snapshot)
                else:
                    env.reset()
            except Exception:
                env.reset()

        refused_any = False
        first_bad: int | None = None
        try:
            for i in range(start, len(messages)):
                events_before = cum_hops
                t0 = time.perf_counter()
                res = env.interact(messages[i], max_tool_hops=hops)
                dt = time.perf_counter() - t0
                refused = bool(getattr(res, "agent_refused", False))
                refused_any = refused_any or refused
                cum_wall += dt
                trace_now = env.export_trace_dict()
                events = trace_now.get("tool_events", [])
                cum_hops = len(events)
                # a message "goes bad" if it refused or any of its new events was denied
                new_denied = any(
                    not e.get("ok") and str(e.get("error") or "").startswith(("denied", "confirm_required"))
                    for e in events[events_before:]
                )
                if first_bad is None and (refused or new_denied):
                    first_bad = i
                if cache_on:
                    self._cache_put(
                        messages[: i + 1],
                        _PrefixEntry(snapshot=env.snapshot(), cum_wall_s=cum_wall, cum_hops=cum_hops, n_messages=i + 1),
                    )
            trace = env.export_trace_dict()
        except Exception as exc:
            return ReplayResult(trace={}, wall_s=cum_wall, fatal=f"{type(exc).__name__}: {exc}", cold=(start == 0))

        total_hops = len(trace.get("tool_events", []))
        replay_s = self._cost(total_hops, len(messages), cum_wall)
        return ReplayResult(
            trace=trace, wall_s=cum_wall, replay_s=replay_s, refused=refused_any,
            reused_prefix_len=start, cold=(start == 0), first_bad_message=first_bad,
        )

    def replay_from_snapshot(
        self,
        snapshot: Any,
        suffix: tuple[str, ...],
        *,
        base_cum_wall: float = 0.0,
        base_hops: int = 0,
        base_msgs: int = 0,
        max_tool_hops: int | None = None,
    ) -> ReplayResult:
        """Restore a frontier snapshot and replay only ``suffix`` (repair-from-break)."""
        hops = int(max_tool_hops or self.spec.max_tool_hops)
        env = self._ensure_env()
        try:
            env.restore(snapshot)
        except Exception as exc:
            return ReplayResult(trace={}, wall_s=0.0, fatal=f"restore_failed: {exc}")
        cum_wall = base_cum_wall
        refused_any = False
        try:
            for m in suffix:
                t0 = time.perf_counter()
                res = env.interact(m, max_tool_hops=hops)
                cum_wall += time.perf_counter() - t0
                refused_any = refused_any or bool(getattr(res, "agent_refused", False))
            trace = env.export_trace_dict()
        except Exception as exc:
            return ReplayResult(trace={}, wall_s=cum_wall, fatal=f"{type(exc).__name__}: {exc}")
        total_hops = len(trace.get("tool_events", []))
        replay_s = self._cost(total_hops, base_msgs + len(suffix), cum_wall)
        return ReplayResult(trace=trace, wall_s=cum_wall, replay_s=replay_s, refused=refused_any, cold=False)

    def snapshot_after_prefix(self, messages: tuple[str, ...], length: int) -> _PrefixEntry | None:
        """Return the cached prefix entry for messages[:length], if present."""
        if length <= 0:
            return None
        return self._cache.get(tuple(messages[:length]))

    # -- interactive stepping (for the LLM explorer: try step -> observe -> adapt) --

    def begin_episode(self):
        """Start a fresh interactive episode from the clean root state."""
        env = self._ensure_env()
        try:
            if self._root_snapshot is not None:
                env.restore(self._root_snapshot)
            else:
                env.reset()
        except Exception:
            env.reset()
        return env

    def episode_snapshot(self):
        """Snapshot the live interactive episode state (for LLM-directed branching)."""
        try:
            return self._ensure_env().snapshot()
        except Exception:
            return None

    def current_trace(self) -> dict:
        """The live episode's full trace (tool events + messages) — for query_state,
        so the scientist can OBSERVE taint/secret/tool state instead of guessing."""
        try:
            return dict(self._ensure_env().export_trace_dict() or {})
        except Exception:
            return {}

    def episode_restore(self, snap) -> bool:
        """Restore the live episode to a prior snapshot so the LLM can branch/repair
        from where it broke on a CLEAN state (dropping the accumulated taint/history)."""
        if snap is None:
            return False
        try:
            self._ensure_env().restore(snap)
            return True
        except Exception:
            return False

    def step(self, message: str, *, max_tool_hops: int | None = None) -> dict:
        """Run ONE user message on the live episode; return what just happened.

        Returns the events THIS message produced (name/ok/source/args/error/output),
        the full trace, refusal flag, and wall time — the observation the LLM
        explorer reasons over before choosing its next message.
        """
        env = self._ensure_env()
        hops = int(max_tool_hops or self.spec.max_tool_hops)
        before = len(env.export_trace_dict().get("tool_events", []))
        # token-level diagnostics (P5): snapshot the MONOTONIC total (never shrinks, unlike the
        # capped stats list) so this step's own generations can be recovered via a delta even if
        # the ring buffer evicted older entries from the front in between (fixes a real bug: an
        # absolute-length slice start silently broke once total generations passed the cap).
        gens_total_before = getattr(self._gen_backend, "_gen_total", None)
        t0 = time.perf_counter()
        try:
            res = env.interact(message, max_tool_hops=hops)
            wall = time.perf_counter() - t0
            trace = env.export_trace_dict()
        except Exception as exc:
            return {"trace": {}, "new_events": [], "refused": False, "wall_s": time.perf_counter() - t0,
                    "gens": [], "fatal": f"{type(exc).__name__}: {exc}"}
        events = trace.get("tool_events", [])
        gens: list[dict] = []
        if gens_total_before is not None:
            gen_stats = getattr(self._gen_backend, "_gen_stats", None) or []
            produced = int(getattr(self._gen_backend, "_gen_total", gens_total_before)) - gens_total_before
            if produced > 0:
                gens = list(gen_stats[-produced:]) if produced <= len(gen_stats) else list(gen_stats)
        return {
            "trace": trace,
            "new_events": events[before:],
            "refused": bool(getattr(res, "agent_refused", False)),
            "wall_s": wall,
            "gens": gens,
            "fatal": None,
        }


class ProvidedEnvSandbox(AttackSandbox):
    """Replay core over an OPAQUE env supplied by the official harness.

    This is the submission-mode sandbox. It never constructs an env, never loads
    a model, never touches fixtures — it uses ONLY the env contract the evaluator
    hands us (reset / interact / export_trace_dict / snapshot / restore). All the
    prefix-cache + snapshot logic from AttackSandbox is inherited unchanged, so
    the same search core drives both research and 9000 s submission.
    """

    def __init__(
        self,
        env: Any,
        *,
        max_tool_hops: int = 8,
        seed: int = 123,
        use_prefix_cache: bool = True,
        max_cache_entries: int = 4000,
    ) -> None:
        # per_hop/per_message = 0 -> EV uses the REAL measured decode time.
        self.spec = TargetSpec(
            agent="provided", guardrail="provided", seed=seed, max_tool_hops=max_tool_hops,
            per_hop_s=0.0, per_message_s=0.0, use_prefix_cache=use_prefix_cache,
            max_cache_entries=max_cache_entries,
        )
        self._provided_env = env
        self._env = None
        self._root_snapshot = None
        # a ProvidedEnvSandbox wraps an OPAQUE harness env with no GGUF backend to introspect —
        # step()'s gen-stats lookup degrades to [] via getattr's default, same as any non-GGUF
        # target, but the attribute itself must exist (this class doesn't call super().__init__()).
        self._gen_backend = None
        self._cache: dict[tuple[str, ...], _PrefixEntry] = {}
        self._cache_order: list[tuple[str, ...]] = []
        self.reuse_events = 0
        self.reuse_saved_messages = 0

    def _ensure_env(self):
        if self._env is None:
            self._env = self._provided_env
            try:
                self._env.reset()
            except Exception:
                pass
            try:
                self._root_snapshot = self._env.snapshot()
            except Exception:
                self._root_snapshot = None
        return self._env
