"""TridentSandbox — a POLICY-mode adapter that lets the score_jed scientist drive the Trident
benchmark (CybORG CAGE4 / CyberWheel rollout servers) through the SAME RedTeamSandbox interface the
JED sandbox uses.

SKELETON / design scaffold: the control flow and the wire contract are real and match the downloaded
Trident code (`agentic/cage_reward.py`, `agentic/cage_default_run.py`, the GNN_CAGE4/HMARL/cyberwheel
servers), but it is not wired to a live server by default — set TRIDENT_SERVER_URL to enable the real
HTTP path; otherwise `step()` returns a clear "not configured" StepResult so imports and dry runs work
with no network / no venv.

MAPPING (JED  ->  Trident):
  user message                 ->  attack STRATEGY (NL/JSON) the scientist emits
  agent+guardrail turn (step)  ->  Coder(strategy) -> FiniteStateRedAgent code -> POST /rollout
  tool-call trace + predicates ->  rollout summary {red_reward, blue_reward, log} + TridentBlueDrop
  episode snapshot/restore     ->  surrogate: the accumulated strategy list (rollouts are stateless)
  cold replay (evaluator)      ->  one server rollout of the final compiled policy over eval seeds

The scientist's meta-strategy layer (`sample_strategies`) is a natural fit for the Trident Planner:
each meta = one strategy -> one rollout -> one reward. Interactive-only actions (send/forge/minimize)
collapse to "propose/refine the strategy"; `run_script`/`plan` map to the multi-step policy body.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .interface import RedTeamSandbox, StepResult


# Trident rollout server response keys (from the downloaded servers): status, reward/score,
# blue_reward, red_reward, traceback, error.
_DEFAULT_TIMEOUT_S = float(os.environ.get("TRIDENT_TIMEOUT_S", "60"))


class TridentSandbox(RedTeamSandbox):
    mode = "policy"

    def __init__(
        self,
        *,
        server_url: str | None = None,
        base_code_path: str | None = None,
        agent_cls_name: str = "FiniteStateRedAgent",
        seed: int = 123,
        eval_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
        coder=None,                      # optional engine-side Coder; None -> pass strategy as code directly
        max_repair: int = 3,
    ) -> None:
        # server_url: the /rollout endpoint (e.g. http://localhost:8931/rollout for GNN-CAGE4,
        # 8932 CyberWheel, 8933/8934 HMARL). None -> read TRIDENT_SERVER_URL; still None -> dry mode.
        self.server_url = server_url or os.environ.get("TRIDENT_SERVER_URL") or ""
        self.agent_cls_name = agent_cls_name
        self.seed = int(seed)
        self.eval_seeds = tuple(eval_seeds)
        self.coder = coder
        self.max_repair = int(max_repair)
        # base FiniteStateRedAgent the Coder splices a strategy into (Code-as-Policy).
        self._base_code = ""
        if base_code_path and Path(base_code_path).is_file():
            self._base_code = Path(base_code_path).read_text(encoding="utf-8")
        self._actions: list[str] = []    # surrogate episode state = the strategies applied so far
        self._last_trace: dict = {}

    # -- RedTeamSandbox surface ---------------------------------------------------------------------
    def begin_episode(self) -> None:
        self._actions = []
        self._last_trace = {}

    def step(self, action: str, **kw) -> StepResult:
        """action = an attack STRATEGY. Compile -> rollout -> reward."""
        self._actions.append(action)
        t0 = time.perf_counter()
        code = self._compile(action)
        if not self.server_url:
            # DRY MODE: no live server. Return a well-formed 'not configured' result so the scientist
            # loop, imports, and unit tests run without network / a Trident venv.
            res = StepResult(
                trace={"status": "dry", "red_reward": 0.0, "blue_reward": 0.0,
                       "log": "TRIDENT_SERVER_URL not set — dry run; compiled %d chars of policy" % len(code)},
                wall_s=time.perf_counter() - t0,
                feedback="[trident] dry mode: set TRIDENT_SERVER_URL to run a real rollout.",
                ok=True, raw={"code": code})
            self._last_trace = res.trace
            return res
        res = self._rollout(code, is_sanity=False, t0=t0)
        self._last_trace = res.trace
        return res

    def current_trace(self) -> dict:
        return dict(self._last_trace)

    def episode_snapshot(self) -> Any:
        # rollouts are stateless -> snapshot is the strategy prefix; restore replays it (see replay()).
        return tuple(self._actions)

    def episode_restore(self, snap: Any) -> bool:
        if isinstance(snap, (list, tuple)):
            self._actions = list(snap)
            return True
        return False

    def replay(self, actions: tuple[str, ...], *, use_cache: bool = False, **kw) -> StepResult:
        """Faithful cold path: compile the FINAL strategy and run ONE server rollout over eval_seeds
        (what cage_default_run.py does). Falls back to the base sequential replay in dry mode."""
        if not self.server_url:
            return super().replay(actions, use_cache=use_cache, **kw)
        t0 = time.perf_counter()
        code = self._compile(actions[-1] if actions else "")
        return self._rollout(code, is_sanity=False, t0=t0, seeds=self.eval_seeds)

    # -- internals ----------------------------------------------------------------------------------
    def _compile(self, strategy: str) -> str:
        """Strategy -> executable FiniteStateRedAgent source (Code-as-Policy). Uses the engine Coder if
        provided; else, if the strategy already looks like Python, pass it through; else wrap it as a
        docstring stub over the base code (dry-mode friendly)."""
        if self.coder is not None:
            try:
                return self.coder.develop(strategy, base_code=self._base_code)   # matches PythonDeveloper
            except Exception:
                pass
        if "class " in strategy and "def " in strategy:
            return strategy
        if self._base_code:
            return self._base_code
        return "# strategy (uncompiled):\n# " + strategy.replace("\n", "\n# ")

    def _rollout(self, code: str, *, is_sanity: bool, t0: float, seeds=None) -> StepResult:
        """POST {code, agent_cls_name, seed, is_sanity_check} to the rollout server, with the Trident
        self-repair loop on exec/HTTP error. Returns a StepResult carrying red/blue reward + log."""
        import requests
        seeds = seeds or (self.seed,)
        last_err = ""
        for attempt in range(self.max_repair):
            payload = {"code": code, "agent_cls_name": self.agent_cls_name,
                       "seed": int(seeds[0]), "seeds": list(seeds), "is_sanity_check": is_sanity}
            try:
                r = requests.post(self.server_url, json=payload, timeout=_DEFAULT_TIMEOUT_S + 5.0).json()
            except Exception as e:
                last_err = "HTTP error: %r" % e
                time.sleep(1.0 * (attempt + 1))
                continue
            if r.get("status") == "ok" or ("reward" in r or "red_reward" in r or "blue_reward" in r):
                red = r.get("red_reward", r.get("reward", r.get("score")))
                blue = r.get("blue_reward")
                trace = {"status": r.get("status", "ok"), "red_reward": red, "blue_reward": blue,
                         "log": r.get("log", r.get("summary", ""))}
                return StepResult(trace=trace, wall_s=time.perf_counter() - t0,
                                  feedback="[trident] rollout ok red=%s blue=%s" % (red, blue),
                                  ok=True, raw=r)
            last_err = r.get("traceback", r.get("error", "unknown execution error"))
            # self-repair: hand the traceback back to the Coder for the next attempt
            if self.coder is not None:
                try:
                    code = self.coder.repair(code, last_err)   # optional; skip if Coder lacks repair
                except Exception:
                    pass
        return StepResult(trace={"status": "error", "red_reward": 0.0, "blue_reward": 0.0,
                                 "log": last_err},
                          wall_s=time.perf_counter() - t0,
                          feedback="[trident] rollout FAILED after %d attempts: %s" % (self.max_repair, last_err[:200]),
                          ok=False, raw={"error": last_err})


def build_trident_sandbox(env: str = "gnn_cage4", **kw) -> TridentSandbox:
    """Convenience: pick the default port/base-code for a named Trident environment.
    env in {gnn_cage4:8931, cyberwheel:8932, hmarl3:8933, hmarl4:8934, marl1:8941}. Honors
    TRIDENT_SERVER_URL override. base_code defaults to the repo's FiniteStateRedAgent template."""
    ports = {"gnn_cage4": 8931, "cyberwheel": 8932, "hmarl3": 8933, "hmarl4": 8934, "marl1": 8941}
    port = ports.get(env, 8931)
    url = os.environ.get("TRIDENT_SERVER_URL") or ("http://localhost:%d/rollout" % port if os.environ.get("TRIDENT_LIVE") else "")
    return TridentSandbox(server_url=url, **kw)
