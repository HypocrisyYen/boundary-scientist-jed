"""CybORGSandbox — ONLINE-mode adapter (mode 'b').

Unlike TridentSandbox (POLICY mode: one action = a whole compiled rollout), this maps our scientist's
generation flow to a SINGLE EPISODE played turn-by-turn — the faithful "online attack/defense" reading:

    begin_episode()  = env.reset()                       -> the network at t0
    step(action)     = env.step(red_action) for ONE turn -> observation + per-step reward
    current_trace()  = the accumulated observation/killchain state for the CURRENT step
    episode_snapshot / episode_restore = CybORG deep-copy state (Go-Explore over game states, if the
                       env exposes save/restore; else a surrogate = the action prefix, replayed)
    episode reward   = the Objective's cumulative signal over the episode

So ONE full `investigate()` generation flow = ONE episode of live red-vs-blue play, with our memory +
tree carried across the turns. This is what lets a long episode stay affordable: the scientist reasons
with distilled memory instead of re-deriving each turn (see the memory critique in BENCHMARKS_COMPARISON /
the notes below).

SKELETON: the interface + action mapping are real (CAGE4's 9-action red space); the CybORG env is only
imported lazily. Without CybORG installed (no venv) it runs in DRY mode so imports/dry-runs work.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

from .interface import RedTeamSandbox, StepResult


def _sum_reward(v) -> float:
    """Sum a CybORG reward breakdown ({'BlueRewardMachine': -3, 'action_cost': 0}) to a scalar."""
    if isinstance(v, dict):
        return float(sum(x for x in v.values() if isinstance(x, (int, float))))
    return float(v) if isinstance(v, (int, float)) else 0.0


def cyborg_state_sig(trace: dict) -> str:
    """Compact CybORG state signature for StateValueMemory (known-hosts / red-sessions bucket)."""
    s = str((trace or {}).get("obs", ""))
    return "cyborg:" + s if s else "cyborg:init"


# CAGE4 red action vocabulary (see FiniteStateRedAgent.action_list) — the scientist chooses among these
# per turn, targeting a host. In DRY mode we just echo the intent; live mode maps to CybORG Action objs.
CAGE4_RED_ACTIONS = (
    "DiscoverRemoteSystems", "AggressiveServiceDiscovery", "StealthServiceDiscovery",
    "DiscoverDeception", "ExploitRemoteService", "PrivilegeEscalate",
    "Impact", "DegradeServices", "Withdraw",
)


class CybORGSandbox(RedTeamSandbox):
    mode = "online"

    def __init__(self, *, scenario: str = "Scenario4", blue_agent: str = "gnn",
                 max_steps: int = 100, seed: int = 123) -> None:
        self.scenario = scenario
        self.blue_agent = blue_agent          # which fixed DRL blue we red-team against
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self._env = None                      # lazily-built CybORG env (None => dry mode)
        self._obs: dict = {}
        self._cum_reward = 0.0
        self._t = 0
        self._actions: list[str] = []

    # -- lazy env build (dry-safe) ------------------------------------------------------------------
    def _ensure_env(self):
        if self._env is not None:
            return self._env
        # DRY unless explicitly enabled (keeps imports/dry-runs free of the heavy CybORG deps).
        if os.environ.get("CYBORG_LIVE") != "1":
            return None
        try:
            # Put the downloaded CybORG source on the path (CYBORG_ROOT), then build a REAL CAGE4
            # EnterpriseScenarioGenerator: a defending blue (cc4BlueRandomAgent by default — takes real
            # defensive actions), green users, and red as Sleep (WE inject red actions via step()).
            root = os.environ.get("CYBORG_ROOT",
                                  os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                               "external_benchmarks", "trident", "GNN_CAGE4"))
            root = os.path.abspath(root)
            if root not in sys.path:
                sys.path.insert(0, root)
            from CybORG import CybORG
            from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
            import CybORG.Agents as _A
            blue_cls = getattr(_A, {"random": "cc4BlueRandomAgent", "monitor": "MonitorAgent",
                                    "sleep": "SleepAgent"}.get(self.blue_agent, "cc4BlueRandomAgent"),
                               _A.cc4BlueRandomAgent)
            sg = EnterpriseScenarioGenerator(blue_agent_class=blue_cls,
                                             green_agent_class=_A.EnterpriseGreenAgent,
                                             red_agent_class=_A.SleepAgent, steps=self.max_steps)
            self._env = CybORG(sg, "sim", seed=self.seed)
            self._env.reset()
            self._red = next((a for a in self._env.active_agents if "red" in a), "red_agent_0")
        except Exception as e:
            self._env = None
            self._build_err = repr(e)[:300]
        return self._env

    # -- RedTeamSandbox surface --------------------------------------------------------------------
    def begin_episode(self) -> None:
        self._actions = []
        self._cum_reward = 0.0
        self._t = 0
        env = self._ensure_env()
        if env is not None:
            env.reset()
            self._red = next((a for a in env.active_agents if "red" in a), "red_agent_0")
            self._obs = env.get_observation(self._red)
        else:
            self._obs = {"status": "dry", "network": "unbuilt", "known_hosts": []}

    def step(self, action: str, **kw) -> StepResult:
        """action = one red move, e.g. 'ExploitRemoteService' (a CAGE4 red action type; params auto-filled
        from the live action space — valid targets grow as red discovers the network)."""
        self._actions.append(action)
        self._t += 1
        t0 = time.perf_counter()
        env = self._ensure_env()
        if env is None:
            verb = action.split()[0] if action else "?"
            ok = verb in CAGE4_RED_ACTIONS
            r = StepResult(
                trace={"status": "dry", "t": self._t, "obs": {}, "reward": 0.0, "cum_reward": self._cum_reward,
                       "log": "dry env — parsed red action %r (valid=%s); build err=%s" % (
                           verb, ok, getattr(self, "_build_err", "n/a"))},
                wall_s=time.perf_counter() - t0,
                feedback="[cyborg] dry mode: set CYBORG_LIVE=1 (+ CybORG deps) to play a real episode.", ok=ok)
            self._obs = r.trace
            return r
        # build a valid CybORG red action from the scientist's chosen type + the live action space
        cy_action, built = self._to_cyborg_action(action, env)
        results = env.step(agent=self._red, action=cy_action)      # ONE live turn vs the defending blue
        rewards = env.get_rewards() if hasattr(env, "get_rewards") else {}
        blue = _sum_reward(rewards.get("Blue")) if isinstance(rewards, dict) else 0.0
        red_own = _sum_reward(rewards.get("Red")) if isinstance(rewards, dict) else 0.0
        # attack impact = red's own reward PLUS the blue defender's penalty (blue goes negative as red
        # degrades the network) — this is the "blue performance drop" the objective maximizes.
        red_reward = red_own + max(0.0, -blue)
        self._cum_reward += red_reward
        self._obs = env.get_observation(self._red)
        done = self._t >= self.max_steps
        summ = self._summarize_obs(self._obs)
        return StepResult(
            trace={"t": self._t, "action_built": built, "reward": red_reward, "blue_reward": blue,
                   "red_reward": red_reward, "cum_reward": self._cum_reward, "done": done, "obs": summ,
                   "log": "red=%s -> %s | red_reward=%.2f blue=%.2f" % (built, summ, red_reward, blue)},
            wall_s=time.perf_counter() - t0,
            feedback="[cyborg] t=%d %s red_reward=%.2f (blue=%.2f) cum=%.2f | %s" % (
                self._t, built, red_reward, blue, self._cum_reward, summ),
            ok=True, raw=getattr(results, "observation", None))

    def current_trace(self) -> dict:
        return {"t": self._t, "cum_reward": self._cum_reward, "obs": self._obs,
                "actions": list(self._actions)}

    def episode_snapshot(self) -> Any:
        env = self._ensure_env()
        if env is not None and hasattr(env, "get_state"):
            try:
                return ("state", env.get_state())          # true Go-Explore checkpoint if supported
            except Exception:
                pass
        return ("prefix", tuple(self._actions))            # surrogate: replay the action prefix

    def episode_restore(self, snap: Any) -> bool:
        env = self._ensure_env()
        if isinstance(snap, tuple) and snap and snap[0] == "state" and env is not None and hasattr(env, "set_state"):
            try:
                env.set_state(snap[1]); return True
            except Exception:
                return False
        if isinstance(snap, tuple) and snap and snap[0] == "prefix":
            self.begin_episode()
            for a in snap[1]:
                self.step(a)
            return True
        return False

    # -- helpers ------------------------------------------------------------------------------------
    def _to_cyborg_action(self, action: str, env):
        """Map the scientist's chosen action TYPE -> a VALID parametrized CybORG red action, filling
        each constructor param from the live action space (valid subnet/ip/host grow as red discovers
        the net). Returns (action_obj, human_label). Falls back to Sleep if the type isn't achievable
        yet (e.g. Exploit with no discovered host) — a real, in-distribution 'wasted turn'."""
        import inspect
        verb = (action.split()[0] if action else "Sleep").strip().rstrip(".,")
        sp = env.get_action_space(self._red)
        classes = {c.__name__: c for c in sp.get("action", [])}
        cls = classes.get(verb) or classes.get("Sleep")
        if cls is None:
            return next(iter(sp.get("action", [None])), None), "Sleep"
        # collect valid values per param name from the action space
        def _valid(name):
            d = sp.get(name, {})
            return [k for k, ok in d.items() if ok] if isinstance(d, dict) else []
        kwargs = {}
        try:
            params = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
            for p in params:
                if p == "session":
                    kwargs[p] = (_valid("session") or [0])[0]
                elif p == "agent":
                    kwargs[p] = self._red
                elif p in ("subnet", "ip_address", "hostname", "username", "password", "process",
                           "port", "target_session"):
                    vals = _valid(p)
                    if not vals:
                        return classes.get("Sleep")(), "Sleep(no %s yet)" % p   # can't target -> real no-op turn
                    kwargs[p] = vals[0]
            obj = cls(**kwargs)
            return obj, "%s(%s)" % (verb, ",".join(str(v)[:18] for k, v in kwargs.items() if k not in ("agent", "session")))
        except Exception as e:
            sl = classes.get("Sleep")
            return (sl() if sl else None), "Sleep(build_err:%s)" % repr(e)[:40]

    def _summarize_obs(self, obs) -> str:
        # compact, LLM-facing killchain state the scientist reasons over (feeds StateValueMemory sig)
        try:
            if not isinstance(obs, dict):
                return ""
            hosts = [k for k in obs.keys() if k not in ("success",)]
            sess = sum(1 for h in hosts if isinstance(obs.get(h), dict) and obs[h].get("Sessions"))
            return "known_hosts=%d red_sessions=%d success=%s" % (len(hosts), sess, obs.get("success"))
        except Exception:
            return ""


def build_cyborg_sandbox(blue_agent: str = "gnn", **kw) -> CybORGSandbox:
    """blue_agent in {gnn, hmarl3, hmarl4, marl1}. Honors CYBORG_LIVE=1 to attempt a real env."""
    return CybORGSandbox(blue_agent=blue_agent, **kw)
