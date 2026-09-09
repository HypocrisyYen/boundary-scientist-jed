"""Benchmark-agnostic red-team interfaces + adapters.

Lets the score_jed scientist (strata_search + BoundaryScientist) drive ANY red-team benchmark through
one small contract, not just the JED aicomp_sdk. See GENERALIZATION.md for the design.
"""
from .interface import (
    RedTeamSandbox, StepResult, Objective, Score,
    JEDRawPerSecond, TridentBlueDrop,
)
from .trident_sandbox import TridentSandbox, build_trident_sandbox
from .cyborg_sandbox import CybORGSandbox, build_cyborg_sandbox
from .agentdojo_sandbox import AgentDojoSandbox, AgentDojoAttackSuccess, build_agentdojo_sandbox
from .online_loop import online_attack_defense

__all__ = [
    "RedTeamSandbox", "StepResult", "Objective", "Score",
    "JEDRawPerSecond", "TridentBlueDrop",
    "TridentSandbox", "build_trident_sandbox",
    "CybORGSandbox", "build_cyborg_sandbox",
    "AgentDojoSandbox", "AgentDojoAttackSuccess", "build_agentdojo_sandbox",
    "online_attack_defense",
]
