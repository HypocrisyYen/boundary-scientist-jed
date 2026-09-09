"""score_jed — a JED Red-Team attack "score system".

Offline miner that applies the AI-score method (LLM code/program mutation +
UCT tree search + research-idea injection, evaluated in a scoring sandbox) to
the Kaggle "AI Agent Security — Multi-Step Tool Attacks" benchmark.

Where score_golf mines the shortest correct Python program, score_jed mines a
library of high-EV *attack programs* (replayable user-message chains) that
maximize risk-adjusted raw score per replay second.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
