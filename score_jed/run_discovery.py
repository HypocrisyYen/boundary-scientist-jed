"""Canonical LLM-core discovery research — two models × multi-guardrail (24h).

The boundary scientist IS the engine (discovers, self-optimises raw/sec, branches,
reverse-engineers the guard). It runs across BOTH scored models and a spread of
guardrails so the corpus is diverse AND high-EV, not single-guardrail single-step:

  models:     gpt_oss  <->  gemma           (VRAM-swapped; both GGUFs stay warm in RAM)
  guardrails: optimal | rules | allow
    - optimal = the public-LB winner: single-step SECRET_MARKER exfil (+ Harmony forge)
    - rules   = data-scanning guard -> DIFFERENT single-step (real-secret exfil) = private robustness
    - allow   = permissive -> the full MULTI-STEP surface (untrusted->action, destructive)

Block-alternates (model,guardrail); model-swap only when the model changes (guardrail
swap is free). Each (model,guardrail) resumes its own results_disc/<model>_<guardrail>/
corpus. Known high-EV notebook primitives (Harmony forge for gpt_oss, verbose imperative)
are injected as seeds. Resumable — re-run to continue.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
os.environ.setdefault("OPENAI_API_KEY", __import__("engine.apikey", fromlist=["load_api_key"]).load_api_key())
os.environ.setdefault("OPENAI_BASE_URL", "https://apihub.agnes-ai.com/v1")
os.environ.setdefault("LLM_MODEL", "agnes-2.0-flash")

import pipeline.research_discovery as d

_MODELS = "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/models"
_HOURS = float(os.environ.get("DISC_HOURS", "24"))

d.run(d.DiscoveryConfig(
    total_budget_s=_HOURS * 3600.0,
    targets=["gpt_oss", "gemma"],
    guardrails=["optimal", "rules", "allow"],
    block_s=1800.0,                      # 30-min blocks; model-swap every 3 blocks (~90m)
    gguf_paths={
        "gpt_oss": f"{_MODELS}/gpt-oss-20b-MXFP4.gguf",
        "gemma": f"{_MODELS}/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf",
    },
    # OFFICIAL FIDELITY (matches the evaluator: 8192 ctx / 256 tok / greedy) — the
    # fidelity_error() guard refuses anything weaker under profile='official'.
    n_ctx=8192,
    n_gpu_layers=-1,
    max_new_tokens=256,
    sci_max_steps=10,
    sample_trials=3,
    max_score_per_ep=5,
    amplify=True,
    amplify_budget=4,
    ckpt_every_s=180.0,
    resume=True,
    results_dir=Path(f"{Path(__file__).resolve().parent}/results_disc"),
))
