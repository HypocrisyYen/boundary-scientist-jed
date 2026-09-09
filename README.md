# Boundary Scientist — a red-team research agent for multi-step tool attacks

An LLM **"boundary scientist"** that reverse-engineers an unknown agent guardrail *by experiment*
— probing, forming hypotheses, running controlled A/B tests, and organizing multi-step tool attacks
in a persistent, resumable research tree — plus the trace-SFT / RL pipeline that distills its
exploration into an 8B model and transfers it from **red** (attacker) to **blue** (defender).

Built for the Kaggle **AI Agent Security — Multi-Step Tool Attacks (JED Red-Team)** competition
(OpenAI · Google · IEEE). Final: **11.82 private, 269 / 4,252.**

- **Working note:** [`WRITEUP_FINAL.md`](WRITEUP_FINAL.md)

---

## What's here

```
score_jed/     the attack system (submission + the boundary-scientist engine)
  attack_scientist.py     submission entry point (AttackAlgorithm.run)
  engine/                 boundary_scientist, strata_search, sandbox, guardrail
                          learner, hypothesis graph, memory layers, DSL SDK, benchmarks/
  pipeline/               research/discovery loop
  tools/                  utilities (incl. secret_scan.py)
  evidence/               red→blue transfer results + reproduction scripts
  README.md · ARCHITECTURE.md

rl_infra/      training: turn scientist traces into a model, then red → blue
  sci_common.py           shared scientist builder + DSL variant generators (archetypes)
  strata_real.py          SFT-trace collection (runs the scientist over target × variant)
  sft_train*.py           QLoRA trace-SFT (teach an 8B to be the scientist)
  build_blue_from_red.py  harvest danger patterns from red golden traces → blue SFT data
  blue_train.py           QLoRA blue-defender training
  train_grpo.py / train_sci*.py / selfplay*.py   RL + self-play
  *.jsonl                 curated SFT data (sft_consolidated_golden, blue_sft, sft_baseline)
```

Three entry points — submission, SFT-collection, RL — all call the **same** `strata_search` core and
share **one** `real_rawps = (Σ severity + 2·cells) / wall` formula, so an objective change updates
every consumer at once.

## The three ideas

1. **Boundary scientist** (`score_jed/engine/boundary_scientist.py`) — each node is one LLM decision
   from a 13-verb action vocabulary (`send / experiment(A/B) / run_script / query_state / restore /
   recall / …`), run against the live target+guardrail; `diagnose()` returns the guardrail's verbatim
   verdict + a per-predicate gap analysis. Every probe (even a zero-score denial) becomes an
   addressable, resumable frontier node. Seven bounded memory layers keep a long episode coherent
   without exploding the prompt.
2. **Trace-SFT** — a *golden trace* is the scientist's full `explore → think → act → feedback` decision
   log for one predicate-firing path. QLoRA-fine-tuning an 8B on these (completion-only loss) teaches
   it to run genuine multi-step research the base model cannot structure.
3. **Red → blue transfer (with its disconfirming control)** — the red scientist's golden traces are a
   *labeled catalogue of dangerous requests, discovered by search*. Distilled into defender SFT, they
   teach the same 8B to refuse **16/40 held-out guardrail-allowed attacks** it never trained on
   (e.g. `id_rsa`, `rm -rf`, `dd`, `nc` reverse shell — 5/5) vs the base model's **2/40**, benign
   utility intact (35/40). The transfer is real — but a size-matched generic-safety control refuses
   *more* (32/40) while over-refusing benign (22/40), so red's edge is **utility preservation, not
   stronger refusal**. Honest write-up in WRITEUP §7. Evidence:
   `score_jed/evidence/red_to_blue_heldout{,_generic}.{json,log}`.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate        # Python 3.11+
pip install -r requirements.txt                      # or: openai, transformers, peft, datasets, torch
```

**API keys are never stored in this repo.** The strategist LLM key is read from the environment
(or an untracked `.secrets/agnes_key.txt`), in this precedence:

```bash
export OPENAI_API_KEY=...        # or OPENROUTER_API_KEY / MOONSHOT_API_KEY
export OPENAI_BASE_URL=...        # optional; auto-set for OpenRouter/Moonshot
export LLM_MODEL=...              # strategist model id
```

See `score_jed/engine/apikey.py` / `llm_client.py` for the full resolution order.

## Run

```bash
# Submission attack (drives env.interact, returns replayable candidates)
python score_jed/attack_scientist.py

# Collect SFT golden traces (scientist over target × DSL variants)
python rl_infra/strata_real.py

# Trace-SFT: teach an 8B to be the scientist
python rl_infra/sft_train_json_8b.py

# Red → blue: harvest danger patterns, train + evaluate the defender
python rl_infra/build_blue_from_red.py
python rl_infra/blue_train.py
ADAPTER=sft_model/adapter_8b_blue2 python score_jed/evidence/red_to_blue_full_test.py
```

## Docs

| File | What |
|---|---|
| [`WRITEUP_FINAL.md`](WRITEUP_FINAL.md) | The competition working note (full narrative + results) |
| [`WRITEUP_KAGGLE.md`](WRITEUP_KAGGLE.md) | Same note, with rendered PNG diagrams |
| [`score_jed/ARCHITECTURE.md`](score_jed/ARCHITECTURE.md) | System architecture |
| [`score_jed/README.md`](score_jed/README.md) | The discovery/search stack in detail |

## Not in the repo (kept locally, git-ignored)

Trained weights and large regenerable data are excluded via `.gitignore`: `rl_infra/ckpt*/`,
`sft_model/`, `*.safetensors/.pt/.bin/.gz`, `rl_infra/sft_data.jsonl`, all logs and `__pycache__`,
and any secret files (`llm_config.json`, `.env`, `.secrets/`). The curated `*.jsonl` SFT data is
kept so the pipeline is reproducible.
