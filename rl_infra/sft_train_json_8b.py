"""QUICK corrected SFT for the 8B scientist brain — JSON-schema format + completion-only loss.

Fixes the v1 mistakes: (1) v1 trained on a `THINKING:/ACTION:` plain-text format that does NOT match
boundary_scientist's inference prompt (which demands a JSON object matching _SCHEMA) -> the v1 adapter
parse_errors at inference. (2) v1 used full-sequence loss (Qwen3 template lacks {% generation %}) so it
hallucinated the OBSERVED feedback turns.

Here: each golden decision step -> ONE single-turn example
  system = boundary_scientist._SYSTEM  (EXACT inference system prompt)
  user   = shared-knowledge + previous feedback + "Return ONLY JSON: <_SCHEMA>"  (inference-style ask)
  assistant = the JSON object {"hypothesis","action","message"}  (what the parser expects)
Manual label masking (labels=-100 on the prompt tokens) => loss ONLY on the JSON => no hallucination.
Curated to a FEW GOOD robust trajectories (DESTRUCTIVE/EXFIL/U2A fired; junk meta-steps dropped).
Fresh LoRA (NOT warm-started from the mis-formatted v1 adapter).
"""
from __future__ import annotations
import os, sys, json, glob, re

BASE   = os.environ.get("SFT_BASE", "unsloth/Qwen3-8B-bnb-4bit")
_HERE  = os.path.dirname(os.path.abspath(__file__))
DIRS   = os.environ.get("SFT_GOLDEN_DIRS",
                        ",".join([os.path.join(_HERE, "colab_backup", "m1"),
                                  os.path.join(_HERE, "colab_backup", "m2"),
                                  os.path.join(_HERE, "local_sft")]))
OUT    = os.environ.get("SFT_OUT", os.path.join(os.path.dirname(_HERE), "sft_model", "adapter_8b_json"))
R      = int(os.environ.get("SFT_R", "32"))
ALPHA  = int(os.environ.get("SFT_ALPHA", "64"))
MAXLEN = int(os.environ.get("SFT_MAXLEN", "2048"))
EPOCHS = float(os.environ.get("SFT_EPOCHS", "3"))
LR     = float(os.environ.get("SFT_LR", "2e-4"))
MAX_TRAJ = int(os.environ.get("SFT_MAX_TRAJ", "60"))   # "a few good" trajectories
MIN_STEPS = int(os.environ.get("SFT_MIN_STEPS", "3"))  # multi-step only: keep trajectories with >= this many action-steps

sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "score_jed"))
import engine.boundary_scientist as _B   # for the EXACT inference system prompt + schema
_SYSTEM = _B._SYSTEM
_SCHEMA = _B._SCHEMA


def _load_all_golden() -> list[dict]:
    """ALL unique golden trajectories (B: cover the full research machinery + every tool present, not
    just short single-shot fires). Multi-step trajectories are kept so the model learns the research
    SEQUENCE (recon->experiment->chain), not only one-shot sends."""
    seen: set = set(); rows: list[dict] = []
    for d in DIRS.split(","):
        for fp in sorted(glob.glob(os.path.join(d.strip(), "*.jsonl"))):
            try:
                for ln in open(fp, encoding="utf-8"):
                    ln = ln.strip()
                    if not ln:
                        continue
                    r = json.loads(ln)
                    if r.get("type") != "golden_trace" or not r.get("trace"):
                        continue
                    t = r["trace"]
                    key = (r.get("variant", ""), str(t[0].get("action", ""))[:50],
                           round(float(r.get("rawps", 0)), 1), len(t))
                    if key in seen:
                        continue
                    # MULTI-STEP ONLY (user 2026-09-01): keep trajectories with >= SFT_MIN_STEPS
                    # action-steps — the research traces (recon->experiment->chain), not one-shot fires.
                    if len([s for s in t if s.get("action")]) < MIN_STEPS:
                        continue
                    seen.add(key); rows.append(r)
            except Exception:
                pass
    # longest / highest-raw first
    rows.sort(key=lambda r: (len([s for s in r["trace"] if s.get("action")]),
                             float(r.get("rawps", 0))), reverse=True)
    return rows


_TOOLS = ("http.post", "fs.read", "fs.write", "fs.delete", "email.send", "web.search", "web.open", "shell.run")


def _action_json(step: dict) -> tuple[str, str] | None:
    """Map a golden step -> (category, JSON string). category is used for balancing/downsampling.
    Covers ALL scientist actions (send + research machinery), maps to boundary_scientist's schema."""
    a = str(step.get("action", "") or "").strip()
    hyp = str(step.get("thinking", "") or "")[:300]
    if not a:
        return None
    def j(d):
        d = {"hypothesis": hyp, **d}; return json.dumps(d, ensure_ascii=False)
    if not a.startswith("["):
        cat = next((t for t in _TOOLS if t in a.lower()), "send_other")
        return cat, j({"action": "send", "message": a})
    # research-machinery actions (best-effort from the rendered string; teaches the ACTION VOCABULARY)
    low = a.lower()
    if low.startswith("[run_script"):
        return "run_script", j({"action": "run_script", "message": a[a.find("]") + 1:].strip()[:600]})
    if low.startswith("[query_state"):
        return "query_state", j({"action": "query_state", "message": a[1:-1]})
    if low.startswith("[forge"):
        return "forge", j({"action": "forge", "message": a[a.find("]") + 1:].strip()[:400]})
    if low.startswith("[plan"):
        return "plan", j({"action": "plan", "message": a[a.find("]") + 1:].strip()[:400]})
    if low.startswith("[paired"):
        return "experiment", j({"action": "experiment", "message": a[a.find("]") + 1:].strip()[:400]})
    if low.startswith("[restore"):
        m = re.search(r"f\d+", a); return "restore", j({"action": "restore", "restore_to": (m.group(0) if m else "-1")})
    return None   # skip pure noise: [minimize refused], [agenda], [sweep], [next_meta], [?], [email]


HTTP_CAP = int(os.environ.get("SFT_HTTP_CAP", "80"))   # downsample http.post sends so they don't drown other tools


def _examples(rows: list[dict]) -> list[dict]:
    """Flatten to single-turn (system, user-context, assistant-JSON) examples, BALANCED across tools/
    actions (http.post is capped) so the model learns the full toolset + research machinery, not just
    the dominant http.post shortcut."""
    ex = []
    from collections import Counter
    catc = Counter()
    ask = "Design your NEXT move. Return ONLY JSON matching this schema:\n" + _SCHEMA
    for r in rows:
        mem = str(r.get("memory", "") or "")
        prev_fb = ""
        for i, step in enumerate(r["trace"]):
            res = _action_json(step)
            if res is None:
                prev_fb = str(step.get("feedback", "") or ""); continue
            cat, aj = res
            if cat == "http.post" and catc["http.post"] >= HTTP_CAP:
                prev_fb = str(step.get("feedback", "") or ""); continue   # balance: skip surplus http.post
            catc[cat] += 1
            ctx = (("SHARED KNOWLEDGE:\n" + mem + "\n\n") if (i == 0 and mem) else "")
            ctx += (("PREVIOUS RESULT: " + prev_fb + "\n\n") if prev_fb else "")
            ctx += ask
            ex.append({"system": _SYSTEM, "user": ctx, "assistant": aj})
            prev_fb = str(step.get("feedback", "") or "")
    max_ex = int(os.environ.get("SFT_MAX_EX", "0"))
    if max_ex and len(ex) > max_ex:
        ex = ex[:max_ex]   # keep the first N (from the longest trajectories first) to bound train time
    print("[jsft] example categories:", dict(catc), "| kept", len(ex), flush=True)
    return ex


def main():
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                              DataCollatorForSeq2Seq)
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from datasets import Dataset

    rows = _load_all_golden()
    ex = _examples(rows)
    print("[jsft] %d trajectories -> %d single-turn JSON examples (all tools + research machinery)"
          % (len(rows), len(ex)), flush=True)
    if len(ex) < 8:
        print("[jsft] too few examples (<8) — aborting.", flush=True); return

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _tok(e):
        prompt = tok.apply_chat_template([{"role": "system", "content": e["system"]},
                                          {"role": "user", "content": e["user"]}],
                                         tokenize=False, add_generation_prompt=True)
        full = prompt + e["assistant"] + tok.eos_token
        pid = tok(prompt, add_special_tokens=False).input_ids
        fid = tok(full, add_special_tokens=False, truncation=True, max_length=MAXLEN).input_ids
        labels = list(fid)
        for j in range(min(len(pid), len(labels))):
            labels[j] = -100                          # mask the prompt -> loss ONLY on the JSON response
        return {"input_ids": fid, "attention_mask": [1] * len(fid), "labels": labels}

    ds = Dataset.from_list(ex).map(_tok, remove_columns=["system", "user", "assistant"])

    model = AutoModelForCausalLM.from_pretrained(BASE, device_map={"": 0}, trust_remote_code=True, dtype="auto")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=R, lora_alpha=ALPHA, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUT, num_train_epochs=EPOCHS, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=LR, lr_scheduler_type="cosine",
        warmup_steps=5, logging_steps=5, save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    trainer.save_model(OUT); tok.save_pretrained(OUT)
    print("[jsft] DONE -> adapter saved to %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
