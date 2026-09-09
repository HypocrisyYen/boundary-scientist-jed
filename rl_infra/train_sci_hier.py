"""HIERARCHICAL GRPO trainer (StraTA-style) on the UNIFIED core (engine.hierarchical.hierarchical_rollout).

StraTA (arxiv 2605.06642) mapped to ours (see rl_infra/RL_UNIFICATION_PLAN.md):
  - per variant (task): sample N metas (strategies); run M investigations (rollouts) per meta.
  - reward r_ij = real_rawps of rollout j under meta i (our global objective).
  - STRATEGY-layer advantage: R_i = delta*mean + (1-delta)*max of {r_ij} (delta-aggregation);
    A_meta_i = zscore({R_i}) over the N metas -> trains the META-GENERATION tokens (sample_strategies).
  - ACTION-layer advantage: A_act_ij = zscore({r_ij}) within meta i's M rollouts -> trains the
    DECISION tokens; minus kappa for steps the self-judge flagged as off-strategy/wasted (Eq16-17).
  - loss = sum_i PG(meta_i, A_meta_i) + sum_ij sum_act PG(a, A_act_ij) + BETA*KL(pi||pi_ref).
  - policy starts FROM the SFT adapter (curriculum: warmup-SFT -> RL); ref = the SFT policy (adapter
    disabled == the 4bit base is NOT the ref; we snapshot ref logp with disable_adapter, i.e. base —
    matching train_sci_real's KL-to-base. To KL-to-SFT instead, load a frozen copy; base-KL is fine
    for a light regularizer here).

The scientist BRAIN is the policy (in-process HF, on the A100); the target AGENT (gpt_oss/gemma) is a
GPU llama.cpp gguf (also on the A100) — per user: scientist and agent both on the A100 for RL.

Env: HR_STEPS(200) HR_N(8 metas) HR_M(4 rollouts) HR_TREE(24) HR_MAXSTEPS(6) HR_LR(1e-5) HR_BETA(0.02)
     HR_DELTA(0.5) HR_KAPPA(0.1) HR_SAVE(5) HR_BASE(unsloth/Qwen3-14B-bnb-4bit) HR_SFT_ADAPTER(path)
     HR_CKPT(/content/work/rl_infra/ckpt_rl_hier) HR_CURRIC(1: base variants early, archetype later)
     RL_GPT_GGUF, RL_TGT_NGL(-1), STRATA_TARGETS(gpt_oss)
"""
from __future__ import annotations
import os, sys, time, glob, re
import torch

for _p in ("/content/work/rl_infra", "/content/work/score_jed",
           "/content/work/ai-agent-security-multi-step-tool-attacks",
           "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/rl_infra",
           "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/score_jed",
           "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/ai-agent-security-multi-step-tool-attacks"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import engine.llm_client as llm_client                       # noqa: E402
from engine.boundary_scientist import BoundaryScientist      # noqa: E402
from engine.hierarchical import hierarchical_rollout         # noqa: E402
from sci_common import build_full_scientist, base_variants, archetype_variants  # noqa: E402

STEPS   = int(os.environ.get("HR_STEPS", "200"))
N_META  = int(os.environ.get("HR_N", "8"))
M_ROLL  = int(os.environ.get("HR_M", "4"))
TREE    = int(os.environ.get("HR_TREE", "24"))
MAXSTEPS = int(os.environ.get("HR_MAXSTEPS", "6"))
LR      = float(os.environ.get("HR_LR", "1e-5"))
BETA    = float(os.environ.get("HR_BETA", "0.02"))
DELTA   = float(os.environ.get("HR_DELTA", "0.5"))
KAPPA   = float(os.environ.get("HR_KAPPA", "0.1"))
SAVE    = int(os.environ.get("HR_SAVE", "5"))
MAXNEW  = int(os.environ.get("HR_MAXNEW", "512"))
BASE    = os.environ.get("HR_BASE", "unsloth/Qwen3-14B-bnb-4bit")
SFT_ADAPTER = os.environ.get("HR_SFT_ADAPTER", "")
CKPT    = os.environ.get("HR_CKPT", "/content/work/rl_infra/ckpt_rl_hier")
CURRIC  = os.environ.get("HR_CURRIC", "1") == "1"
TARGETS = [t.strip() for t in os.environ.get("STRATA_TARGETS", "gpt_oss").split(",") if t.strip()]
NGL     = int(os.environ.get("RL_TGT_NGL", "-1"))

_POLICY = {"tok": None, "model": None, "dev": None, "capture": None, "ncall": 0}


def _render(messages):
    return "\n\n".join(m.get("content", "") for m in messages)


def _cap_reset():
    _POLICY["capture"] = []


def _cap_read():
    return list(_POLICY["capture"] or [])


def policy_chat(messages, *, n=1, model=None, temperature=None):
    """The scientist brain = the in-process HF policy. Captures (prompt_ids, comp_ids) for every
    generation so the SAME tokens feed the GRPO loss. Applies the model's chat template."""
    tok, mdl, dev = _POLICY["tok"], _POLICY["model"], _POLICY["dev"]
    msgs = list(messages)
    try:
        if getattr(tok, "chat_template", None):
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = _render(msgs)
    except Exception:
        prompt = _render(msgs)
    ids = tok(prompt, return_tensors="pt").input_ids
    if ids.shape[1] > 4096:
        ids = ids[:, -4096:]
    ids = ids.to(dev)
    with torch.no_grad():
        out = mdl.generate(ids, max_new_tokens=MAXNEW, do_sample=True,
                           temperature=(1.0 if temperature is None else float(temperature)),
                           top_p=0.95, pad_token_id=tok.eos_token_id)
    comp = out[0, ids.shape[1]:]
    text = tok.decode(comp, skip_special_tokens=True)
    _POLICY["ncall"] += 1
    if _POLICY["capture"] is not None:
        _POLICY["capture"].append({"prompt_ids": ids[0].detach().cpu(),
                                   "comp_ids": comp.detach().cpu()})
    return [text]


def _lsoftmax(logits, index):
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, index.unsqueeze(-1)).squeeze(-1)


def _zscore(vals):
    t = torch.tensor([float(v) for v in vals], dtype=torch.float32)
    if t.numel() > 1 and float(t.std()) > 1e-6:
        z = (t - t.mean()) / (t.std() + 1e-6)
    else:
        z = torch.zeros_like(t)
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).tolist()


def _variants_for_step(step):
    """Curriculum: early steps -> base (warmup) variants, later -> archetype (hard)."""
    if CURRIC and step < max(1, STEPS // 4):
        return base_variants(2, seed=step)
    return archetype_variants(2, seed=step)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

    print("[HR] loading policy 4bit base %s (SFT adapter=%r)" % (BASE, SFT_ADAPTER or None), flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0},
                                                 trust_remote_code=True, dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # resume > SFT-adapter warm-start > fresh LoRA
    ckpts = sorted(glob.glob(CKPT + "/checkpoint-*"), key=lambda d: int(re.findall(r"\d+", d.split("-")[-1])[0] or 0))
    start_step = 0
    if ckpts:
        model = PeftModel.from_pretrained(model, ckpts[-1], is_trainable=True)
        start_step = int(re.findall(r"\d+", ckpts[-1].split("-")[-1])[0]) + 1
        print("[HR] RESUMED from", ckpts[-1], "-> step", start_step, flush=True)
    elif SFT_ADAPTER and os.path.isdir(SFT_ADAPTER):
        model = PeftModel.from_pretrained(model, SFT_ADAPTER, is_trainable=True)
        print("[HR] warm-started from SFT adapter", SFT_ADAPTER, flush=True)
    else:
        lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    dev = model.device
    _POLICY.update(tok=tok, model=model, dev=dev)
    llm_client.chat = policy_chat                              # the scientist brain IS the policy
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    print("[HR] start: steps=%d N=%d M=%d delta=%.2f kappa=%.2f beta=%.3f targets=%s"
          % (STEPS, N_META, M_ROLL, DELTA, KAPPA, BETA, TARGETS), flush=True)
    for step in range(start_step, STEPS):
        t0 = time.time()
        batch = []          # each: {prompt_ids, comp_ids, adv}
        logrows = []
        agent_name = TARGETS[step % len(TARGETS)]              # alternate gpt_oss<->gemma
        for params in _variants_for_step(step):
            _POLICY["ncall"] = 0
            model.eval()
            with torch.no_grad():
                sci, sb = build_full_scientist(params, max_steps=MAXSTEPS, agent=agent_name, n_gpu_layers=NGL)
                cfg = {"n_strat": N_META, "m_rollouts": M_ROLL, "tree_depth": TREE, "agent": agent_name,
                       "kappa": KAPPA, "cap_reset": _cap_reset, "cap_read": _cap_read}
                try:
                    roll = hierarchical_rollout(sci, sb, cfg)
                except Exception as exc:
                    print("[HR] rollout err:", repr(exc)[:140], flush=True)
                    roll = {"metas": [], "floor": 0.0}
                try:
                    sb._sb._env = None
                except Exception:
                    pass
                del sci, sb
                import gc; gc.collect(); torch.cuda.empty_cache()

            metas = roll.get("metas", [])
            # ---- STRATEGY layer: R_i = delta*mean + (1-delta)*max over meta i's rollout rewards ----
            R = []
            for m in metas:
                rs = [float(x.get("reward", 0.0)) for x in m.get("rollouts", [])] or [0.0]
                R.append(DELTA * (sum(rs) / len(rs)) + (1.0 - DELTA) * max(rs))
            A_meta = _zscore(R) if R else []
            for m, a_meta in zip(metas, A_meta):
                for it in (m.get("strategy_capture") or []):
                    if it.get("comp_ids") is not None and it["comp_ids"].numel() > 0:
                        batch.append({"prompt_ids": it["prompt_ids"], "comp_ids": it["comp_ids"], "adv": a_meta})
                # ---- ACTION layer: zscore within this meta's rollouts; -kappa for judged-bad steps ----
                rs = [float(x.get("reward", 0.0)) for x in m.get("rollouts", [])]
                A_act = _zscore(rs) if rs else []
                for j, ro in enumerate(m.get("rollouts", [])):
                    a_act = A_act[j] if j < len(A_act) else 0.0
                    bad = set(ro.get("judged_bad") or [])
                    for k, d in enumerate(ro.get("decisions") or []):
                        if d.get("comp_ids") is None or d["comp_ids"].numel() < 1:
                            continue
                        adv = a_act - (KAPPA if k in bad else 0.0)
                        batch.append({"prompt_ids": d["prompt_ids"], "comp_ids": d["comp_ids"], "adv": adv})
            best = max(R) if R else 0.0
            meanR = (sum(R) / len(R)) if R else 0.0
            logrows.append(("%s:%s" % (agent_name[:4], params.name), best, meanR, len(metas)))

        # free target VRAM before backward (both scientist+target were on GPU)
        try:
            from engine.gguf_agent import clear_shared_backends
            clear_shared_backends()
        except Exception:
            pass
        torch.cuda.empty_cache()

        # ---- GRPO backward (same loss as train_sci_real: PG + BETA*KL-to-base) ----
        model.train(); model.config.use_cache = False
        model.gradient_checkpointing_enable()
        opt.zero_grad(); used = 0; kl_sum = 0.0
        for it in batch:
            a = it["adv"]
            if abs(a) < 1e-6 or it["comp_ids"].numel() < 1:
                continue
            pids = it["prompt_ids"].to(dev); cids = it["comp_ids"].to(dev)
            ids = torch.cat([pids, cids]).unsqueeze(0)
            if ids.shape[1] > 4600:
                ids = ids[:, -4600:]
            n_comp = min(cids.numel(), ids.shape[1] - 1)
            logits = model(ids).logits[0][:-1]; tgt = ids[0][1:]
            logp = _lsoftmax(logits, tgt)[-n_comp:]
            with torch.no_grad():
                with model.disable_adapter():
                    ref_logp = _lsoftmax(model(ids).logits[0][:-1], tgt)[-n_comp:]
            pg = -(a * logp.sum()) / max(n_comp, 1)
            kl = ((torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0).sum()) / max(n_comp, 1)
            (pg + BETA * kl).backward(); used += 1; kl_sum += float(kl.item())
        gnorm = 0.0
        if used:
            gnorm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
            opt.step()
        dt = time.time() - t0
        for (name, best, meanR, nm) in logrows:
            print("[HR] step %d %-14s bestR=%.2f meanR=%.2f metas=%d batch=%d used=%d kl=%.3f gnorm=%.2f (%.0fs)"
                  % (step, name, best, meanR, nm, len(batch), used, (kl_sum / used if used else 0), gnorm, dt), flush=True)
        if step % SAVE == 0:
            d = "%s/checkpoint-%d" % (CKPT, step); model.save_pretrained(d); tok.save_pretrained(d)
            print("[HR] saved", d, flush=True)
    model.save_pretrained(CKPT + "/final"); print("[HR] DONE", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.stdout.flush(); raise
