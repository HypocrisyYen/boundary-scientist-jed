"""(A) RL that trains the REAL scientist brain — aligned to R1/R5/R6/R7 (rewrite 2026-08-26).

The trainable policy IS the strategist (monkeypatch llm_client.chat). Per step we run G independent
FULL-memory scientist investigations (sci_common.build_full_scientist: notebook+hypo+failure+replay+
guardrail_learner+agenda, fresh per variant = R1/R2) on the DSL-ified REAL SDK with real holes (R6),
over DEEP investigations (SR_MAXSTEPS, R7). Reward = the scientist's OWN official EV/s champion
(EVChampSandbox: eval_predicates severity / real replay wall = R5). GRPO group = the G investigations
on one variant; advantage = z-score of their EV/s; the batch MIXES variants (R4). Every captured
decision in an investigation is trained with that investigation's advantage — the decision CHAIN that
found the better method (R3). Warm-started from the SFT adapter (SR_SFT_CKPT) so cold-start has signal.

Env: SR_STEPS(200) SR_G(3) SR_BVARS(2) SR_MAXSTEPS(12) SR_LR(1e-5) SR_BETA(0.02) SR_SAVE(10)
     SR_MAXNEW(320) SR_SFT_CKPT(rl_infra/ckpt_sft) SCI_PROMPT_BUDGET_CHARS(9000).
"""
import os, sys, time, re, glob
import torch

RL = "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/rl_infra"
for p in (RL, "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/score_jed",
          "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/ai-agent-security-multi-step-tool-attacks"):
    if p not in sys.path: sys.path.insert(0, p)

os.environ.setdefault("SCI_PROMPT_BUDGET_CHARS", "9000")   # trainable-size strategist prompt (16GB)

import engine.llm_client as llm_client                # noqa: E402
from engine.boundary_scientist import BoundaryScientist  # noqa: E402  (for _parse — the parse-ok filter, P0-1)
from sci_common import build_full_scientist, DSL_VARIANTS, make_tree, tree_episode  # noqa: E402

QWY = os.environ.get("QWY_PATH", "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/models/qwythos")
STEPS = int(os.environ.get("SR_STEPS", "200"))
G = int(os.environ.get("SR_G", "3"))
BVARS = int(os.environ.get("SR_BVARS", "2"))
MAXSTEPS = int(os.environ.get("SR_MAXSTEPS", "12"))
LR = float(os.environ.get("SR_LR", "1e-5"))
BETA = float(os.environ.get("SR_BETA", "0.02"))
SAVE = int(os.environ.get("SR_SAVE", "10"))
# 320 (matches the docstring): a full scientist JSON action (hypothesis+message+reasoning) is
# typically 200-400 tokens — 160 truncated it mid-JSON so often that the parse-rescue path became
# the learned output distribution (P0-3).
MAXNEW = int(os.environ.get("SR_MAXNEW", "320"))
FORMAT_NEG = float(os.environ.get("SR_FORMAT_NEG", "0.5"))   # fixed penalty advantage for unparseable output (P0-1)
SFT_CKPT = os.environ.get("SR_SFT_CKPT", RL + "/ckpt_sft")
OUT = os.environ.get("SR_OUT", RL + "/ckpt_sci_real")
RESUME = os.environ.get("SR_RESUME", "")   # "auto" (latest checkpoint-N in OUT) or a checkpoint dir

_POLICY = {"model": None, "tok": None, "dev": None, "capture": None}


def _render(messages):
    return "\n\n".join(m.get("content", "") for m in messages)


def _parse_ok(text: str) -> bool:
    """Would the scientist's parser ACCEPT this output as an executable decision? (P0-1 filter.)

    A parse_error action or an empty-message send never reaches the env (the scientist loop
    retries/aborts it), so it must NEVER share the investigation's advantage."""
    try:
        parsed = BoundaryScientist._parse(text)
        if parsed.get("action") == "parse_error":
            return False
        return parsed.get("action") != "send" or bool(parsed.get("message"))
    except Exception:
        return False


def policy_chat(messages, *, n=1, model=None, temperature=None):
    tok, mdl, dev = _POLICY["tok"], _POLICY["model"], _POLICY["dev"]
    _t0 = time.time()
    prompt = _render(messages)
    # --- vLLM CONTAINER path: fast rollout gen via the frozen-for-this-step adapter. The captured
    #     prompt/comp token ids feed the SAME HF loss verbatim, so only the engine changed. ---
    vp = _POLICY.get("vllm")
    if vp is not None:
        text, p_ids, c_ids = vp.generate(prompt, max_new_tokens=MAXNEW,
                                         temperature=(1.0 if temperature is None else float(temperature)))
        _POLICY["ncall"] = _POLICY.get("ncall", 0) + 1
        print("[SR.gen] (vllm) call#%d prompt_tok=%d new_tok=%d %.1fs" %
              (_POLICY["ncall"], len(p_ids), len(c_ids), time.time() - _t0), flush=True)
        if _POLICY["capture"] is not None and c_ids:
            _POLICY["capture"].append({"prompt_ids": torch.tensor(p_ids, dtype=torch.long),
                                       "comp_ids": torch.tensor(c_ids, dtype=torch.long),
                                       "parse_ok": _parse_ok(text)})
        return [text]
    ids = tok(prompt, return_tensors="pt").input_ids
    if ids.shape[1] > 4096:
        ids = ids[:, -4096:]      # TAIL-keep: the schema + latest feedback live at the END
    ids = ids.to(dev)             # (HF truncation=True keeps the HEAD — the wrong half; also
    # consistent with the SFT tail-keep and the loss-phase 4600 tail-keep below)
    # NOTE: KV cache + gradient-checkpointing are toggled at the ROLLOUT/LOSS phase boundaries in the
    # step loop (cache ON here). Do NOT touch use_cache per-call — a per-call reset to False would
    # make every generation after the first O(n^2).
    with torch.no_grad():
        out = mdl.generate(ids, max_new_tokens=MAXNEW, do_sample=True,
                           temperature=(1.0 if temperature is None else float(temperature)),
                           top_p=0.95, pad_token_id=tok.eos_token_id)
    comp = out[0, ids.shape[1]:]
    text = tok.decode(comp, skip_special_tokens=True)
    _POLICY["ncall"] = _POLICY.get("ncall", 0) + 1
    print("[SR.gen] call#%d prompt_tok=%d new_tok=%d %.1fs" %
          (_POLICY["ncall"], ids.shape[1], comp.shape[0], time.time() - _t0), flush=True)
    if _POLICY["capture"] is not None:
        _POLICY["capture"].append({"prompt_ids": ids[0].detach().cpu(), "comp_ids": comp.detach().cpu(),
                                   "parse_ok": _parse_ok(text)})
    return [text]


def _lsoftmax(logits, index):
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, index.unsqueeze(-1)).squeeze(-1)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    tok = AutoTokenizer.from_pretrained(QWY, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    AWQ_MODE = os.environ.get("SR_AWQ") == "1" or "awq" in QWY.lower()
    EIGHT_BIT = (not AWQ_MODE) and os.environ.get("SR_8BIT", "0") == "1"
    FOUR_BIT = (not AWQ_MODE) and (not EIGHT_BIT) and os.environ.get("SR_4BIT", "1") != "0"
    KBIT = FOUR_BIT or EIGHT_BIT or AWQ_MODE   # all quantized paths use gradient checkpointing (per-phase)
    if AWQ_MODE:
        # AWQ base (already 4-bit on disk, ~half of bf16) + trainable LoRA on top. transformers auto-
        # detects the AWQ quantization_config and loads via autoawq; no BitsAndBytesConfig. The SAME
        # AWQ checkpoint can later serve in vLLM (A100-native, unlike FP8-E4M3).
        print("[SR] loading policy AWQ + LoRA:", QWY, flush=True)
        model = AutoModelForCausalLM.from_pretrained(QWY, device_map={"": 0}, trust_remote_code=True,
                                                     dtype=torch.bfloat16)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif EIGHT_BIT:
        # int8 QLoRA: ~9GB (vs 4-bit ~6GB), less quantization loss -> better gradient fidelity; NOTE
        # bnb int8 generation is typically a bit SLOWER than 4-bit nf4, so steps lengthen slightly.
        print("[SR] loading policy 8-bit QLoRA:", QWY, flush=True)
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(QWY, quantization_config=bnb, device_map={"": 0},
                                                     trust_remote_code=True, dtype=torch.bfloat16)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif FOUR_BIT:
        print("[SR] loading policy 4-bit QLoRA:", QWY, flush=True)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(QWY, quantization_config=bnb, device_map={"": 0},
                                                     trust_remote_code=True, dtype=torch.bfloat16)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # bf16 on a big GPU (A100 40GB): dequant-free -> generation ~10-20x faster than bnb 4-bit.
        print("[SR] loading policy bf16 (fast gen):", QWY, flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(QWY, device_map={"": 0}, trust_remote_code=True,
                                                         dtype=torch.bfloat16, attn_implementation="sdpa")
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(QWY, device_map={"": 0}, trust_remote_code=True,
                                                         dtype=torch.bfloat16)
        # bf16 on 40GB A100: no gradient checkpointing (not needed) -> generate() keeps the KV cache.
        model.enable_input_require_grads()
    # ---- resolve RESUME (P1-3): auto = latest checkpoint-N in OUT; else an explicit dir ----
    resume_dir, start_step = "", 1
    if RESUME:
        if RESUME == "auto":
            cands = []
            for d in glob.glob(OUT + "/checkpoint-*"):
                m = re.search(r"checkpoint-(\d+)$", d)
                if m and os.path.isfile(os.path.join(d, "adapter_config.json")):
                    cands.append((int(m.group(1)), d))
            if cands:
                start_step, resume_dir = max(cands)[0] + 1, max(cands)[1]
        elif os.path.isdir(RESUME):
            resume_dir = RESUME
            m = re.search(r"checkpoint-(\d+)$", RESUME.rstrip("/\\"))
            if m:
                start_step = int(m.group(1)) + 1

    _lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                           task_type="CAUSAL_LM", target_modules="all-linear")
    if resume_dir:
        print("[SR] RESUME from step %d:" % (start_step - 1), resume_dir,
              "(optimizer state resets — AdamW moments restart)", flush=True)
        model = PeftModel.from_pretrained(model, resume_dir, is_trainable=True)
    elif os.path.isdir(SFT_CKPT):
        print("[SR] warm-start from SFT adapter:", SFT_CKPT, flush=True)
        model = PeftModel.from_pretrained(model, SFT_CKPT, is_trainable=True)
    else:
        print("[SR] NO SFT adapter (cold) — expect weak early signal", flush=True)
        model = get_peft_model(model, _lora_cfg)
    # P0-2: the KL reference must be the SFT/WARM-START POLICY, not the raw base. The old
    # `disable_adapter()` ref returned the raw base — the SFT-learned JSON format had ZERO KL
    # protection, so the first noisy RL steps could destroy it (garbage -> no score -> zero-group
    # variance -> no gradient: the irreversible parse death spiral). Load a FROZEN copy of the
    # warm-start weights as a second adapter "ref"; the KL forward switches to it.
    if os.path.isdir(SFT_CKPT):
        model.load_adapter(SFT_CKPT, adapter_name="ref")        # frozen copy of the SFT policy
    else:
        model.add_adapter("ref", _lora_cfg)                     # cold: ref == policy at step 0
    for _pn, _pp in model.named_parameters():
        if ".ref." in _pn:
            _pp.requires_grad_(False)
    model.set_adapter("default")
    model.print_trainable_parameters(); model.config.use_cache = False
    if KBIT: model.gradient_checkpointing_enable()   # only the 16GB/4-bit path needs checkpointing
    dev = model.device
    _POLICY.update(model=model, tok=tok, dev=dev)
    if os.environ.get("RL_USE_VLLM") == "1":
        # Container swap: rollout gen -> vLLM (base+current adapter); loss stays HF. Seed vLLM with
        # the current (warm-start/resumed) adapter so step 1's rollout already reflects it.
        from vllm_gen import VLLMPolicy
        _init_ad = OUT + "/vllm_adapter_init"
        model.save_pretrained(_init_ad)
        vp = VLLMPolicy(os.environ.get("RL_VLLM_BASE", QWY),
                        gpu_frac=float(os.environ.get("RL_VLLM_GPU_FRAC", "0.35")))
        vp.set_adapter(_init_ad, 0)
        _POLICY["vllm"] = vp
        print("[SR] vLLM CONTAINER active: rollout gen=vLLM, loss=HF (system unchanged)", flush=True)
    llm_client.chat = policy_chat            # the scientist's brain is now the trainable policy
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    print("[SR] start: steps=%d G=%d bvars=%d maxsteps=%d variants=%d" % (STEPS, G, BVARS, MAXSTEPS, len(DSL_VARIANTS)), flush=True)

    for step in range(start_step, STEPS + 1):
        t0 = time.time(); batch = []; logrows = []
        # ROLLOUT phase: cache ON + checkpointing OFF so generate() is fast (HF forces use_cache=False
        # whenever gradient checkpointing is on -> O(n^2) generation; toggle per-phase, not per-call).
        model.eval(); model.config.use_cache = True
        if KBIT: model.gradient_checkpointing_disable()
        # A batch MIXES models (gpt_oss<->gemma) and SDK variants: outer loop over models (grouped so
        # each GGUF is loaded once/step; clear_shared_backends frees the previous target's VRAM before
        # loading the next), inner over BVARS variants. Each (model,variant) is one GRPO group of G
        # investigations. RL_USE_TREE=1 runs the real Go-Explore+PUCT tree (G frontier-guided episodes
        # on a shared per-cycle tree); else the flat root-launched investigations.
        MODELS = [m.strip() for m in os.environ.get("RL_MODELS", "gpt_oss").split(",") if m.strip()]
        USE_TREE = os.environ.get("RL_USE_TREE", "0") == "1"
        NGL = int(os.environ.get("RL_TGT_NGL", "0"))              # -1 = target on GPU (A100)
        import gc as _gc
        for mi, agent_name in enumerate(MODELS):
            if mi > 0:
                try:
                    from engine.gguf_agent import clear_shared_backends
                    clear_shared_backends()
                except Exception:
                    pass
                torch.cuda.empty_cache()                          # free prev model's target before loading next
            for b in range(BVARS):
                params = DSL_VARIANTS[(step * BVARS + b) % len(DSL_VARIANTS)]
                invs, rewards = [], []
                if USE_TREE:
                    # ONE full-memory scientist + ONE shared tree per (model,variant) cycle: memory and
                    # Go-Explore frontiers persist across the G episodes (single-cycle depth), fresh per
                    # cycle (no cross-variant/model inheritance).
                    sci, sb = build_full_scientist(params, max_steps=MAXSTEPS, agent=agent_name, n_gpu_layers=NGL)
                    tree = make_tree(seed=step * 100 + b)
                    for g in range(G):
                        _POLICY["capture"] = []
                        try:
                            reward, _r = tree_episode(sci, sb, tree, set())
                        except Exception as exc:
                            print("[SR] tree_episode err:", repr(exc)[:120], flush=True); reward = 0.0
                        invs.append(list(_POLICY["capture"])); rewards.append(float(reward))  # OFFICIAL EV/s
                        _POLICY["capture"] = None
                    try:
                        sb._sb._env = None
                    except Exception:
                        pass
                    del sci, sb, tree
                    _gc.collect(); torch.cuda.empty_cache()
                else:
                    for g in range(G):
                        sci, sb = build_full_scientist(params, max_steps=MAXSTEPS, agent=agent_name, n_gpu_layers=NGL)
                        _POLICY["capture"] = []
                        try:
                            sci.investigate(set())
                        except Exception as exc:
                            print("[SR] investigate err:", repr(exc)[:120], flush=True)
                        invs.append(list(_POLICY["capture"])); rewards.append(float(sb.best_ev))  # OFFICIAL EV/s
                        _POLICY["capture"] = None
                        # free per-investigation env (target KV/context) + torch cache -> no VRAM creep.
                        try:
                            sb._sb._env = None
                        except Exception:
                            pass
                        del sci, sb
                        _gc.collect(); torch.cuda.empty_cache()
                rt = torch.tensor(rewards)
                # Robust GRPO advantage: torch.std of a SINGLE sample is NaN, and an all-equal group has
                # std 0 -> no signal. Guard both so a degenerate group never poisons the gradient with NaN.
                if rt.numel() > 1 and float(rt.std()) > 1e-6:
                    adv = (rt - rt.mean()) / (rt.std() + 1e-6)
                else:
                    adv = torch.zeros_like(rt)
                adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
                for decisions, a in zip(invs, adv.tolist()):
                    for d in decisions:
                        batch.append({"prompt_ids": d["prompt_ids"], "comp_ids": d["comp_ids"], "adv": a})
                logrows.append(("%s:%s" % (agent_name[:4], params.name), max(rewards), rt.mean().item(), len(sum(invs, []))))

        # Free the target model's VRAM BEFORE the backward pass (user's OOM-avoidance ask): the
        # gpt_oss/gemma GGUF (~14-16GB) is not needed during the loss/optimizer step, and freeing it
        # gives the deeper maxsteps rollouts + backward room on the 40GB card. Both GGUFs stay warm in
        # RAM/page-cache so the next rollout reloads to VRAM fast (~15s). This is also the exact hook
        # used to alternate gpt<->gemma targets between steps.
        try:
            from engine.gguf_agent import clear_shared_backends
            clear_shared_backends()
        except Exception as _e:
            print("[SR] clear_shared_backends skipped:", repr(_e)[:80], flush=True)
        torch.cuda.empty_cache()
        # LOSS phase: checkpointing ON + cache OFF so the backward fits 16GB.
        model.train(); model.config.use_cache = False
        if KBIT: model.gradient_checkpointing_enable()
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
            # SYNC the updated LoRA into the vLLM container ONCE per step (amortized over the next
            # step's ~192 rollout gens): save to a rolling dir, point vLLM at it via a fresh id.
            if _POLICY.get("vllm") is not None:
                _ad = OUT + "/vllm_adapter"
                model.save_pretrained(_ad); _POLICY["vllm"].set_adapter(_ad, step)
        dt = time.time() - t0
        for (name, best, meanR, ndec) in logrows:
            print("[SR] step %d %-11s bestEV/s=%.2f meanEV/s=%.2f decisions=%d used=%d kl=%.3f gnorm=%.2f (%.0fs)"
                  % (step, name, best, meanR, ndec, used, (kl_sum / used if used else 0), gnorm, dt), flush=True)
        if step % SAVE == 0:
            d = "%s/checkpoint-%d" % (OUT, step); model.save_pretrained(d); tok.save_pretrained(d)
            print("[SR] saved", d, flush=True)
    model.save_pretrained(OUT + "/final"); print("[SR] DONE", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.stdout.flush(); raise
