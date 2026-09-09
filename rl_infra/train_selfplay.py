"""Custom GRPO self-play trainer: the SOLVER learns a GENERAL 'find a higher-scoring method in
ANY env' ability by multi-turn interaction with DSL-variant environments (the GENERATOR).

Why custom (not TRL GRPOTrainer): multi-turn needs exact token masking (train ONLY on the
solver's ATTEMPT tokens, never on injected env FEEDBACK). TRL's environment_factory path needs a
tool-calling chat template (Qwythos has none) and rollout_func has no completion-mask. So we drive
generation + loss ourselves with selfplay.rollout_episode's gen_mask.

Per step: pick ONE env spec (from the diverse generator); run G independent K-turn episodes on it;
reward = clip(breakthrough = best_score/floor - 1); GRPO advantage = group-normalize the G rewards;
loss = -adv * sum(logp of solver tokens) + beta*KL(policy||base). Grouping is WITHIN one env, so the
advantage says 'which trajectory found a better method here' — trains search, generalizes across envs.

Env: SP_STEPS(300) SP_G(4) SP_K(3) SP_LR(1e-5) SP_BETA(0.02) SP_SAVE(20) SP_RMAX(8) RL_TGT_HOPS(2).
"""
import os, sys, time, random
import torch

RL = "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/rl_infra"
JED = "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/score_jed"
SDK = "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/ai-agent-security-multi-step-tool-attacks"
for p in (RL, JED, SDK):
    if p not in sys.path: sys.path.insert(0, p)

from engine import sdk_variants as V          # noqa: E402
from selfplay import SolverEnv, rollout_episode  # noqa: E402

QWY = os.environ.get("QWY_PATH", "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/models/qwythos")
STEPS = int(os.environ.get("SP_STEPS", "300"))
G = int(os.environ.get("SP_G", "4"))
K = int(os.environ.get("SP_K", "3"))
LR = float(os.environ.get("SP_LR", "1e-5"))
BETA = float(os.environ.get("SP_BETA", "0.02"))
SAVE = int(os.environ.get("SP_SAVE", "20"))
RMAX = float(os.environ.get("SP_RMAX", "8"))
OUT = os.environ.get("SP_OUT", RL + "/ckpt_selfplay")

# GENERATOR: the diverse blind-SDK distribution — each hole has a DIFFERENT reachable method, so
# the solver must probe & adapt (packing / many-posts / many-targets / fs.read / email / PIVOT).
_NAMED = ["no_dedup", "double_exfil", "cell_per_event", "broad_sentinel",
          "read_open", "cd5_open", "datascan_pivot", "clean"]
_FLOORS = {}   # spec.name -> floor (cache; family_sweep is expensive on the CPU target)


def sample_spec(step):
    # 70% curated holes (round-robin), 30% fuzzed random specs (generalization pressure)
    if (step % 10) < 7:
        return V.VULN_CATALOG[_NAMED[(step - 1) % len(_NAMED)]]   # step-1 so index 0 (no_dedup) isn't skipped
    return V.random_spec(seed=1000 + step)


def _lsoftmax(logits, index):
    # selective log-softmax at the given token indices; numerically stable
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, index.unsqueeze(-1)).squeeze(-1)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    print("[SP] loading policy 4-bit QLoRA:", QWY, flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(QWY, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(QWY, quantization_config=bnb, device_map={"": 0},
                                                 trust_remote_code=True, dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules="all-linear")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    dev = model.device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    print("[SP] start: steps=%d G=%d K=%d lr=%g beta=%g" % (STEPS, G, K, LR, BETA), flush=True)

    for step in range(1, STEPS + 1):
        t0 = time.time()
        spec = sample_spec(step)
        # G independent episodes on the SAME env spec (share the cached floor)
        model.eval()
        eps, rewards = [], []
        for g in range(G):
            env = SolverEnv(spec)
            if spec.name in _FLOORS:
                env._floor = _FLOORS[spec.name]
            ro = rollout_episode(model, tok, env, K=K, device=dev, max_new=48, temperature=1.0)
            _FLOORS[spec.name] = env._floor
            eps.append(ro)
            rewards.append(max(-1.0, min(RMAX, ro["breakthrough"])))
        rt = torch.tensor(rewards)
        adv = (rt - rt.mean()) / (rt.std() + 1e-6)

        # ---- GRPO policy-gradient update (train only on solver tokens via gen_mask) ----
        model.train()
        opt.zero_grad()
        used = 0; loss_val = 0.0; kl_val = 0.0
        for ro, a in zip(eps, adv.tolist()):
            if abs(a) < 1e-6:            # zero advantage -> no signal
                continue
            ids = ro["ids"].unsqueeze(0)
            m = ro["gen_mask"][1:].float().to(dev)
            if m.sum() < 1:
                continue
            logits = model(ids).logits[0][:-1]
            logp = _lsoftmax(logits, ids[0][1:])
            with torch.no_grad():
                with model.disable_adapter():
                    ref_logits = model(ids).logits[0][:-1]
                ref_logp = _lsoftmax(ref_logits, ids[0][1:])
            pg = -(a * (logp * m).sum()) / m.sum()
            kl = (((torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0) * m).sum()) / m.sum()
            loss = pg + BETA * kl
            loss.backward()
            used += 1; loss_val += float(loss.item()); kl_val += float(kl.item())
        gnorm = 0.0
        if used:
            gnorm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
            opt.step()

        bts = [ro["breakthrough"] for ro in eps]
        beat = sum(1 for b in bts if b > 0.01) / len(bts)
        best_ep = max(eps, key=lambda r: r["breakthrough"])
        ba = best_ep["transcript"][-1][0] if best_ep["transcript"] else ""
        print("[SP] step %d %-16s meanR=%.3f maxBT=%.2f beat=%.2f floor=%.0f used=%d kl=%.3f gnorm=%.2f (%.0fs) best=%r"
              % (step, spec.name, rt.mean().item(), max(bts), beat, best_ep["floor"], used,
                 (kl_val / used if used else 0), gnorm, time.time() - t0, ba[:60]), flush=True)
        if step % SAVE == 0:
            d = "%s/checkpoint-%d" % (OUT, step)
            model.save_pretrained(d); tok.save_pretrained(d)
            print("[SP] saved", d, flush=True)

    model.save_pretrained(OUT + "/final"); print("[SP] DONE", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.stdout.flush(); raise
