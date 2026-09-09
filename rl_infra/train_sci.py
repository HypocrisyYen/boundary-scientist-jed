"""Hierarchical GRPO for the scientist SOLVER (train the decision chain that finds good methods).

Per step: pick B_VARS variants from the diverse reachable generator; for each, expand G diverse
BRANCHES from the champion frontier (L1 strategy + L2 action chain, selfplay_sci.rollout_branch).
Reward (A1) = breakthrough = branch_best/floor - 1. GRPO advantage = normalize within each variant's
G branches (a group); the BATCH mixes branches from all B_VARS variants (req: batch spans SDKs).
Loss trains ONLY the policy's STRATEGY+ACTION tokens (gen_mask): -adv*Σlogp + beta*KL(policy||base).
Fresh env/champion/context per variant (no cross-variant inheritance).

Env: SCI_STEPS(300) SCI_G(4) SCI_T(2) SCI_BVARS(2) SCI_LR(1e-5) SCI_BETA(0.02) SCI_SAVE(20)
     SCI_RMAX(8) RL_TGT_HOPS(2) RL_TGT_TOKENS(320).
"""
import os, sys, time
import torch

RL = "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/rl_infra"
JED = "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/score_jed"
SDK = "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/ai-agent-security-multi-step-tool-attacks"
for p in (RL, JED, SDK):
    if p not in sys.path: sys.path.insert(0, p)

from engine import sdk_variants as V           # noqa: E402
from selfplay import SolverEnv                  # noqa: E402
from selfplay_sci import rollout_branch         # noqa: E402

QWY = os.environ.get("QWY_PATH", "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/models/qwythos")
STEPS = int(os.environ.get("SCI_STEPS", "300"))
G = int(os.environ.get("SCI_G", "4"))
T = int(os.environ.get("SCI_T", "2"))
BVARS = int(os.environ.get("SCI_BVARS", "2"))
LR = float(os.environ.get("SCI_LR", "1e-5"))
BETA = float(os.environ.get("SCI_BETA", "0.02"))
SAVE = int(os.environ.get("SCI_SAVE", "20"))
RMAX = float(os.environ.get("SCI_RMAX", "8"))
OUT = os.environ.get("SCI_OUT", RL + "/ckpt_sci")

# GENERATOR: diverse reachable holes, each a DIFFERENT method (pack / many-posts / many-targets /
# fs.read / email / PIVOT) so the solver must probe & adapt -> generalizable decision chains.
_NAMED = ["no_dedup", "double_exfil", "cell_per_event", "broad_sentinel",
          "read_open", "cd5_open", "datascan_pivot", "clean"]
_FLOORS = {}


def sample_spec(i):
    if (i % 10) < 8:
        return V.VULN_CATALOG[_NAMED[i % len(_NAMED)]]
    return V.random_spec(seed=7000 + i)


def _lsoftmax(logits, index):
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, index.unsqueeze(-1)).squeeze(-1)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    print("[SCI] loading policy 4-bit QLoRA:", QWY, flush=True)
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
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    dev = model.device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    print("[SCI] start: steps=%d G=%d T=%d bvars=%d lr=%g beta=%g" % (STEPS, G, T, BVARS, LR, BETA), flush=True)

    for step in range(1, STEPS + 1):
        t0 = time.time()
        batch = []            # list of dicts: {ids, gen_mask, adv, spec}
        logrows = []
        model.eval()
        for b in range(BVARS):
            spec = sample_spec(step * BVARS + b)
            # fresh env + champion per variant (no inheritance)
            env0 = SolverEnv(spec)
            floor = _FLOORS.get(spec.name) or env0.floor()
            _FLOORS[spec.name] = floor
            branches, rewards = [], []
            for g in range(G):
                env = SolverEnv(spec); env._floor = floor       # independent env per branch
                ro = rollout_branch(model, tok, env, T=T, device=dev, champion=0.0,
                                    recent=[], facts=[], temperature=1.0)
                branches.append(ro)
                rewards.append(float(ro["branch_best"]))        # RAW campaign score (P0: not a clipped ratio)
            rt = torch.tensor(rewards)
            # GRPO advantage = z-score of campaign_raw WITHIN the group (the docstring-promised version).
            # No floor-ratio / clip -> no saturation-to-cap and no ratio-explosion; floor stays logging-only.
            adv = (rt - rt.mean()) / (rt.std() + 1e-6)
            for ro, a in zip(branches, adv.tolist()):
                batch.append({"ids": ro["ids"], "gen_mask": ro["gen_mask"], "adv": a})
            bestb = max(branches, key=lambda r: r["branch_best"])
            logrows.append((spec.name, floor, max(r["branch_best"] for r in branches),
                            rt.mean().item(), max(rewards),
                            (bestb["strategy"][:40], bestb["actions"][-1][0][:44] if bestb["actions"] else "")))

        # ---- one GRPO update over the MIXED batch (branches from all variants) ----
        model.train(); opt.zero_grad()
        used = 0; kl_sum = 0.0
        for item in batch:
            a = item["adv"]
            if abs(a) < 1e-6:
                continue
            ids = item["ids"].unsqueeze(0)
            m = item["gen_mask"][1:].float().to(dev)
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
            (pg + BETA * kl).backward()
            used += 1; kl_sum += float(kl.item())
        gnorm = 0.0
        if used:
            gnorm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
            opt.step()

        dt = time.time() - t0
        for (name, floor, best, meanR, maxR, (strat, act)) in logrows:
            print("[SCI] step %d %-15s floor=%.0f best=%.0f BT=%.2f meanR=%.2f used=%d kl=%.3f gnorm=%.2f (%.0fs)"
                  " strat=%r act=%r"
                  % (step, name, floor, best, best / max(floor, 1e-6) - 1.0, meanR, used,
                     (kl_sum / used if used else 0), gnorm, dt, strat, act), flush=True)
        if step % SAVE == 0:
            d = "%s/checkpoint-%d" % (OUT, step); model.save_pretrained(d); tok.save_pretrained(d)
            print("[SCI] saved", d, flush=True)

    model.save_pretrained(OUT + "/final"); print("[SCI] DONE", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.stdout.flush(); raise
