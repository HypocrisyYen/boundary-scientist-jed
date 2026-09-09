"""(B) SFT warm-start: behavior-clone the 9B on the collected (scientist prompt -> JSON action) pairs
so it emits valid scientist actions that score (schema + the discovery methods for the real holes).
Completion-only loss (prompt tokens masked). 4-bit QLoRA. Output adapter feeds RL (train_sci_real).

Env: SFT_DATA(rl_infra/sft_data.jsonl) SFT_EPOCHS(3) SFT_LR(1e-4) SFT_MAXLEN(1600) SFT_CKPT(rl_infra/ckpt_sft).
"""
import os, sys, json, math
import torch

RL = "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/rl_infra"
QWY = os.environ.get("QWY_PATH", "E:/AI_Agent_Security-Multi-Step_Tool_Attacks/models/qwythos")
DATA = os.environ.get("SFT_DATA", RL + "/sft_data.jsonl")
EPOCHS = int(os.environ.get("SFT_EPOCHS", "3"))
LR = float(os.environ.get("SFT_LR", "1e-4"))
MAXLEN = int(os.environ.get("SFT_MAXLEN", "768"))    # prompt(tail)+completion tokens; short -> fast backward, 16GB-safe
CKPT = os.environ.get("SFT_CKPT", RL + "/ckpt_sft")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    print("[SFTt] %d pairs; loading 4-bit QLoRA %s" % (len(rows), QWY), flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(QWY, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(QWY, quantization_config=bnb, device_map={"": 0},
                                                 trust_remote_code=True, dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules="all-linear")
    model = get_peft_model(model, lora); model.print_trainable_parameters()
    model.gradient_checkpointing_enable(); model.config.use_cache = False; model.train()
    dev = model.device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    # Pre-tokenize: build (input_ids, labels) with the PROMPT masked (-100), completion supervised.
    exs = []
    for r in rows:
        p_ids = tok(r["prompt"], add_special_tokens=False).input_ids
        c_ids = tok("\n" + r["completion"], add_special_tokens=False).input_ids + [tok.eos_token_id]
        # keep the prompt TAIL so the completion stays in context within MAXLEN
        keep_p = max(0, MAXLEN - len(c_ids))
        p_ids = p_ids[-keep_p:]
        ids = p_ids + c_ids
        labels = [-100] * len(p_ids) + list(c_ids)
        exs.append((ids, labels))
    print("[SFTt] tokenized; median len=%d" % sorted(len(i) for i, _ in exs)[len(exs) // 2], flush=True)

    step = 0
    for ep in range(EPOCHS):
        for ids, labels in exs:
            t = torch.tensor([ids], device=dev)
            lab = torch.tensor([labels], device=dev)
            out = model(t, labels=lab)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            step += 1
            if step % 5 == 0:
                print("[SFTt] ep%d step%d loss=%.3f" % (ep, step, float(loss.item())), flush=True)
            if step % int(os.environ.get("SFT_SAVE_EVERY", "30")) == 0:
                model.save_pretrained(CKPT); tok.save_pretrained(CKPT)
                print("[SFTt] saved adapter @ step%d -> %s" % (step, CKPT), flush=True)
    model.save_pretrained(CKPT); tok.save_pretrained(CKPT)
    print("[SFTt] DONE -> %s" % CKPT, flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.stdout.flush(); raise
