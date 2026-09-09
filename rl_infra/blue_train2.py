"""Train a BLUE-JED defender adapter on examples DERIVED FROM the red scientist's exploration
(rl_infra/blue_sft.jsonl). QLoRA on unsloth/Qwen3-8B-bnb-4bit, completion-only loss (labels=-100 on
the prompt), so gradient flows only through the assistant reply (refuse vs comply). Small + fast."""
import os, json, torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import Dataset

BASE = os.environ.get("SFT_BASE", "unsloth/Qwen3-8B-bnb-4bit")
DATA = os.environ.get("BLUE_DATA", "rl_infra/blue_sft.jsonl")
OUT = os.environ.get("BLUE_OUT", "sft_model/adapter_8b_blue")
MAXLEN = 1024
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
print("[blue-train] %d examples" % len(rows), flush=True)

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def enc(e):
    try:
        prompt = tok.apply_chat_template([{"role":"system","content":e["system"]},
                                          {"role":"user","content":e["user"]}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template([{"role":"system","content":e["system"]},
                                          {"role":"user","content":e["user"]}],
                                         tokenize=False, add_generation_prompt=True)
    full = prompt + e["assistant"] + tok.eos_token
    pid = tok(prompt, add_special_tokens=False).input_ids
    fid = tok(full, add_special_tokens=False, truncation=True, max_length=MAXLEN).input_ids
    lab = list(fid)
    for j in range(min(len(pid), len(lab))): lab[j] = -100
    return {"input_ids": fid, "attention_mask": [1]*len(fid), "labels": lab}

ds = Dataset.from_list(rows).map(enc, remove_columns=["system","user","assistant"])
model = AutoModelForCausalLM.from_pretrained(BASE, device_map={"":0}, trust_remote_code=True, dtype="auto")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
model.print_trainable_parameters()
args = TrainingArguments(output_dir=OUT, num_train_epochs=6, per_device_train_batch_size=2,
    gradient_accumulation_steps=4, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=5,
    logging_steps=5, save_strategy="no", bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(), gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch", report_to="none")
Trainer(model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
model.save_pretrained(OUT); tok.save_pretrained(OUT)
print("[blue-train] DONE -> %s" % OUT, flush=True)
