"""vLLM generation CONTAINER for the RL policy — swaps ONLY the rollout generation engine.

Design (the standard RLHF split; nothing else in the system changes):
  * ROLLOUT: the trainable LoRA is frozen for the step -> vLLM serves base+LoRA and generates every
    scientist decision FAST (5-10x the bnb-4bit HF generate()). We read back prompt_token_ids +
    output.token_ids so the loss phase is byte-identical to the HF path.
  * LOSS: unchanged -> the HF trainable model does the forward pass on those captured ids for logp.
  * SYNC (once/step, amortized over ~192 gens): after opt.step, the trainer saves the updated LoRA
    adapter to a fresh dir; VLLMPolicy.set_adapter(dir, step) points vLLM at it via a new LoRARequest
    id, so the next rollout reflects the update. No per-generation sync.

The scientist / DSL / tree / reward / GRPO loss are untouched — this is a runtime container, not a
system change. Requires a vLLM-compatible env (torch 2.13 / CUDA 13) — see cutover script; do NOT
import this in the torch-2.11 training env.

Env: RL_VLLM_BASE (policy base path), RL_VLLM_GPU_FRAC (0.35), RL_VLLM_MAXLORA (16).
"""
from __future__ import annotations
import os


class VLLMPolicy:
    def __init__(self, base_path: str, *, max_lora_rank: int = 16, gpu_frac: float = 0.35,
                 max_model_len: int = 8192, dtype: str = "bfloat16"):
        from vllm import LLM, SamplingParams  # noqa: import here so the training env never touches vLLM
        self._SamplingParams = SamplingParams
        # gpu_frac is deliberately small: vLLM shares the card with the HF trainable model + the
        # (also vLLM-served) target. LoRA enabled so we can hot-swap the trained adapter each step.
        # vLLM 0.28 does NOT support bitsandbytes quant; qwythos is unquantized bf16, so bf16 is the
        # only in-flight option (an offline AWQ/GPTQ convert would shrink it). RL_VLLM_QUANT can pass a
        # vLLM-supported method (awq/gptq/...) if a pre-quantized base is provided.
        # qwythos = Qwen3.5-VL: a MAMBA-hybrid, vision-language arch. Per the vLLM Qwen3.5 docs it
        # needs trust_remote_code and a Mamba SSM cache dtype; VL adds the vision tower on top. These
        # flags give a retry a real chance (the first attempt omitted them and never surfaced the
        # EngineCore root error). RL_VLLM_MAMBA_DTYPE (float16/float32), RL_VLLM_QUANT for a pre-quant base.
        max_model_len = int(os.environ.get("RL_VLLM_MAXLEN", str(max_model_len)))   # smaller = less KV
        _kw = dict(model=base_path, enable_lora=True, max_lora_rank=max_lora_rank,
                   max_model_len=max_model_len, dtype=dtype, trust_remote_code=True,
                   gpu_memory_utilization=gpu_frac, enforce_eager=True)
        _mdt = os.environ.get("RL_VLLM_MAMBA_DTYPE")
        if _mdt:
            _kw["mamba_ssm_cache_dtype"] = _mdt
        _q = os.environ.get("RL_VLLM_QUANT")
        if _q:
            _kw["quantization"] = _q
        self.llm = LLM(**_kw)
        self._adapter_dir: str | None = None
        self._lora_id: int = 0

    def set_adapter(self, adapter_dir: str, step: int) -> None:
        """Point vLLM at the freshly-saved LoRA (once/step). A new id forces vLLM to reload it."""
        self._adapter_dir = adapter_dir
        self._lora_id = int(step) + 1        # non-zero, monotonic -> vLLM treats each step's adapter as new

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float = 0.95):
        """Return (text, prompt_token_ids, completion_token_ids) — the ids feed the HF loss verbatim."""
        from vllm.lora.request import LoRARequest
        sp = self._SamplingParams(max_tokens=max_new_tokens, temperature=max(temperature, 0.01), top_p=top_p)
        lora = (LoRARequest("policy", self._lora_id, self._adapter_dir)
                if self._adapter_dir else None)
        out = self.llm.generate([prompt], sp, lora_request=lora, use_tqdm=False)[0]
        comp = out.outputs[0]
        return comp.text, list(out.prompt_token_ids), list(comp.token_ids)
