"""Pluggable attacker-LLM client (analog of score_golf.kimi_client).

Reads configuration from env / an optional ``llm_config.json`` and exposes a
single ``chat`` entry point. If no API key is configured it degrades gracefully
to "no client" so the whole search still runs on DSL mutation + the strategy
bank alone (the system is never blocked on an external service).

Supported backends (auto-detected from env):
  * OpenAI-compatible (OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL)
  * OpenRouter        (OPENROUTER_API_KEY)
  * Moonshot / Kimi   (MOONSHOT_API_KEY, base https://api.moonshot.cn/v1)
  * Local GGUF        (LLM_LOCAL_GGUF=path[, LLM_LOCAL_NGL, LLM_LOCAL_CTX]) -- self-contained, no API
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Request robustness (the strategist IS the engine — a silent API stall used to
# degrade the whole run to empty output with no signal). Tunable via env.
_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "180"))
_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
_BACKOFF_S = float(os.environ.get("LLM_BACKOFF_S", "2.0"))

# Process-wide telemetry so the loop can SEE whether the strategist is healthy
# (calls, transient failures retried, hard failures, parse-empties, wall time,
# and the last error string) instead of flying blind.
TELEMETRY: dict[str, float | int | str] = {
    "calls": 0, "ok": 0, "retries": 0, "hard_fail": 0, "empty": 0, "wall_s": 0.0, "last_error": "",
}

# Whether this (base_url, model) accepts response_format={"type":"json_object"} — forces the
# API's own decoder to emit valid JSON, eliminating the "model writes 30KB of prose instead of
# the schema" failure class at the source rather than patching the regex extractor further.
# Learned per endpoint (not hardcoded): optimistically tried once; if the provider rejects the
# param, cached as unsupported so we never pay for a failing probe again. Some OpenAI-compatible
# gateways (OpenRouter/Moonshot) may not support it identically to the primary Agnes endpoint.
_JSON_MODE_OK: dict[tuple[str, str], bool] = {}


def telemetry_snapshot() -> dict:
    return dict(TELEMETRY)


@dataclass
class LLMConfig:
    api_key: str | None
    base_url: str | None
    model: str
    max_tokens: int = 1200
    temperature: float = 0.6
    # Mitigates a REAL failure mode found via the trace mechanism (2026-08-13): a long episode
    # can send the strategist into a verbatim-repetition loop ("Let me construct a message... "
    # repeated near-identically) that burns the whole token budget and NEVER reaches JSON
    # (has_json=False — not a truncated-mid-JSON issue, response_format alone doesn't prevent
    # this). A mild frequency_penalty is the standard mitigation for exact-phrase looping.
    frequency_penalty: float = 0.3
    presence_penalty: float = 0.0
    # Local GGUF strategist backend (self-contained, no external API). When set,
    # chat()/chat_many() run an in-process llama.cpp model instead of the OpenAI
    # client, so the scientist works with NO network / API key (LLM_LOCAL_GGUF=path).
    local_gguf: str | None = None
    local_ngl: int = -1
    local_ctx: int = 8192
    # Local HF/safetensors strategist backend (transformers + optional PEFT LoRA adapter). Lets the
    # RL-updated model be the scientist core DIRECTLY (no GGUF conversion): LLM_LOCAL_HF=base_path
    # [, LLM_LOCAL_ADAPTER=lora_path, LLM_LOCAL_4BIT=1].
    local_hf: str | None = None
    local_adapter: str | None = None
    local_4bit: bool = True
    local_8bit: bool = False   # LLM_LOCAL_8BIT=1 -> int8 (~9GB, less quant loss); takes precedence over 4bit

    @property
    def enabled(self) -> bool:
        return bool(self.api_key or self.local_gguf or self.local_hf)

    @property
    def is_local(self) -> bool:
        return bool(self.local_gguf or self.local_hf)   # explicit local opt-in wins over any API key

    @property
    def is_hf(self) -> bool:
        return bool(self.local_hf) and not self.local_gguf


def _load_config_file() -> dict:
    for name in ("llm_config.json", "score_jed/llm_config.json"):
        p = Path(name)
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def resolve_config(model: str | None = None) -> LLMConfig:
    cfg = _load_config_file()
    env = os.environ.get
    api_key = (
        env("OPENAI_API_KEY")
        or env("OPENROUTER_API_KEY")
        or env("MOONSHOT_API_KEY")
        or cfg.get("api_key")
    )
    base_url = env("OPENAI_BASE_URL") or cfg.get("base_url")
    if not base_url:
        if env("OPENROUTER_API_KEY"):
            base_url = "https://openrouter.ai/api/v1"
        elif env("MOONSHOT_API_KEY"):
            base_url = "https://api.moonshot.cn/v1"
    resolved_model = model or env("LLM_MODEL") or cfg.get("model") or "gpt-4o-mini"
    # Thinking models (e.g. Agnes 2.0 Flash) spend the whole budget on hidden
    # reasoning before emitting the JSON, so default generously and allow an env
    # override (LLM_MAX_TOKENS). 6000 lets reasoning + a 3-program JSON complete.
    max_tokens = int(env("LLM_MAX_TOKENS") or cfg.get("max_tokens", 8000))
    temperature = float(env("LLM_TEMPERATURE") or cfg.get("temperature", 0.6))
    frequency_penalty = float(env("LLM_FREQUENCY_PENALTY") or cfg.get("frequency_penalty", 0.3))
    presence_penalty = float(env("LLM_PRESENCE_PENALTY") or cfg.get("presence_penalty", 0.0))
    local_gguf = env("LLM_LOCAL_GGUF") or cfg.get("local_gguf")
    local_ngl = int(env("LLM_LOCAL_NGL") or cfg.get("local_ngl", -1))
    local_ctx = int(env("LLM_LOCAL_CTX") or cfg.get("local_ctx", 8192))
    local_hf = env("LLM_LOCAL_HF") or cfg.get("local_hf")
    local_adapter = env("LLM_LOCAL_ADAPTER") or cfg.get("local_adapter")
    local_4bit = str(env("LLM_LOCAL_4BIT") or cfg.get("local_4bit", "1")).lower() not in ("0", "false", "")
    local_8bit = str(env("LLM_LOCAL_8BIT") or cfg.get("local_8bit", "0")).lower() in ("1", "true", "yes")
    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=resolved_model,
        max_tokens=max_tokens,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        local_gguf=local_gguf,
        local_ngl=local_ngl,
        local_ctx=local_ctx,
        local_hf=local_hf,
        local_adapter=local_adapter,
        local_4bit=local_4bit,
        local_8bit=local_8bit,
    )


# --- Local GGUF strategist backend (self-contained; no API/network) --------------
# Cached per (path, ngl, ctx) so repeated calls reuse one loaded model. Uses the same
# process-wide backend registry as the target agent when available, else a bare Llama.
_LOCAL_LLM: dict[tuple, object] = {}


def _get_local_llm(cfg: "LLMConfig"):
    key = (cfg.local_gguf, cfg.local_ngl, cfg.local_ctx)
    llm = _LOCAL_LLM.get(key)
    if llm is None:
        from llama_cpp import Llama
        llm = Llama(
            model_path=cfg.local_gguf,
            n_ctx=cfg.local_ctx,
            n_gpu_layers=cfg.local_ngl,
            verbose=False,
        )
        _LOCAL_LLM[key] = llm
    return llm


def _strip_harmony(text: str) -> str:
    """gpt-oss (Harmony) emits raw channel control tokens in content, e.g.
    '<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>ANSWER'.
    Keep only the FINAL channel's message (the real answer) and drop any leftover control
    tokens, so the downstream JSON extractor sees clean content. No-op for non-Harmony models."""
    import re
    if "<|channel|>final<|message|>" in text:
        text = text.rsplit("<|channel|>final<|message|>", 1)[-1]
    # Qwen3-family thinking models: keep only what's AFTER the closing </think> (the real answer). If a
    # </think> exists, drop everything up to it; also strip any dangling open <think>...(unclosed).
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"</?think>", "", text)
    return re.sub(r"<\|[^|]*\|>", "", text).strip()


# --- Local HF/safetensors strategist backend (transformers + optional PEFT LoRA) ----------------
# Lets the RL-updated model (base + adapter, safetensors) be the scientist core with NO GGUF convert.
_HF: dict = {}


def _get_hf(cfg: "LLMConfig"):
    key = (cfg.local_hf, cfg.local_adapter, cfg.local_4bit, getattr(cfg, "local_8bit", False))
    ent = _HF.get(key)
    if ent is None:
        import os as _os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # offline-friendly: a Kaggle model dir (e.g. /kaggle/input/models/.../14b-fp8/1) is a local
        # path; try normal load, fall back to local_files_only if the hub is unreachable.
        _offline = _os.path.isdir(str(cfg.local_hf))
        tok = AutoTokenizer.from_pretrained(cfg.local_hf, trust_remote_code=True,
                                            local_files_only=_offline)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # device_map="auto" spreads the strategist across all visible GPUs (e.g. Kaggle T4x2).
        # dtype="auto" is CRITICAL for pre-quantized checkpoints (fp8/awq/gptq, e.g. a Kaggle
        # qwen-3 14b-fp8): forcing bfloat16 ignores/mis-loads the checkpoint's own quant config.
        # Only override dtype/quant when the caller explicitly asks for bitsandbytes 4/8-bit.
        kw = dict(device_map="auto", trust_remote_code=True, dtype="auto",
                  local_files_only=_offline)
        # OOM mitigation (submission): our brain shares the 2xT4 with the EVALUATOR's target model.
        # device_map="auto" otherwise spreads the brain across BOTH cards, leaving neither with room
        # for the target -> OOM. LLM_LOCAL_MAXMEM_GB caps the brain's per-GPU footprint so the target
        # fits (e.g. "10" -> 10GB/GPU for the brain). Unset = no cap (old behavior).
        _mm = _os.environ.get("LLM_LOCAL_MAXMEM_GB")
        if _mm:
            try:
                import torch as _tq
                _n = _tq.cuda.device_count() or 1
                kw["max_memory"] = {**{i: f"{float(_mm)}GiB" for i in range(_n)}, "cpu": "24GiB"}
            except Exception:
                pass
        if getattr(cfg, "local_8bit", False):
            from transformers import BitsAndBytesConfig
            kw["dtype"] = torch.bfloat16
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)   # ~9GB, less quant loss than 4bit
        elif cfg.local_4bit:
            from transformers import BitsAndBytesConfig
            kw["dtype"] = torch.bfloat16
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(cfg.local_hf, **kw)
        except Exception as _e:
            # fp8/quantized checkpoints occasionally reject device_map/dtype combos — retry minimal.
            TELEMETRY["last_error"] = f"hf_load_retry:{type(_e).__name__}: {_e}"[:200]
            model = AutoModelForCausalLM.from_pretrained(
                cfg.local_hf, device_map="auto", trust_remote_code=True,
                dtype="auto", local_files_only=_offline)
        if cfg.local_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, cfg.local_adapter)   # the RL-updated adapter
        model.eval(); model.config.use_cache = True
        ent = (model, tok)
        _HF[key] = ent
    return ent


def _hf_chat(cfg: "LLMConfig", messages: list[dict], n: int, temp: float) -> list[str]:
    import torch
    model, tok = _get_hf(cfg)
    # CRITICAL: apply the model's chat template so an instruct/SFT model sees IN-DISTRIBUTION input.
    # Without this, a role-tagged conversation is fed as raw concatenated text, which an instruct
    # model (and especially our chat-template-trained SFT adapter) responds to poorly — it stops
    # emitting the JSON decision the scientist parser needs, so the scientist silently produces no
    # valid action. Fall back to a plain join only if the tokenizer defines no chat template.
    prompt = None
    # Qwind3-family models default to THINKING mode: apply_chat_template injects a <think> block and the
    # model spends its whole output budget reasoning inside it, so with a small max_new_tokens the
    # generation ends BEFORE the </think> + JSON answer -> decoded content is empty. Disable thinking so
    # the model emits the JSON directly. enable_thinking is only accepted by templates that support it,
    # so try it first and fall back. (Root cause of the 0-char local-brain output on the full prompt.)
    try:
        if getattr(tok, "chat_template", None):
            try:
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            except TypeError:
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:
        TELEMETRY["last_error"] = f"hf_template:{type(exc).__name__}: {exc}"[:200]
        prompt = None
    if prompt is None:
        prompt = "\n\n".join(m.get("content", "") for m in messages)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=cfg.local_ctx).input_ids.to(model.device)
    # honor the configured max_tokens (was hard-capped at 512 -> a thinking model or a long JSON decision
    # gets truncated before closing the brace -> parse_error/empty). Env-tunable ceiling.
    _max_new = min(int(cfg.max_tokens), int(os.environ.get("LLM_HF_MAX_NEW_TOKENS", "2048")))
    outs: list[str] = []
    for _ in range(max(1, n)):
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=_max_new, do_sample=(temp > 0),
                               temperature=max(temp, 0.01), top_p=0.95, pad_token_id=tok.eos_token_id)
        outs.append(_strip_harmony(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)))
    return outs


def _local_chat(cfg: "LLMConfig", messages: list[dict], n: int, temp: float) -> list[str]:
    """Run n chat completions on the in-process local strategist. Routes to the HF/safetensors
    backend when LLM_LOCAL_HF is set, else the GGUF backend. Harmony-stripped."""
    if cfg.is_hf:
        return _hf_chat(cfg, messages, n, temp)
    llm = _get_local_llm(cfg)
    outs: list[str] = []
    for _ in range(max(1, n)):
        try:
            resp = llm.create_chat_completion(
                messages=messages,
                max_tokens=cfg.max_tokens,
                temperature=temp,
            )
            raw = (resp["choices"][0]["message"].get("content") or "") if resp.get("choices") else ""
            outs.append(_strip_harmony(raw))
        except Exception as exc:
            TELEMETRY["last_error"] = f"local:{type(exc).__name__}: {exc}"[:200]
            outs.append("")
    return outs


def _extract_content(choice) -> str:
    content = choice.message.content or ""
    # Thinking models (Agnes 2.0 Flash) sometimes leave the JSON only in
    # reasoning_content with an empty content field — fall back to it so the
    # downstream JSON extractor can still recover the programs.
    if not content.strip():
        rc = getattr(choice.message, "reasoning_content", None)
        if not rc:
            try:
                rc = choice.message.model_dump().get("reasoning_content")
            except Exception:
                rc = None
        content = rc or ""
    return content


def _create(client, cfg: "LLMConfig", messages: list[dict], n: int, temp: float, use_json: bool):
    kwargs = dict(model=cfg.model, messages=messages, n=n, temperature=temp, max_tokens=cfg.max_tokens,
                  frequency_penalty=cfg.frequency_penalty, presence_penalty=cfg.presence_penalty)
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs)


def chat(messages: list[dict], *, n: int = 1, model: str | None = None, temperature: float | None = None) -> list[str]:
    """Return up to ``n`` completion strings; [] if no client / after retries fail.

    Retries transient API errors with exponential backoff and a per-request
    timeout, and records telemetry, so a flaky provider can no longer silently
    starve the run (the old code swallowed every exception as empty output).
    """
    cfg = resolve_config(model)
    if not cfg.enabled:
        return []
    temp = cfg.temperature if temperature is None else temperature
    TELEMETRY["calls"] += 1
    t0 = time.perf_counter()
    if cfg.is_local:
        # CRITICAL: wrap the local backend like the API path. A local HF/GGUF error (CUDA OOM,
        # chat-template failure, load error) used to PROPAGATE out of chat() -> investigate() ->
        # caught by strata_search which then `break`s the meta instantly (the "tree not growing"
        # symptom: metas finish in ~0s, bestRawps=0, GPU idle). Now it degrades to [] + telemetry,
        # exactly like a failed API call, so the search continues and the error is visible.
        try:
            out = _local_chat(cfg, messages, n, temp)
        except Exception as exc:
            TELEMETRY["hard_fail"] += 1
            TELEMETRY["last_error"] = f"local_chat:{type(exc).__name__}: {exc}"[:200]
            TELEMETRY["wall_s"] = float(TELEMETRY["wall_s"]) + (time.perf_counter() - t0)
            return []
        TELEMETRY["wall_s"] = float(TELEMETRY["wall_s"]) + (time.perf_counter() - t0)
        TELEMETRY["ok" if any(s.strip() for s in out) else "empty"] += 1
        return out
    key = (cfg.base_url or "", cfg.model)
    use_json = _JSON_MODE_OK.get(key, True)          # optimistic until proven unsupported
    try:
        from openai import OpenAI

        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=_TIMEOUT_S, max_retries=0)
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                try:
                    resp = _create(client, cfg, messages, n, temp, use_json)
                except Exception:
                    if not use_json:
                        raise
                    # this endpoint/model rejected response_format — fall back ONCE, same
                    # attempt, no wasted backoff sleep, and remember it for every future call.
                    use_json = False
                    resp = _create(client, cfg, messages, n, temp, use_json)
                _JSON_MODE_OK[key] = use_json
                out = [_extract_content(c) for c in resp.choices]
                TELEMETRY["wall_s"] = float(TELEMETRY["wall_s"]) + (time.perf_counter() - t0)
                if any(s.strip() for s in out):
                    TELEMETRY["ok"] += 1
                else:
                    TELEMETRY["empty"] += 1
                return out
            except Exception as exc:                       # transient — back off and retry
                last_exc = exc
                TELEMETRY["retries"] += 1
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_S * (2 ** attempt))
        TELEMETRY["hard_fail"] += 1
        TELEMETRY["last_error"] = f"{type(last_exc).__name__}: {last_exc}"[:200]
        TELEMETRY["wall_s"] = float(TELEMETRY["wall_s"]) + (time.perf_counter() - t0)
        return []
    except Exception as exc:                               # import/client-build failure
        TELEMETRY["hard_fail"] += 1
        TELEMETRY["last_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return []


def chat_many(
    batches: list[list[dict]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_workers: int = 6,
) -> list[str]:
    """Fire many chat requests CONCURRENTLY and return one completion per batch.

    Agnes is I/O-bound (~150 s/call, mostly waiting on the API), so threading
    collapses K sequential calls into ~one call's wall time. This is what lets us
    call the LLM far more often — for cold-start design and per-node mutation —
    without the run time scaling linearly. Failed/empty calls yield "".
    """
    cfg = resolve_config(model)
    if not cfg.enabled or not batches:
        return ["" for _ in batches]
    temp = cfg.temperature if temperature is None else temperature
    if cfg.is_local:
        # One shared in-process model: run sequentially (llama.cpp is not thread-safe
        # for concurrent create_chat_completion on one context).
        out: list[str] = []
        for msgs in batches:
            TELEMETRY["calls"] += 1
            r = _local_chat(cfg, msgs, 1, temp)
            txt = r[0] if r else ""
            TELEMETRY["ok" if txt.strip() else "empty"] += 1
            out.append(txt)
        return out
    from concurrent.futures import ThreadPoolExecutor
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=_TIMEOUT_S, max_retries=0)
    key = (cfg.base_url or "", cfg.model)

    def _one(msgs: list[dict]) -> str:
        TELEMETRY["calls"] += 1
        use_json = _JSON_MODE_OK.get(key, True)
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                try:
                    resp = _create(client, cfg, msgs, 1, temp, use_json)
                except Exception:
                    if not use_json:
                        raise
                    use_json = False
                    resp = _create(client, cfg, msgs, 1, temp, use_json)
                _JSON_MODE_OK[key] = use_json
                txt = _extract_content(resp.choices[0]) if resp.choices else ""
                TELEMETRY["ok" if txt.strip() else "empty"] += 1
                return txt
            except Exception as exc:
                last_exc = exc
                TELEMETRY["retries"] += 1
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_S * (2 ** attempt))
        TELEMETRY["hard_fail"] += 1
        TELEMETRY["last_error"] = f"{type(last_exc).__name__}: {last_exc}"[:200]
        return ""

    with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as ex:
        return list(ex.map(_one, batches))
