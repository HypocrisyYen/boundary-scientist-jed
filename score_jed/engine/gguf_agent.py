"""Wire the real GGUF target models (GPT-OSS 20B / Gemma 4) into the sandbox.

The competition replays candidates against ``gpt_oss`` and ``gemma`` served as
**GGUF via llama.cpp** (data.txt). The SDK already ships the adapter we need:
``LlamaCppChatTemplateBackend.from_model_path`` builds a llama.cpp backend from a
``.gguf`` path, and both ``GPTOSSAgent``/``Gemma4Agent`` accept a prebuilt backend
(they auto-switch to the default request builder for a llama.cpp backend). So we
do NOT reimplement tool-call parsing — we assemble the SDK pieces.

This module:
  * locates ``.gguf`` weights (mirroring the notebook's find_ggufs/pick_* logic);
  * builds a llama.cpp backend + the matching SDK agent;
  * returns an ``agent_factory`` for ``TargetSpec``.

Everything imports cleanly even without ``llama_cpp`` or weights present — the
hard dependency is only touched when a factory is actually invoked, and failures
raise a clear, actionable error instead of silently degrading to the toy agent
(which would produce misleading EV).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

# Agent kinds that map to a GGUF-backed target.
GGUF_KINDS = ("gpt_oss", "gemma", "gemma_4")

# Default places to hunt for weights: env override, a local models/ dir, and the
# Kaggle mounts (so the same code runs unchanged in the competition container).
_DEFAULT_SEARCH_DIRS = (
    "models",
    "score_jed/models",
    "/kaggle/input",
    "/kaggle/working",
    "/mnt/data",
)

_ENV_PATH = {
    "gpt_oss": "GPT_OSS_MODEL_PATH",
    "gemma": "GEMMA_MODEL_PATH",
    "gemma_4": "GEMMA4_MODEL_PATH",
}


def find_ggufs(search_dirs: tuple[str, ...] | None = None) -> list[Path]:
    dirs = search_dirs
    if dirs is None:
        env_dirs = os.environ.get("GGUF_SEARCH_DIRS", "")
        extra = tuple(d for d in env_dirs.split(os.pathsep) if d.strip())
        dirs = extra + _DEFAULT_SEARCH_DIRS
    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        root = Path(d)
        if not root.exists():
            continue
        try:
            for f in root.glob("**/*.gguf"):
                r = str(f.resolve())
                if r not in seen:
                    seen.add(r)
                    out.append(f)
        except Exception:
            continue
    return out


def _pick_gpt(paths: list[Path]) -> Path | None:
    scored = []
    for p in paths:
        n = p.name.lower()
        s = 5 * ("gpt" in n) + 5 * ("oss" in n) + ("q4" in n) + (p.suffix.lower() == ".gguf")
        if s:
            scored.append((s, len(str(p)), p))
    return sorted(scored, key=lambda x: (-x[0], x[1]))[0][2] if scored else None


def _pick_gemma(paths: list[Path]) -> Path | None:
    scored = []
    for p in paths:
        n = p.name.lower()
        if "gemma" not in n or any(x in n for x in ("mmproj", "projector", "clip")):
            continue
        s = 10 + 3 * ("26b" in n) + ("q4" in n) + ("it" in n)
        scored.append((s, len(str(p)), p))
    return sorted(scored, key=lambda x: (-x[0], x[1]))[0][2] if scored else None


def resolve_model_path(kind: str, explicit: str | None = None) -> str:
    """explicit arg > env var > auto-pick from search dirs. Raises if unfound."""
    if explicit:
        p = Path(explicit)
        if not (p.is_file() and p.suffix.lower() == ".gguf"):
            raise FileNotFoundError(f"{kind}: explicit gguf path is not a .gguf file: {explicit}")
        return str(p)
    env_key = _ENV_PATH.get(kind, "")
    env_val = os.environ.get(env_key) if env_key else None
    if env_val:
        if not Path(env_val).is_file():
            raise FileNotFoundError(f"{kind}: {env_key}={env_val} does not exist")
        return env_val
    ggufs = find_ggufs()
    picker = _pick_gpt if kind == "gpt_oss" else _pick_gemma
    chosen = picker(ggufs)
    if chosen is None:
        raise FileNotFoundError(
            f"{kind}: no .gguf found. Set {env_key} or GGUF_SEARCH_DIRS, or pass gguf_path. "
            f"(searched {len(ggufs)} gguf files)"
        )
    return str(chosen)


import re as _re

_HARMONY_TAG = _re.compile(r"<\|[^|>]*\|>")


def _sanitize_harmony_content(text: str) -> str:
    """Strip gpt-oss harmony control tags from a replayed assistant content string.

    gpt-oss emits ``<|channel|>analysis<|message|>...<|end|>...<|channel|>final
    <|message|>ANSWER``. When such raw text is fed back as an assistant ``content``
    on the next turn, llama.cpp's chat handler raises ValueError. We keep only the
    text after the LAST ``<|message|>`` (the final channel) and drop any remaining
    control tags. The attacker's user messages are untouched — only the model's own
    replayed turns are cleaned — so mined candidates are unaffected.
    """
    if "<|" not in text:
        return text
    if "<|message|>" in text:
        text = text.rsplit("<|message|>", 1)[1]
    for stop in ("<|end|>", "<|return|>", "<|start|>", "<|call|>"):
        if stop in text:
            text = text.split(stop, 1)[0]
    return _HARMONY_TAG.sub("", text).strip()


def _wrap_sanitizing_chat(llm: Any) -> None:
    """Patch llm.create_chat_completion to sanitize harmony tags in the history."""
    orig = llm.create_chat_completion

    def _patched(*args, **kwargs):
        msgs = kwargs.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
                    m["content"] = _sanitize_harmony_content(m["content"])
        return orig(*args, **kwargs)

    llm.create_chat_completion = _patched


def build_gguf_backend(
    kind: str,
    model_path: str,
    *,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    max_new_tokens: int = 256,
    verbose: bool = False,
    llama_kwargs: dict[str, Any] | None = None,
):
    """Build an SDK llama.cpp backend for the given model kind."""
    from aicomp_sdk.agents.hf_chat_template.backends.llama_cpp import LlamaCppChatTemplateBackend

    if kind == "gpt_oss":
        from aicomp_sdk.agents.gpt_oss_agent import build_gpt_oss_backend_config as cfg_builder
    else:
        from aicomp_sdk.agents.gemma4_agent import build_gemma4_backend_config as cfg_builder

    config = cfg_builder(model_path=model_path, max_new_tokens=max_new_tokens)
    backend = LlamaCppChatTemplateBackend.from_model_path(
        model_path=model_path,
        config=config,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
        supports_tools=True,
        llama_kwargs=llama_kwargs,
    )
    # gpt-oss harmony re-ingestion guard (harmless for other models: no <| tags)
    try:
        _wrap_sanitizing_chat(backend.llm)
    except Exception:
        pass
    return backend


def _wrap_agent(kind: str, backend):
    if kind == "gpt_oss":
        from aicomp_sdk.agents.gpt_oss_agent import GPTOSSAgent

        return GPTOSSAgent(backend)
    from aicomp_sdk.agents.gemma4_agent import Gemma4Agent

    return Gemma4Agent(backend)


# Process-wide registry of loaded backends, so every sandbox / guardrail run in
# this process reuses ONE loaded local model instead of loading a second 12GB copy
# (which OOMs a 23GB GPU). Keyed by the load parameters that would change weights.
_SHARED_BACKENDS: dict[tuple, Any] = {}


_GEN_STATS_CAP = 500


def _install_gen_stats(backend) -> None:
    """Wrap backend.llm.create_chat_completion to record per-generation wall time + token
    counts on backend._gen_stats (capped ring buffer) — the same measurement fidelity_bench.py
    proved out, installed once here so every consumer (fidelity_bench, the live discovery loop,
    a future submission-time diagnostic) gets real token-level data for free. Idempotent: skips
    if this backend was already wrapped (checked via a marker attribute, not by identity of the
    wrapped function, since re-wrapping a wrap would double-count)."""
    llm = getattr(backend, "llm", None)
    if llm is None or getattr(llm, "_sj_stats_wrapped", False):
        return
    orig = llm.create_chat_completion
    stats: list[dict] = []
    backend._gen_stats = stats
    # _gen_total is a MONOTONIC count of every generation ever recorded (never shrinks, even
    # when `stats` itself is capped/front-evicted below). A consumer that only saw an absolute
    # length before the call (e.g. AttackSandbox.step()) can't safely use that as a slice start
    # once eviction has shifted every index — but a DELTA against this monotonic counter
    # (`_gen_total_after - _gen_total_before`) survives eviction, so slice from the end instead:
    # `stats[-delta:]` — see AttackSandbox.step().
    backend._gen_total = 0

    def _captured(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig(*args, **kwargs)
        # Prod-cost simulation (isolated: default 0, only the blind-explore run sets it).
        # The real evaluator relays every next_action cross-container and re-prefills the full
        # history with NO KV reuse (measured: prod gen2==floor~3s though it decodes ~5 tok).
        # A flat per-generation floor makes LOCAL raw/s prod-faithful, so the scientist explores
        # the SAME cost landscape it will face on submission (multi-hop pays the floor per hop;
        # the single wastes a whole floor on its forced final). SJ_GEN_FLOOR_S=0 -> no-op.
        _floor = float(os.environ.get("SJ_GEN_FLOOR_S", "0") or "0")
        if _floor > 0:
            time.sleep(_floor)
        dt = time.perf_counter() - t0
        usage = (out.get("usage") or {}) if isinstance(out, dict) else {}
        stats.append({
            "wall_s": round(dt, 4),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        })
        backend._gen_total += 1
        if len(stats) > _GEN_STATS_CAP:
            del stats[: len(stats) - _GEN_STATS_CAP]
        return out

    llm.create_chat_completion = _captured
    llm._sj_stats_wrapped = True


def get_shared_backend(
    kind: str,
    model_path: str,
    *,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    max_new_tokens: int = 256,
    verbose: bool = False,
    llama_kwargs: dict[str, Any] | None = None,
):
    """Return the one process-wide backend for these load params (build on first use)."""
    key = (kind, model_path, n_ctx, n_gpu_layers, max_new_tokens)
    backend = _SHARED_BACKENDS.get(key)
    if backend is None:
        backend = build_gguf_backend(
            kind, model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
            max_new_tokens=max_new_tokens, verbose=verbose, llama_kwargs=llama_kwargs,
        )
        _SHARED_BACKENDS[key] = backend
    _install_gen_stats(backend)
    return backend


def clear_shared_backends() -> None:
    """Close and drop all shared backends and FREE VRAM — used to swap the active
    model in/out of the GPU for two-model alternation (both GGUFs stay warm in the OS
    page cache / RAM, so re-loading the other model to VRAM is fast)."""
    import gc
    for b in list(_SHARED_BACKENDS.values()):
        for closer in ("close", "_close", "__del__"):
            try:
                fn = getattr(b, closer, None)
                if fn:
                    fn()
                    break
            except Exception:
                pass
    _SHARED_BACKENDS.clear()
    gc.collect()   # force llama_cpp to release the GPU allocation before the next load


def build_gguf_agent_factory(
    kind: str,
    *,
    gguf_path: str | None = None,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    max_new_tokens: int = 256,
    share_backend: bool = True,
    verbose: bool = False,
    llama_kwargs: dict[str, Any] | None = None,
) -> Callable[[], Any]:
    """Return a factory that yields a GGUF-backed SDK agent.

    The oracle creates a fresh env (and calls the factory) per replay. Loading a
    20B GGUF per replay would be ruinous, so with ``share_backend=True`` (default)
    the weights are loaded ONCE and shared **process-wide** — every sandbox and
    every guardrail in this process reuses the same local model (no VRAM double
    load). ``agent.reset_state`` + a fresh env per replay keep trials independent.
    """
    if kind not in GGUF_KINDS:
        raise ValueError(f"unknown gguf kind: {kind!r} (expected one of {GGUF_KINDS})")

    resolved_path = resolve_model_path(kind, gguf_path)

    try:
        import llama_cpp  # noqa: F401
    except Exception as err:  # pragma: no cover - depends on host
        raise RuntimeError(
            "llama_cpp is required for GGUF targets. Install `llama-cpp-python` "
            "(GPU build for T4 parity), then retry."
        ) from err

    def _factory():
        if share_backend:
            backend = get_shared_backend(
                kind, resolved_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                max_new_tokens=max_new_tokens, verbose=verbose, llama_kwargs=llama_kwargs,
            )
        else:
            backend = build_gguf_backend(
                kind, resolved_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                max_new_tokens=max_new_tokens, verbose=verbose, llama_kwargs=llama_kwargs,
            )
            _install_gen_stats(backend)
        agent = _wrap_agent(kind, backend)
        agent._sj_backend = backend   # side-channel handle for token-level diagnostics (P5);
        return agent                  # a plain dynamic attribute, doesn't touch the SDK's classes

    _factory.model_path = resolved_path  # type: ignore[attr-defined]
    _factory.kind = kind  # type: ignore[attr-defined]
    return _factory
