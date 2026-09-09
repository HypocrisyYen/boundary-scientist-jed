"""Long-running Research Lab — 12h+ continuous high-EV attack discovery.

Design goals (why this is a chunked loop, not one 12h Searcher.run):
  * ROBUST to Colab runtime resets — every chunk distills + persists the prior and
    an append-only archive; on restart, --resume reloads and continues;
  * BOUNDED memory over 12h — a fresh Searcher per chunk drops accumulated nodes /
    snapshots; the loaded GGUF model + sandbox are reused (ONE model load for 12h);
  * CONTINUOUS optimization — each chunk seeds from the growing prior (proven arms)
    plus the builtin bank, so it exploits winners while exploring;
  * ANTI-STAGNATION — when a chunk fails to raise the best predicted raw, the next
    chunk injects MORE cold-start LLM designs (fresh invention) and widens search;
  * a persistent RECORD — research_prior.json (top merged arms) + research_archive.jsonl
    (every eligible method found), the corpus the 9000s submission warm-starts from.

Run (Colab, GPU + Agnes strategist):
  OPENAI_API_KEY=... OPENAI_BASE_URL=https://apihub.agnes-ai.com/v1 LLM_MODEL=agnes-2.0-flash \
  python3 -m pipeline.research_lab --hours 12 --agent gpt_oss \
    --gguf /content/models/gpt-oss-20b-MXFP4.gguf --guardrail optimal --llm 2 --resume
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine import generator, reporting, scoring  # noqa: E402
from engine.explorer import LLMExplorer  # noqa: E402
from engine.prior import CompactPrior, append_archive, distill, merge_priors  # noqa: E402
from engine.program import builtin_programs  # noqa: E402
from engine.sandbox import AttackSandbox, TargetSpec  # noqa: E402
from engine.uct import Searcher  # noqa: E402

_PRED_FAMILIES = ("EXFILTRATION", "UNTRUSTED_TO_ACTION", "DESTRUCTIVE_WRITE", "CONFUSED_DEPUTY")


@dataclass
class LabConfig:
    total_budget_s: float = 43200.0     # 12 hours
    chunk_s: float = 1800.0             # 30-minute chunks
    agent: str = "gpt_oss"
    guardrail: str = "optimal"
    seed: int = 123
    gguf_path: str | None = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    max_new_tokens: int = 192
    trials: int = 1
    beam: int = 8
    mutations: int = 6
    llm: int = 2
    llm_every: int = 3
    llm_top_m: int = 3
    cold_start_base: int = 6
    cold_start_stagnation_step: int = 4   # extra cold-start designs per stagnant chunk
    llm_wall_cap: float = 0.2
    prior_top_k: int = 200
    replay_budget_s: float = 9000.0
    # interactive LLM explorer (closed-loop, novelty-seeking multi-step discovery)
    explore_episodes: int = 0           # episodes per chunk (0 = off; research runs set >0)
    explore_max_steps: int = 6          # max turns (messages) per episode
    resume: bool = False
    results_dir: Path = _ROOT / "results"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _arm_families(arm) -> set[str]:
    """Predicate families this genome tripped (empty for pre-diversity stores)."""
    return getattr(arm, "predicate_families", set()) or set()


def _family_breakdown(store) -> dict[str, int]:
    """How many distinct eligible METHODS trip each predicate family — the metric
    that shows research WIDENING the attack surface, not deepening one pattern."""
    counts = {f: 0 for f in _PRED_FAMILIES}
    for a in store.arms.values():
        if a.eligible:
            for f in _arm_families(a):
                if f in counts:
                    counts[f] += 1
    return counts


def run(cfg: LabConfig) -> None:
    rd = cfg.results_dir
    rd.mkdir(parents=True, exist_ok=True)
    prior_path = rd / "research_prior.json"
    archive_path = rd / "research_archive.jsonl"
    bank = generator.load_strategy_bank(_ROOT / "memory" / "strategy_bank.jsonl")

    # one model load, reused across every chunk for the whole run
    spec = TargetSpec(agent=cfg.agent, guardrail=cfg.guardrail, seed=cfg.seed, gguf_path=cfg.gguf_path,
                      n_ctx=cfg.n_ctx, n_gpu_layers=cfg.n_gpu_layers, max_new_tokens=cfg.max_new_tokens)
    sb = AttackSandbox(spec)

    # Persistent posterior store — accumulates raw samples across ALL chunks so
    # success_p tightens and cells accumulate (breaks the fresh-per-chunk plateau).
    from engine.posterior import PosteriorStore
    store_path = rd / "posterior_store.json"
    store = PosteriorStore()
    target = f"{cfg.agent}+{cfg.guardrail}"
    if cfg.resume and store_path.is_file():
        try:
            store = PosteriorStore.load(store_path)
            print(f"[lab] resume: posterior store {len(store.arms)} genomes loaded", flush=True)
        except Exception as exc:
            print(f"[lab] resume failed: {exc}", flush=True)

    t0 = time.perf_counter()
    best_raw = store.predicted_corpus_raw(cfg.prior_top_k)
    stagnation = 0
    chunk = 0
    print(f"[lab] START target={target} total={cfg.total_budget_s/3600:.1f}h "
          f"chunk={cfg.chunk_s/60:.0f}m resume_raw={best_raw:.0f} genomes={len(store.arms)}", flush=True)

    while (time.perf_counter() - t0) < cfg.total_budget_s:
        chunk += 1
        remaining = cfg.total_budget_s - (time.perf_counter() - t0)
        chunk_budget = min(cfg.chunk_s, remaining)
        if chunk_budget < 5:  # skip only a tiny final tail
            break
        # ADAPTIVE LLM budget: throttle the (slow) strategist by its MEASURED
        # downstream productivity vs the fast operators, instead of blindly pouring
        # more LLM cold-start in on stagnation.
        ops = store.operator_stats()
        _llm_g = sum(ops.get(s, {}).get("genomes", 0) for s in ("llm", "llm_cold"))
        _llm_e = sum(ops.get(s, {}).get("eligible", 0) for s in ("llm", "llm_cold"))
        _oth_g = sum(v["genomes"] for k, v in ops.items() if k not in ("llm", "llm_cold"))
        _oth_e = sum(v["eligible"] for k, v in ops.items() if k not in ("llm", "llm_cold"))
        llm_prod = (_llm_e / _llm_g) if _llm_g >= 12 else 1.0          # eligible rate (assume good early)
        oth_prod = (_oth_e / _oth_g) if _oth_g > 0 else 0.1
        llm_factor = min(1.0, (llm_prod + 0.01) / (oth_prod + 0.01))    # 1.0 if LLM keeps up, <1 if worse
        cur_llm_every = max(cfg.llm_every, round(cfg.llm_every / max(0.2, llm_factor)))
        cold = cfg.cold_start_base + round(stagnation * cfg.cold_start_stagnation_step * llm_factor)
        if _llm_g >= 12 and llm_factor < 0.9:
            print(f"    [c{chunk} adaptive] LLM productivity {llm_prod:.2f} vs {oth_prod:.2f} "
                  f"-> factor={llm_factor:.2f}, llm_every={cur_llm_every}, cold={cold}", flush=True)
        # INTERACTIVE EXPLORER phase: before search, let the strategist LLM drive the
        # live agent turn-by-turn hunting NOVEL multi-step paths (new predicate family
        # / new tool-trace cell), rewarded for diversity not EV. Its discoveries become
        # seeds this chunk's Searcher replays + accumulates (source=llm_explore), so the
        # method set widens instead of the corpus just deepening one exfil pattern.
        explore_seeds: list = []
        if cfg.explore_episodes > 0:
            seen_cells = set(store.all_cells())
            seen_families = {f for a in store.arms.values() if a.eligible for f in _arm_families(a)}
            explorer = LLMExplorer(sb, model=None, strategy_bank=bank, max_steps=cfg.explore_max_steps,
                                   on_event=lambda m: print(f"    [c{chunk} {m}", flush=True))
            try:
                explore_seeds, ex_meta = explorer.run_budget(episodes=cfg.explore_episodes,
                                                             seen_cells=seen_cells, seen_families=seen_families)
                print(f"    [c{chunk} explore] {ex_meta['episodes']} episodes, {ex_meta['llm_calls']} llm calls, "
                      f"{ex_meta['steps']} steps -> {len(explore_seeds)} novel programs, "
                      f"families={ex_meta['families']}, new_cells={ex_meta['new_cells']}", flush=True)
            except Exception as exc:
                print(f"    [c{chunk} explore] error: {type(exc).__name__}: {exc}", flush=True)
                sb.clear_cache()

        # seed from the accumulated store's top arms (proven winners keep gaining samples)
        seeds = store.top_programs(cfg.prior_top_k) + explore_seeds + builtin_programs()

        def _ckpt(searcher) -> None:
            try:
                # accumulate current-chunk evidence into the store, then persist the prior
                store.update_from_nodes(searcher.current_result()["nodes"])
                store.save(store_path)
                store.to_prior(target=target, top_k=cfg.prior_top_k, strategies=bank).save(prior_path)
                st = searcher.stats
                print(f"    [c{chunk} ckpt @ {_now()}] genomes={len(store.arms)} pred_raw~{store.predicted_corpus_raw(cfg.prior_top_k):.0f} "
                      f"cells={len(store.all_cells())} | {st.programs_seen} progs, {st.positive_arms} arms, "
                      f"{st.cold_validated_arms} cold, goexp={st.go_explore_expansions}, "
                      f"llm={st.llm_calls}c/{st.llm_programs}p {st.llm_wall_s:.0f}s", flush=True)
            except Exception:
                pass

        searcher = Searcher(
            sandbox=sb, seeds=seeds, time_budget_s=chunk_budget, trials_per_program=cfg.trials,
            beam=cfg.beam, mutations_per_expand=cfg.mutations, llm_per_expand=cfg.llm,
            llm_every=cur_llm_every, llm_top_m=cfg.llm_top_m, llm_cold_start=cold,
            llm_wall_cap=cfg.llm_wall_cap, go_explore=True, strategy_bank=bank, llm_model=None,
            on_event=lambda e: _chunk_log(e, chunk), checkpoint_cb=_ckpt, checkpoint_every_s=120.0,
        )
        try:
            result = searcher.run()
        except Exception as exc:
            print(f"[lab] chunk {chunk} error: {type(exc).__name__}: {exc}", flush=True)
            sb.clear_cache()
            continue

        # ACCUMULATE this chunk's raw samples into the persistent posterior
        n_new = store.update_from_nodes(result["nodes"])
        cumulative = store.to_prior(target=target, top_k=cfg.prior_top_k, strategies=bank)
        cumulative.save(prior_path)
        store.save(store_path)
        append_archive(result, archive_path, chunk=chunk)

        raw = store.predicted_corpus_raw(cfg.prior_top_k)
        if raw <= best_raw + 1e-6:
            stagnation += 1
        else:
            best_raw = raw
            stagnation = 0

        st = result["stats"]
        elapsed_h = (time.perf_counter() - t0) / 3600.0
        # submission-level score (what actually matters): pack the accumulated posterior
        sub_items, sub_raw = scoring.pack_portfolio(store.eligible_evs(), time_budget_s=cfg.replay_budget_s,
                                                    max_candidates=2000, use_p95=True)
        top = sorted((a.to_ev() for a in store.arms.values() if a.eligible),
                     key=lambda e: e.expected_severity_raw, reverse=True)[:4]
        print(f"[lab] chunk {chunk} @ {_now()} elapsed={elapsed_h:.2f}h | "
              f"genomes={len(store.arms)} (+{n_new} new) corpus_raw={raw:.0f} cells={len(store.all_cells())} "
              f"stagnation={stagnation} | SUBMISSION pred_raw={sub_raw:.0f} (~{min(1000.0, sub_raw/200000.0*1000.0):.1f} norm) "
              f"{len(sub_items)} cands | chunk: {st.programs_seen} progs, {st.positive_arms} arms, "
              f"{st.cold_validated_arms} cold, goexp={st.go_explore_expansions}, "
              f"llm={st.llm_calls}c/{st.llm_programs}p/{st.llm_parse_fail}fail {st.llm_wall_s:.0f}s", flush=True)
        print("       top: " + " | ".join(
            f"{e.program_name[:20]}(sev={e.mean_severity_raw:.0f},p={e.success_p:.2f},n={e.n})" for e in top), flush=True)
        fam = _family_breakdown(store)
        print("       diversity (methods per predicate): " + ", ".join(
            f"{k.split('_')[0]}={v}" for k, v in fam.items()), flush=True)

        sb.clear_cache()  # bound snapshot memory between chunks

    # final: write the standard submission pack + prior from the accumulated posterior
    cumulative = store.to_prior(target=target, top_k=cfg.prior_top_k, strategies=bank)
    items, predicted = scoring.pack_portfolio(store.eligible_evs(), time_budget_s=cfg.replay_budget_s,
                                              max_candidates=2000, use_p95=True)
    reporting.write_submission_pack(items, predicted, rd / "submission_pack.json", time_budget_s=cfg.replay_budget_s)
    cumulative.save(rd / "compact_prior.json")  # the name the submission adapter loads by default
    store.save(store_path)
    print(f"[lab] DONE {chunk} chunks, {(time.perf_counter()-t0)/3600:.2f}h | genomes={len(store.arms)} "
          f"corpus_raw={store.predicted_corpus_raw(cfg.prior_top_k):.0f} SUBMISSION_raw={predicted:.0f} "
          f"(~{min(1000.0, predicted/200000.0*1000.0):.1f} norm) | prior->{prior_path}", flush=True)
    ops = store.operator_stats()
    print("[lab] operator productivity (eligible/genomes, ΣEV): " + " | ".join(
        f"{k}={v['eligible']:.0f}/{v['genomes']:.0f} ev={v['exp_raw']:.0f}"
        for k, v in sorted(ops.items(), key=lambda kv: kv[1]['exp_raw'], reverse=True)), flush=True)
    fam = _family_breakdown(store)
    print("[lab] method diversity (distinct eligible methods per predicate): " + ", ".join(
        f"{k}={v}" for k, v in fam.items()), flush=True)


def _prior_to_arms(prior: CompactPrior) -> list:
    """Reconstruct minimal EVArmStats from prior arms for portfolio packing."""
    out = []
    for a in prior.arms:
        arm = scoring.EVArmStats(
            program_sha="", program_name=a.name, family=a.family, mechanism=a.mechanism,
            n=1, n_positive=1, mean_severity_raw=a.severity_raw, mean_replay_s=max(a.mean_replay_s, 1e-3),
            messages=tuple(a.exact_messages), deploy_cell=a.deploy_cell, success_p=a.success_p,
            success_p_lb=a.success_p, expected_severity_raw=a.success_p * a.severity_raw,
            risk_adjusted_raw=a.success_p * a.severity_raw,
        )
        out.append(arm)
    return out


def _chunk_log(e: dict, chunk: int) -> None:
    t = e.get("t")
    if t == "step":
        print(f"    [c{chunk} step {e['step']}] arms={e['arms']} positive={e['positive']} cells={e['cells']}", flush=True)
    elif t == "cold_done":
        print(f"    [c{chunk} cold] designed {e['produced']} programs {e.get('names')}", flush=True)
    elif t == "go_explore":
        print(f"    [c{chunk} go-explore] {e['continuations']} continuations from {e['from']} scored states", flush=True)
    elif t == "llm_call":
        print(f"    [c{chunk} llm] calling critic+designer on {e.get('on')} (~2x150s)...", flush=True)
    elif t == "llm_done":
        print(f"    [c{chunk} designer] +{e['produced']} programs {e.get('names')}", flush=True)
    elif t == "llm_diagnosis":
        print(f"    [c{chunk} critic] {e['text'][:160]}...", flush=True)


def _parse(argv=None) -> LabConfig:
    ap = argparse.ArgumentParser(description="score_jed 12h Research Lab")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--chunk-min", type=float, default=30.0)
    ap.add_argument("--agent", default="gpt_oss")
    ap.add_argument("--guardrail", default="optimal", choices=["optimal", "allow", "rules"])
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--llm", type=int, default=2)
    ap.add_argument("--llm-every", type=int, default=3)
    ap.add_argument("--cold-start", type=int, default=6)
    ap.add_argument("--llm-wall-cap", type=float, default=0.2)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    return LabConfig(
        total_budget_s=a.hours * 3600.0, chunk_s=a.chunk_min * 60.0, agent=a.agent, guardrail=a.guardrail,
        gguf_path=a.gguf, n_ctx=a.n_ctx, n_gpu_layers=a.n_gpu_layers, llm=a.llm, llm_every=a.llm_every,
        cold_start_base=a.cold_start, llm_wall_cap=a.llm_wall_cap, resume=a.resume,
    )


if __name__ == "__main__":
    try:
        run(_parse())
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\n[lab] target not ready: {exc}", file=sys.stderr)
        raise SystemExit(2)
