"""score_jed orchestrator — the offline high-EV attack miner.

Mirrors score_golf/pipeline/run_search.py. It builds a target sandbox, seeds the
search with the tactic bank, runs the EV tree search for a budget, then writes:

  * results/arm_ev.csv           — ranked EV table (audit / analysis)
  * results/archive.json         — every program + its arm + trace feedback
  * results/submission_pack.json — budget-packed, ordered candidate messages
                                   the submission attack.py should replay

Run:
  cd score_jed
  python -m pipeline.run_search --budget 120 --guardrail allow
  python -m pipeline.run_search --budget 3000 --agent gpt_oss --guardrail optimal --llm 3
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# make ``engine`` importable when run as a script from anywhere
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine import generator, reporting, scoring  # noqa: E402
from engine.program import builtin_programs  # noqa: E402
from engine.sandbox import AttackSandbox, TargetSpec  # noqa: E402
from engine.uct import Searcher  # noqa: E402


@dataclass
class RunConfig:
    budget_s: float = 120.0
    agent: str = "deterministic"
    guardrail: str = "optimal"
    seed: int = 123
    trials: int = 1   # screen once; VOI cold-resampling adds trials where they matter
    beam: int = 6
    mutations: int = 6
    llm: int = 0
    llm_every: int = 4
    llm_top_m: int = 3
    cold_start: int = 0
    go_explore: bool = True
    prior_top_k: int = 40
    resume: bool = False              # load results/compact_prior.json as extra seeds + merge
    seed_prior: str | None = None     # load an external prior (e.g. V28) as extra seeds
    llm_model: str | None = None
    replay_budget_s: float = 9000.0
    results_dir: Path = _ROOT / "results"
    verbose: bool = True
    # GGUF target options (used when agent in {gpt_oss, gemma, gemma_4})
    gguf_path: str | None = None
    n_ctx: int = 8192
    n_gpu_layers: int = -1
    max_new_tokens: int = 256
    use_prefix_cache: bool = True


def run(cfg: RunConfig) -> dict:
    spec = TargetSpec(
        agent=cfg.agent,
        guardrail=cfg.guardrail,
        seed=cfg.seed,
        gguf_path=cfg.gguf_path,
        n_ctx=cfg.n_ctx,
        n_gpu_layers=cfg.n_gpu_layers,
        max_new_tokens=cfg.max_new_tokens,
        use_prefix_cache=cfg.use_prefix_cache,
    )
    sb = AttackSandbox(spec)
    seeds = builtin_programs()
    bank = generator.load_strategy_bank(_ROOT / "memory" / "strategy_bank.jsonl")

    # Checkpoint/resume + external-prior warm start: prepend prior programs as seeds.
    from engine.prior import CompactPrior, load_external, merge_priors
    resume_prior: CompactPrior | None = None
    prior_path = cfg.results_dir / "compact_prior.json"
    if cfg.resume and prior_path.is_file():
        try:
            resume_prior = CompactPrior.load(prior_path)
            seeds = resume_prior.programs() + seeds
            print(f"[score_jed] resume: {len(resume_prior.arms)} prior arms loaded as seeds", flush=True)
        except Exception as exc:
            print(f"[score_jed] resume failed: {exc}", flush=True)
    ext_prior: CompactPrior | None = None
    if cfg.seed_prior:
        try:
            ext_prior = load_external(cfg.seed_prior)
            seeds = ext_prior.programs() + seeds
            print(f"[score_jed] seeded {len(ext_prior.arms)} arms from external prior {cfg.seed_prior}", flush=True)
        except Exception as exc:
            print(f"[score_jed] external prior load failed: {exc}", flush=True)

    def _log(e: dict) -> None:
        if not cfg.verbose:
            return
        t = e.get("t")
        if t == "step":
            print(f"  step {e['step']:>4}  arms={e['arms']:>4}  positive={e['positive']:>3}  cells={e['cells']:>4}", flush=True)
        elif t == "llm_call":
            print(f"    [llm] step {e['step']}: calling strategist on '{e['on']}' (~150s)...", flush=True)
        elif t == "llm_done":
            print(f"    [llm] step {e['step']}: strategist produced {e['produced']} programs {e.get('names')}", flush=True)
        elif t == "llm_diagnosis":
            print(f"    [critic] step {e['step']}: {e['text'][:200]}...", flush=True)
        elif t == "llm_err":
            print(f"    [llm] step {e['step']}: strategist error: {e['err']}", flush=True)
        elif t == "go_explore":
            print(f"    [go-explore] step {e['step']}: {e['continuations']} continuations from {e['from']} scored states", flush=True)
        elif t == "cold_done":
            print(f"    [cold-start] designed {e['produced']} programs {e.get('names')}", flush=True)

    rd = cfg.results_dir

    def _checkpoint(searcher) -> None:
        # persist a distilled prior mid-run so progress survives a runtime/tunnel drop
        try:
            from engine.prior import distill as _distill
            r = searcher.current_result()
            _distill(r, target=f"{cfg.agent}+{cfg.guardrail}", top_k=cfg.prior_top_k,
                     strategies=bank).save(rd / "compact_prior.json")
            st = r["stats"]
            print(f"    [checkpoint] {st.programs_seen} progs, {st.positive_arms} arms, "
                  f"{st.cells} cells, {st.cold_validated_arms} cold-valid -> prior saved", flush=True)
        except Exception:
            pass

    searcher = Searcher(
        sandbox=sb,
        seeds=seeds,
        time_budget_s=cfg.budget_s,
        trials_per_program=cfg.trials,
        beam=cfg.beam,
        mutations_per_expand=cfg.mutations,
        llm_per_expand=cfg.llm,
        llm_every=cfg.llm_every,
        llm_top_m=cfg.llm_top_m,
        llm_cold_start=cfg.cold_start,
        go_explore=cfg.go_explore,
        strategy_bank=bank,
        llm_model=cfg.llm_model,
        on_event=_log,
        checkpoint_cb=_checkpoint,
        checkpoint_every_s=120.0,
    )
    print(f"[score_jed] mining target={cfg.agent}+{cfg.guardrail} budget={cfg.budget_s}s llm={cfg.llm}", flush=True)
    t0 = time.perf_counter()
    result = searcher.run()
    arms = result["arms"]

    items, predicted_raw = scoring.pack_portfolio(arms, time_budget_s=cfg.replay_budget_s)

    reporting.write_arm_csv(arms, rd / "arm_ev.csv")
    reporting.write_archive(result, rd / "archive.json")
    reporting.write_submission_pack(items, predicted_raw, rd / "submission_pack.json", time_budget_s=cfg.replay_budget_s)

    # Distill the CompactPrior — the handoff artifact the 9000 s submission loads.
    # On resume, MERGE with the prior we loaded so findings accumulate over runs.
    from engine.prior import distill
    prior = distill(result, target=f"{cfg.agent}+{cfg.guardrail}", top_k=cfg.prior_top_k, strategies=bank)
    to_merge = [p for p in (prior, resume_prior, ext_prior) if p is not None]
    if len(to_merge) > 1:
        prior = merge_priors(*to_merge, top_k=cfg.prior_top_k)
        prior.strategies = bank
    prior.save(rd / "compact_prior.json")

    st = result["stats"]
    print("\n[score_jed] done in %.1fs" % (time.perf_counter() - t0))
    print(f"  programs evaluated : {st.programs_seen}")
    print(f"  oracle replays     : {st.oracle_calls}")
    print(f"  prefix-cache reuse : {st.prefix_reuse} replays  ({st.prefix_saved_msgs} leading msgs served from snapshots)")
    print(f"  repair-from-break  : {st.repair_from_break} programs")
    print(f"  go-explore expand  : {st.go_explore_expansions} continuations")
    print(f"  VOI cold-resample  : {st.cold_resamples} replays  ({st.cold_validated_arms} arms cold-validated)")
    print(f"  llm calls          : {st.llm_calls}  ({st.llm_programs} programs, {st.llm_parse_fail} empty)")
    if st.llm_calls:
        print(f"  llm cost/value     : {st.llm_wall_s:.0f}s wall, {st.llm_downstream_eligible} eligible arms, "
              f"downstream EV/s={st.llm_downstream_ev:.2f}")
    print(f"  eligible arms      : {st.positive_arms}  (cold-validated {st.cold_validated_arms})")
    print(f"  unique cells found : {st.cells}")
    print(f"  compact prior      : {len(prior.arms)} arms -> {rd / 'compact_prior.json'}")
    print(f"  packed candidates  : {len(items)}  predicted_raw={predicted_raw:.1f}  "
          f"(~{min(1000.0, predicted_raw/200000.0*1000.0):.2f} normalized)")
    print("  top arms:")
    for a in arms[:8]:
        if a.eligible:
            print(f"    {a.program_name:34s} EV/s={a.severity_ev_per_s:6.3f} "
                  f"sevraw={a.mean_severity_raw:4.0f} cells={len(a.cell_hashes)} p={a.success_p:.2f}")
    print(f"  outputs -> {rd}")
    return {"result": result, "items": items, "predicted_raw": predicted_raw}


def _parse_args(argv: list[str] | None = None) -> RunConfig:
    ap = argparse.ArgumentParser(description="score_jed offline high-EV attack miner")
    ap.add_argument("--budget", type=float, default=120.0, help="search wall-clock seconds")
    ap.add_argument("--agent", default="deterministic", help="deterministic | gpt_oss | gemma | ...")
    ap.add_argument("--guardrail", default="optimal", choices=["optimal", "allow", "rules"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--trials", type=int, default=1, help="initial screen trials (VOI resampling adds cold trials)")
    ap.add_argument("--beam", type=int, default=6)
    ap.add_argument("--mutations", type=int, default=6)
    ap.add_argument("--llm", type=int, default=0, help="LLM proposals per expansion (0 = mutation-only)")
    ap.add_argument("--llm-every", type=int, default=4, help="run the LLM phase every N steps (it is slow)")
    ap.add_argument("--llm-top", type=int, default=3, help="LLM-mutate the top-M nodes concurrently per LLM phase")
    ap.add_argument("--cold-start", type=int, default=0, help="LLM designs this many programs FROM SCRATCH at init")
    ap.add_argument("--no-go-explore", action="store_true", help="disable return-to-archived-state exploration")
    ap.add_argument("--prior-top-k", type=int, default=40, help="how many top arms to keep in the compact prior")
    ap.add_argument("--resume", action="store_true", help="load results/compact_prior.json as seeds + merge (checkpoint)")
    ap.add_argument("--seed-prior", default=None, help="external prior JSON to warm-start from (e.g. a V28 prior)")
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--replay-budget", type=float, default=9000.0, help="target replay budget for packing")
    ap.add_argument("--gguf", default=None, help="explicit .gguf path for a GGUF target (else auto-locate/env)")
    ap.add_argument("--n-ctx", type=int, default=8192)
    ap.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = offload all layers to GPU")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-cache", action="store_true", help="disable the snapshot prefix cache (cold replays)")
    a = ap.parse_args(argv)
    return RunConfig(
        budget_s=a.budget, agent=a.agent, guardrail=a.guardrail, seed=a.seed, trials=a.trials,
        beam=a.beam, mutations=a.mutations, llm=a.llm, llm_every=a.llm_every, llm_top_m=a.llm_top,
        cold_start=a.cold_start, go_explore=not a.no_go_explore, prior_top_k=a.prior_top_k,
        resume=a.resume, seed_prior=a.seed_prior,
        llm_model=a.llm_model, replay_budget_s=a.replay_budget,
        gguf_path=a.gguf, n_ctx=a.n_ctx, n_gpu_layers=a.n_gpu_layers, max_new_tokens=a.max_new_tokens,
        use_prefix_cache=not a.no_cache,
    )


if __name__ == "__main__":
    try:
        run(_parse_args())
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\n[score_jed] target not ready: {exc}", file=sys.stderr)
        print(
            "  For a GGUF target: install `llama-cpp-python` (GPU build for T4 parity) and either\n"
            "  pass --gguf <path/to/model.gguf>, set GPT_OSS_MODEL_PATH / GEMMA_MODEL_PATH, or\n"
            "  set GGUF_SEARCH_DIRS to a folder containing the weights.",
            file=sys.stderr,
        )
        raise SystemExit(2)
