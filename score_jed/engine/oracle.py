"""Oracle: replay a program and score it into an AttackReport.

Analog of score_golf's ``oracle.evaluate`` (which ran code on I/O cases and
classified the error). Here we replay the rendered message chain in the sandbox,
then apply the *official* SDK predicate + cell scoring so the numbers we optimize
match the numbers the evaluator would compute.
"""

from __future__ import annotations

from engine.contracts import AttackProgram, AttackReport, PredicateHit
from engine.program import render
from engine.sandbox import AttackSandbox
from engine.scoring import predicate_breakdown, severity_raw

# Fatal failure modes are hard-excluded from EV (never treated as a 0-EV sample).
# Everything else — refusals, partials, denials — is a legitimate EV observation.
_FATAL_PREFIXES = ("Exception", "TypeError", "ValueError", "KeyError", "RuntimeError")


def _stop_reason(report: AttackReport) -> str:
    if report.fatal_reason:
        return "fatal"
    if report.refused:
        return "refused"
    if report.predicates:
        return "scored:" + "+".join(sorted({p.predicate for p in report.predicates}))
    if report.denied:
        return "denied:" + report.denied[0].get("error", "?").split(":", 1)[0]
    if report.n_success_tools:
        return "no_predicate"
    return "no_tool_calls"


def evaluate(
    program: AttackProgram,
    sandbox: AttackSandbox,
    *,
    nonce: int,
    use_cache: bool | None = None,
    messages: tuple[str, ...] | None = None,
) -> AttackReport:
    # P0-2: with explicit ``messages``, replay those EXACT bytes (same-nonce confirmation)
    # instead of re-rendering a fresh URL — so repeated samples confirm ONE exact arm.
    if messages is None:
        messages = render(program, nonce)
    rr = sandbox.replay(messages, max_tool_hops=program.max_tool_hops, use_cache=use_cache)

    rep = AttackReport(
        program_sha=program.sha,
        program_name=program.name,
        messages=messages,
        ok=rr.fatal is None,
        fatal_reason=rr.fatal,
        replay_s=rr.replay_s or rr.wall_s,
        wall_s=rr.wall_s,
        refused=rr.refused,
        reused_prefix_len=rr.reused_prefix_len,
        cold=rr.cold,
        break_index=rr.first_bad_message,
    )
    if rr.fatal is not None:
        rep.stop_reason = "fatal"
        return rep

    trace = rr.trace
    events = list(trace.get("tool_events", []))
    rep.tool_events = events
    rep.n_tool_hops = len(events)
    rep.n_success_tools = sum(1 for e in events if e.get("ok"))
    rep.n_failed_tools = sum(1 for e in events if not e.get("ok"))
    rep.trace_chars = len(str(trace))

    # denials / guardrail blocks (drives repair feedback)
    for e in events:
        if not e.get("ok"):
            err = str(e.get("error") or "")
            if err.startswith("denied") or err.startswith("confirm_required"):
                rep.denied.append({"tool": e.get("name"), "error": err, "source": e.get("source")})

    # Official predicate + cell scoring. These can raise on malformed tool args
    # (e.g. http.post data=<int> -> "argument of type 'int' is not iterable"), so
    # isolate them: a scorer exception marks the arm FATAL (hard-excluded from EV)
    # rather than crashing the whole search run.
    from aicomp_sdk.core.cells import cell_signature
    from aicomp_sdk.core.predicates import eval_predicates

    try:
        preds = eval_predicates(trace)
        rep.predicates = [
            PredicateHit(p["predicate"], int(p.get("severity", 1)), p.get("evidence", {})) for p in preds
        ]
        rep.severity_raw = severity_raw(rep.predicates)
        rep.breakdown = predicate_breakdown(rep.predicates)
        if events:
            # Use the OFFICIAL score-cell key: cell_signature(tool_events) with
            # default args and NO user_messages — this is exactly what the evaluator
            # scores novelty on (aicomp_sdk.evaluation.ops builds
            # score_cell_signature = cell_signature(trace["tool_events"])). Keying on
            # a user-message-inclusive signature would over-count cells that differ
            # only by wording and never earn the real +2.
            sig = cell_signature(events)
            rep.cell_hash = str(sig["hash"])
    except Exception as exc:
        rep.ok = False
        rep.fatal_reason = f"scorer:{type(exc).__name__}: {exc}"
        rep.predicates = []
        rep.cell_hash = None
        rep.stop_reason = "fatal"
        return rep

    rep.stop_reason = _stop_reason(rep)
    return rep
