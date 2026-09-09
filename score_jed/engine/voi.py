"""Value-of-Information scheduler — WHERE to spend the next measurement.

The discovery loop makes two selection decisions each episode, and both used to be
heuristics (a fixed focus cycle; a length-penalized severity sort). This module
replaces them with ONE principled criterion: spend the next scarce measurement
where it most reduces our uncertainty about *where raw/sec can still be improved*.

Two consumers, same principle:

  * :func:`pick_focus` — which predicate FAMILY the next investigation targets.
    VOI is high where we are still uncertain (a family with no coverage and no
    confirmed block, a champion arm with a wide confidence interval, an untested
    hypothesis to resolve) and LOW where information is already in hand (a
    well-covered family, or a wall we have hit N times with zero score — a learned
    structural block, further attempts there are uninformative). Every few episodes
    it deliberately returns ``None`` (open breadth) so exploration never starves.

  * :func:`candidate_value` — which discovered RECIPE earns scarce cold replays.
    Value = the in-episode raw/sec proxy (severity ÷ replay time — the north star)
    plus novelty (a new cell/family) plus hypothesis relevance plus measurement
    uncertainty. LENGTH is a COST feature only (a mild penalty) — it can NEVER
    eliminate a longer, higher-EV recipe before it is ever measured (fixes the
    shortest-5 pre-measurement cut).

Nothing here decides WHICH attack is best; it only decides which measurement to
buy next. The measured posterior + deterministic verdicts still decide truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

# families are the four scoring predicates (the focus axis).
PREDICATES = ("EXFILTRATION", "CONFUSED_DEPUTY", "UNTRUSTED_TO_ACTION", "DESTRUCTIVE_WRITE")


@dataclass(frozen=True)
class FocusWeights:
    unknown: float = 1.0       # never covered AND not yet exhausted -> maximal info
    uncertainty: float = 0.8   # a wide CI on the family champion -> measuring narrows it
    open_hyp: float = 0.5      # an untested hypothesis targeting the family -> resolve it
    coverage: float = 0.7      # many eligible arms -> diminishing info (saturation)
    futility: float = 1.0      # focused, never scored -> uninformative wall


@dataclass(frozen=True)
class CandidateWeights:
    new_cell: float = 2.0
    new_family: float = 3.0
    open_hyp: float = 1.0
    uncertainty: float = 1.5
    severity: float = 0.5      # per-finding value (cap-bound lane): reward high severity, √-discount latency
    length: float = 0.1        # COST ONLY — never a hard cut
    # a flat bonus for a candidate whose OWN causally-stated mechanism ("why" it should work) does
    # not overlap the mechanisms already represented among the current top arms — same shape as
    # new_cell/new_family. Only recipes the LLM has actually explained get this (unmeasured/
    # unexplained candidates get none), so it specifically rewards mechanism-diverse REASONING,
    # not just any novel-looking recipe.
    mechanism_diversity: float = 2.0


def focus_voi(coverage: dict[str, int], focus_stats: dict[str, dict],
              open_hyps: dict[str, int], uncertainty: dict[str, float], *,
              families: tuple[str, ...] = PREDICATES, block_after: int = 3,
              w: FocusWeights = FocusWeights()) -> dict[str, float]:
    """Value of information for each family. Higher = more worth investigating next.

    ``coverage``     family -> # eligible arms already found there;
    ``focus_stats``  family -> {"attempts", "scores"} (how often we've targeted it / it paid);
    ``open_hyps``    family -> # open (untested) hypotheses about it;
    ``uncertainty``  family -> success-probability CI width of its champion arm (0 if none).

    A family focused ``block_after`` times with ZERO score is a confirmed structural block:
    its VOI collapses to a tiny coverage-tiebreak, so it stops being re-targeted WITHOUT
    hardcoding which family it is (it is learned from the scientist's own experiments).
    """
    voi: dict[str, float] = {}
    for f in families:
        cov = float(coverage.get(f, 0))
        st = focus_stats.get(f, {}) or {}
        attempts = int(st.get("attempts", 0))
        scores = int(st.get("scores", 0))
        if attempts >= block_after and scores == 0:
            # learned block: only an infinitesimal, coverage-ordered residue so ties resolve
            # deterministically toward the least-explored blocked family.
            voi[f] = -1e6 + 1.0 / (1.0 + cov)
            continue
        unknown = 1.0 if (cov == 0 and attempts == 0) else 0.0
        unc = float(uncertainty.get(f, 1.0 if cov == 0 else 0.0))
        openq = min(float(open_hyps.get(f, 0)), 3.0) / 3.0
        saturation = cov / (1.0 + cov)
        futility = float(max(0, attempts - scores))
        voi[f] = (w.unknown * unknown + w.uncertainty * unc + w.open_hyp * openq
                  - w.coverage * saturation - w.futility * futility * 0.25)
    return voi


def pick_focus(ep: int, voi: dict[str, float], *, breadth_every: int = 4,
               families: tuple[str, ...] = PREDICATES) -> str | None:
    """Pick the family to focus this episode, or ``None`` for an OPEN (breadth) episode.

    Every ``breadth_every``-th episode is open so the search keeps discovering new
    mechanisms; otherwise take the maximum-VOI family (deterministic: highest VOI, then
    canonical family order). If every family is a confirmed block, the max is still the
    least-explored blocked one — which degrades gracefully to breadth on the next cycle."""
    if breadth_every > 0 and ep % breadth_every == 0:
        return None
    if not voi:
        return None
    order = {f: i for i, f in enumerate(families)}
    return min(voi, key=lambda f: (-voi[f], order.get(f, 99)))


def mechanism_is_novel(candidate_why: str, top_arm_whys: Sequence[str], *, jaccard_max: float = 0.3) -> bool:
    """True iff ``candidate_why`` (the LLM's own causal note for this recipe) shares little
    token overlap with any of the current top arms' notes — i.e. this candidate reasons about a
    DIFFERENT mechanism than what's already well-represented at the top of the leaderboard.
    Returns False (no bonus) when either side has no note — this only rewards LLM-explained
    recipes being genuinely different, not merely unexplained/unmeasured ones."""
    a = _tokens(candidate_why)
    if not a:
        return False
    for w in top_arm_whys:
        b = _tokens(w)
        if not b:
            continue
        inter = len(a & b)
        union = len(a | b)
        if union and inter / union > jaccard_max:
            return False
    return True


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


def candidate_value(*, severity: float, latency: float, new_cell: bool = False,
                    new_family: bool = False, open_hyp: float = 0.0, uncertainty: float = 0.0,
                    length: int = 1, mechanism_novel: bool = False,
                    w: CandidateWeights = CandidateWeights()) -> float:
    """VOI of buying cold replays for one discovered recipe (pre-measurement selection).

    TWO value lanes, matching the submission knapsack (Σseverity + 2·cells over a capped, time-
    budgeted set): the raw/sec proxy (severity ÷ replay time) rewards FAST FILLERS; a severity term
    (severity ÷ √latency) rewards HIGH-SEVERITY chains that are worth most per finding once the
    finding-count cap binds — so a reliable slow multi-predicate chain is NOT dropped for being slow.
    Novelty / hypothesis-relevance / uncertainty add value; LENGTH is only a mild cost.
    ``mechanism_novel`` (see :func:`mechanism_is_novel`) rewards a candidate that reasons about a
    mechanism different from the current top arms', so a single fast champion doesn't starve
    replays away from a differently-mechanized recipe that might COMBINE with it later.
    """
    lat = max(float(latency), 0.1)
    raw_proxy = float(severity) / lat                       # filler lane (budget-bound)
    sev_value = float(severity) / (lat ** 0.5)              # high-severity lane (cap-bound)
    return (raw_proxy + w.severity * sev_value
            + w.new_cell * (1.0 if new_cell else 0.0)
            + w.new_family * (1.0 if new_family else 0.0)
            + w.open_hyp * float(open_hyp)
            + w.uncertainty * float(uncertainty)
            + w.mechanism_diversity * (1.0 if mechanism_novel else 0.0)
            - w.length * float(max(0, length)))
