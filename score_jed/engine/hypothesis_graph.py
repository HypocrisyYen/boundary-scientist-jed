"""Evidence-backed hypothesis memory — replaces the flat LabNotebook.

The old notebook stored raw LLM sentences (any string > 6 chars) and showed the
model only the last 24 of ~1600 (~1.5% recall), with no scope, confidence, or
contradiction handling. This module makes each unit of knowledge a SCOPED,
STATUSED, EVIDENCED hypothesis, and retrieves by RELEVANCE to the current move,
so a 1000-hypothesis store still surfaces the handful that matter.

A hypothesis carries:
  * scope   — {model, guardrail}: a rule found on `allow` must NOT be assumed on `optimal`;
  * status  — proposed / supported / confirmed / contradicted / refuted (from evidence counts);
  * confidence — Beta mean over supporting vs contradicting experiments;
  * evidence — the experiment ids that support/contradict it (falsification-ready).

Retrieval blends token overlap with the query, scope match, informativeness
(confirmed rules + testable open hypotheses both score high), and recency.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_WORD = re.compile(r"[a-z0-9_.]+")
# NB: negation words (not/no/never/cannot/without/n't) are DELIBERATELY NOT stopped —
# dropping them made "allowed" == "not allowed" collapse to one hypothesis (finding 9).
_STOP = {"the", "a", "an", "is", "it", "to", "of", "and", "or", "in", "on", "for", "with",
         "that", "this", "was", "its", "at", "by", "be", "as", "if", "then"}
# BUG FIX (found in code review): this used to ALSO match domain verbs (block/deny/reject/
# refuse/fail) alongside true negation particles, on the theory that "X is blocked" is
# semantically similar to "X is not allowed". That conflates SEMANTIC negativity (what the verb
# means) with GRAMMATICAL polarity (whether the statement itself is negated) — with the effect
# that "email.send is blocked by the guardrail" (an AFFIRMATIVE claim) and "email.send is NOT
# blocked by the guardrail" (its actual negation) BOTH matched "blocked" and got the same "neg"
# tag, so the near-dup Jaccard check in add() silently merged two CONTRADICTORY hypotheses into
# one — exactly the "allowed" == "not allowed" collapse this module's design already claims to
# prevent, just via a different word family. Only true negation particles belong here; a domain
# verb's own token already participates in the ordinary Jaccard token-overlap comparison.
_NEG = re.compile(r"\b(not|no|never|cannot|can't|won't|doesn't|isn't|aren't|without)\b")


def _classify(s: int, c: int) -> str:
    """Single authoritative status rule from supporting/contradicting COUNTS, so the
    incremental (`register`) and authoritative (`recompute_from_experiments`) paths can
    never drift. Symmetric: two decisive trial-backed experiments one-sided ⇒ confirmed /
    refuted (matches ``evidence.hypothesis_verdict``); a bare majority ⇒ supported/contradicted."""
    if s >= 2 and c == 0:
        return "confirmed"
    if c >= 2 and s == 0:
        return "refuted"
    if c > s:
        return "contradicted"
    if s > c:
        return "supported"
    return "proposed"


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 1}


def _polarity(statement: str) -> str:
    return "neg" if _NEG.search((statement or "").lower()) else "pos"


def _norm(statement: str) -> str:
    # polarity-prefixed key so "http.post allowed" and "http.post NOT allowed" differ.
    return _polarity(statement) + "|" + " ".join(sorted(_tokens(statement)))[:200]


def _scoped_norm(statement: str, scope: dict | None) -> str:
    """Dedup key that includes SCOPE (finding 12) — the same claim under different
    (model, guardrail) scopes are DIFFERENT hypotheses, not one."""
    scope = scope or {}
    return f"{scope.get('model','')}|{scope.get('guardrail','')}||" + _norm(statement)


# Near-duplicate re-proposal detection: exact _norm matching only catches identical token
# SETS, so a lightly reworded restatement ("keyed on (tool, data) pair" vs "...pair, not
# (tool, url, data)") gets treated as a brand-new hypothesis instead of a re-proposal of the
# same idea — the corpus fills with near-clones and nothing signals the LLM it is circling.
# A looser Jaccard-overlap check (same scope + polarity) catches this generically, for
# whatever topic happens to recur — this is DATA (a counter on the hypothesis), not a
# hardcoded rule about any specific claim.
_NEAR_DUP_JACCARD = 0.45
_CHURN_WARN_AT = 3   # re-proposed this many times while not converging -> surface a warning


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b) if inter else 0.0


# an atomic claim carries ONE verdict/clause; a bundle like "CONFIRMED A, REFUTED B,
# maybe C" must NOT be stored as one hypothesis (finding 11).
_VERDICT_MARK = re.compile(r"\b(confirm(?:ed|s)?|refut(?:ed|es)?|support(?:ed|s)?|"
                           r"contradict(?:ed|s)?|proven?|disproven?)\b", re.I)


def _is_composite(statement: str) -> bool:
    s = (statement or "").strip()
    if ";" in s or "\n" in s:
        return True
    if len(_VERDICT_MARK.findall(s)) >= 2:
        return True
    # two substantial clauses joined by ", <conj>" / " and also " / " whereas "
    clauses = [c for c in re.split(r",\s+(?:and|but|while|whereas|however|also)\b|\.\s+", s) if len(c.strip()) > 18]
    return len(clauses) >= 2


@dataclass
class Hypothesis:
    hid: str
    statement: str
    scope: dict                       # {"model": ..., "guardrail": ...}
    status: str = "proposed"
    confidence: float = 0.5
    supporting: list[str] = field(default_factory=list)   # experiment ids
    contradicting: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    proposal_count: int = 1           # how many times this idea (incl. near-duplicate rewordings)
                                       # has been (re-)proposed — DATA the churn warning reads

    def register(self, exp_id: str, supports: bool) -> None:
        (self.supporting if supports else self.contradicting).append(exp_id)
        s, c = len(self.supporting), len(self.contradicting)
        self.confidence = (s + 1.0) / (s + c + 2.0)          # Beta(1,1) posterior mean
        st = _classify(s, c)
        if st != "proposed":
            self.status = st
        self.updated = time.time()


class HypothesisGraph:
    def __init__(self) -> None:
        self.hyps: dict[str, Hypothesis] = {}
        self._by_norm: dict[str, str] = {}          # normalized-statement -> hid
        self._exp_ctr = 0

    # -- construction ---------------------------------------------------------

    def _mk_id(self, statement: str) -> str:
        return "h" + hashlib.sha256(statement.encode("utf-8")).hexdigest()[:10]

    def add(self, statement: str, scope: dict, *, tags: list[str] | None = None,
            status: str = "proposed", confidence: float = 0.5) -> Hypothesis | None:
        statement = (statement or "").strip()
        if len(statement) < 8:
            return None
        if _is_composite(statement):                 # only ONE atomic falsifiable claim per hypothesis
            return None
        scope = dict(scope or {})
        norm = _scoped_norm(statement, scope)        # scope-aware dedup (finding 12)
        if norm in self._by_norm:                    # dedupe EXACT-token-set statements WITHIN a scope
            existing = self.hyps[self._by_norm[norm]]
            existing.proposal_count += 1
            existing.updated = time.time()
            return existing
        # NEAR-duplicate check: a reworded restatement of an existing idea in the same scope
        # (same polarity, high token overlap) is a RE-PROPOSAL, not a new hypothesis — bump its
        # counter instead of cloning it (this is what makes churn visible as data).
        pol = _polarity(statement)
        toks = _tokens(statement)
        md, gr = scope.get("model"), scope.get("guardrail")
        for h in self.hyps.values():
            if h.scope.get("model") != md or h.scope.get("guardrail") != gr:
                continue
            if _polarity(h.statement) != pol:
                continue
            if _jaccard(toks, _tokens(h.statement)) >= _NEAR_DUP_JACCARD:
                h.proposal_count += 1
                h.updated = time.time()
                return h
        hid = self._mk_id(norm)
        h = Hypothesis(hid=hid, statement=statement[:280], scope=dict(scope or {}),
                       status=status, confidence=confidence, tags=list(tags or []))
        self.hyps[hid] = h
        self._by_norm[norm] = hid
        return h

    def new_experiment_id(self) -> str:
        self._exp_ctr += 1
        return f"exp{self._exp_ctr:06d}"

    def register_evidence(self, hid: str, exp_id: str, supports: bool) -> None:
        # Incremental convenience only — the AUTHORITATIVE verdict comes from
        # recompute_from_experiments() over the immutable EventStore.
        h = self.hyps.get(hid)
        if h is not None:
            h.register(exp_id, supports)

    def recompute_from_experiments(self, experiments: list[dict]) -> None:
        """AUTHORITATIVE (P0-C): derive every hypothesis verdict ONLY from real Experiments
        that carry LINKED TRIALS. An experiment with empty ``trial_ids`` is inadmissible, and
        the LLM's text can never move a verdict — only measured, trial-backed experiments can.
        Overwrites supporting/contradicting from the event store (idempotent, recomputable)."""
        from collections import defaultdict
        by_hyp: dict[str, list[dict]] = defaultdict(list)
        for e in experiments:
            if not e.get("trial_ids"):
                continue                                  # no measurement -> not admissible
            if e.get("verdict") not in ("supports", "contradicts"):
                continue
            hid = e.get("hypothesis_id")
            if hid:
                by_hyp[hid].append(e)
        for hid, h in self.hyps.items():
            exps = by_hyp.get(hid, [])
            h.supporting = [e["experiment_id"] for e in exps if e.get("verdict") == "supports"]
            h.contradicting = [e["experiment_id"] for e in exps if e.get("verdict") == "contradicts"]
            s, c = len(h.supporting), len(h.contradicting)
            h.confidence = (s + 1.0) / (s + c + 2.0)
            h.status = _classify(s, c)                       # authoritative, from trial-backed experiments only

    # -- retrieval ------------------------------------------------------------

    def relevant(self, query: str, scope: dict, k: int = 8, *, in_scope_only: bool = True) -> list[Hypothesis]:
        """Top-k IN-SCOPE hypotheses by relevance (finding 10: scope is a HARD filter,
        not a soft bonus). An `allow` finding is NOT returned for an `optimal` query;
        cross-scope knowledge is only surfaced via ``transfer_candidates`` (a transfer lane).
        """
        qt = _tokens(query)
        now = time.time()
        gr = (scope or {}).get("guardrail")
        md = (scope or {}).get("model")

        def in_scope(h: Hypothesis) -> bool:
            if gr and h.scope.get("guardrail") not in (None, gr):
                return False
            if md and h.scope.get("model") not in (None, md):
                return False
            return True

        def score(h: Hypothesis) -> float:
            ht = _tokens(h.statement)
            overlap = len(qt & ht) / (1.0 + len(qt | ht)) if qt else 0.0
            if h.status == "confirmed":
                info = 0.35
            elif h.status == "refuted":
                info = -0.30
            elif h.status in ("supported", "contradicted"):
                info = 0.15
            else:
                info = 0.20 * (1.0 - abs(h.confidence - 0.5) * 2.0)   # peak at 0.5
            recency = 0.10 * (1.0 / (1.0 + (now - h.updated) / 3600.0))
            return overlap + info + recency

        pool = [h for h in self.hyps.values() if (not in_scope_only or in_scope(h))]
        return sorted(pool, key=score, reverse=True)[:k]

    def transfer_candidates(self, query: str, scope: dict, k: int = 4) -> list[Hypothesis]:
        """SUPPORTED/CONFIRMED hypotheses from OTHER scopes — surfaced only in a transfer
        lane, never treated as facts in the current scope (finding 10)."""
        gr = (scope or {}).get("guardrail")
        qt = _tokens(query)
        out = [h for h in self.hyps.values()
               if h.status in ("confirmed", "supported") and gr and h.scope.get("guardrail") not in (None, gr)]
        out.sort(key=lambda h: (h.status == "confirmed", len(qt & _tokens(h.statement))), reverse=True)
        return out[:k]

    def churning(self, scope: dict, *, min_count: int = _CHURN_WARN_AT, k: int = 3) -> list[Hypothesis]:
        """In-scope hypotheses re-proposed (incl. near-duplicate rewordings) at least
        ``min_count`` times while STILL not converging (not confirmed/supported) — a data-driven
        signal that this line of inquiry has diminishing returns, generic to whatever topic the
        LLM happens to be circling (mirrors FailureIndex's tool-denial dead-end pattern, but for
        repeated REASONING rather than repeated tool denials)."""
        gr, md = (scope or {}).get("guardrail"), (scope or {}).get("model")

        def in_scope(h: Hypothesis) -> bool:
            return (not gr or h.scope.get("guardrail") in (None, gr)) and (not md or h.scope.get("model") in (None, md))

        pool = [h for h in self.hyps.values()
               if in_scope(h) and h.proposal_count >= min_count and h.status not in ("confirmed", "supported")]
        return sorted(pool, key=lambda h: h.proposal_count, reverse=True)[:k]

    def render(self, query: str, scope: dict, k: int = 8, *, transfer_k: int = 4) -> str:
        hits = self.relevant(query, scope, k)
        icon = {"confirmed": "✓CONFIRMED", "refuted": "✗REFUTED", "supported": "~supported",
                "contradicted": "~contradicted", "proposed": "?open"}
        out: list[str] = []
        if hits:
            out.append("RELEVANT KNOWLEDGE (evidence-backed; test the open ones, exploit the confirmed):")
            for h in hits:
                sc = h.scope.get("guardrail", "?")
                out.append(f"  [{icon.get(h.status, h.status)} p={h.confidence:.2f} @{sc}] {h.statement}")
        else:
            out.append("(no hypotheses yet — this is early; probe and record what you confirm/refute)")
        # CROSS-SCOPE TRANSFER LANE — knowledge proven in OTHER scopes, explicitly labeled
        # NOT-a-fact-here. This is how a mechanism proven on one guardrail reaches the others
        # without breaking scope rigor: it arrives as a testable prior, not a belief.
        xfer = self.transfer_candidates(query, scope, transfer_k)
        if xfer:
            out.append("CROSS-SCOPE TRANSFER (proven in OTHER scopes; UNVERIFIED here — cheap to test, do not assume):")
            for h in xfer:
                out.append(f"  [transfer from @{h.scope.get('guardrail', '?')}] {h.statement}")
        churn = self.churning(scope)
        if churn:
            out.append("CIRCLING WARNING (re-proposed repeatedly without converging — try a DIFFERENT "
                       "structural direction, not another rewording of the same idea):")
            for h in churn:
                out.append(f"  [x{h.proposal_count}, still {h.status}] {h.statement}")
        return "\n".join(out)

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"hyps": [asdict(h) for h in self.hyps.values()],
                                 "exp_ctr": self._exp_ctr}, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HypothesisGraph":
        g = cls()
        p = Path(path)
        if p.is_file():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                for hd in d.get("hyps", []):
                    h = Hypothesis(**hd)
                    g.hyps[h.hid] = h
                    g._by_norm[_scoped_norm(h.statement, h.scope)] = h.hid
                g._exp_ctr = int(d.get("exp_ctr", 0))
            except Exception:
                pass
        return g

    def prune(self, max_size: int = 800) -> int:
        """Bound the graph for a 24h run: keep CONFIRMED rules + the most informative /
        recent hypotheses; drop refuted noise and stale low-confidence proposals first.
        Returns how many were dropped."""
        if len(self.hyps) <= max_size:
            return 0
        now = time.time()

        def keep_score(h: Hypothesis) -> float:
            base = {"confirmed": 3.0, "supported": 1.0, "contradicted": 0.2,
                    "proposed": 0.5, "refuted": -1.0}.get(h.status, 0.3)
            evidence = 0.3 * (len(h.supporting) + len(h.contradicting))
            recency = 1.0 / (1.0 + (now - h.updated) / 86400.0)
            return base + evidence + recency

        ranked = sorted(self.hyps.values(), key=keep_score, reverse=True)
        keep = ranked[:max_size]
        dropped = len(self.hyps) - len(keep)
        self.hyps = {h.hid: h for h in keep}
        self._by_norm = {_scoped_norm(h.statement, h.scope): h.hid for h in keep}
        return dropped

    def stats(self) -> dict:
        by_status: dict[str, int] = {}
        for h in self.hyps.values():
            by_status[h.status] = by_status.get(h.status, 0) + 1
        return {"total": len(self.hyps), **by_status}
