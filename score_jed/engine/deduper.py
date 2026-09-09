"""Fingerprints for dedupe (analog of score_golf.deduper.fingerprints).

Two levels, mirroring score_golf's (ast_fp, text_fp):
  * behavior_fp — coarse tactic identity (family/mechanism/encoding + intents);
    collapses near-clones so search doesn't waste replays re-testing the same idea;
  * text_fp — exact rendered-message identity; the true arm key.
"""

from __future__ import annotations

import hashlib

from engine.contracts import AttackProgram


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def behavior_fp(program: AttackProgram) -> str:
    return _sha("|".join(program.behavior_key))


def text_fp(program: AttackProgram, rendered: tuple[str, ...]) -> str:
    return _sha("␟".join(rendered))


def fingerprints(program: AttackProgram, rendered: tuple[str, ...]) -> tuple[str, str]:
    """Return (behavior_fp, text_fp)."""
    return behavior_fp(program), text_fp(program, rendered)
