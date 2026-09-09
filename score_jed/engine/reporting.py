"""Persist search outputs: EV arm table, cell archive, and a submission pack."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from engine import scoring
from engine.scoring import EVArmStats, PortfolioItem


def write_arm_csv(arms: Sequence[EVArmStats], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "program", "sha", "family", "mechanism", "n", "n_positive", "success_p", "success_p_lb",
        "mean_severity_raw", "expected_severity_raw", "risk_adjusted_raw", "mean_replay_s",
        "severity_ev_per_s", "risk_adjusted_ev_per_s", "n_cells", "eligible", "fatal",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for a in arms:
            w.writerow(a.compact())


def write_archive(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stats": vars(result["stats"]),
        "n_arms": len(result["arms"]),
        "n_cells": len(result["archive_cells"]),
        "top_arms": [a.compact() for a in result["arms"][:50]],
        "programs": [
            {
                "name": n.program.name,
                "family": n.program.family,
                "mechanism": n.program.mechanism,
                "encoding": n.program.encoding,
                "source": n.program.source,
                "steps": [(s.text, s.intent) for s in n.program.steps],
                "arm": n.arm.compact(),
                "feedback": n.feedback,
            }
            for n in result["nodes"].values()
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_submission_pack(
    items: Sequence[PortfolioItem],
    predicted_raw: float,
    path: str | Path,
    *,
    time_budget_s: float,
) -> None:
    """Emit the ordered candidate messages the submission attack.py should replay."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates = [
        {
            "program": it.arm.program_name,
            "messages": list(it.arm.messages),
            "predicted_raw": round(it.predicted_raw, 3),
            "predicted_s": round(it.predicted_s, 4),
            "severity_ev_per_s": round(it.arm.severity_ev_per_s, 5),
        }
        for it in items
    ]
    payload = {
        "time_budget_s": time_budget_s,
        "predicted_raw": round(predicted_raw, 2),
        "predicted_normalized": round(min(1000.0, predicted_raw / 200000.0 * 1000.0), 4),
        "n_candidates": len(candidates),
        "candidates": candidates,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
