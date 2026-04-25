from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ResistanceSignal:
    resistant: bool
    reason: str
    confidence: float


def detect_resistance(
    *,
    recent_hba1c: List[float],
    regimen: Dict[str, Any],
    min_weeks_on_regimen: int = 8,
) -> ResistanceSignal:
    """
    Inputs:
      - recent_hba1c: last N HbA1c readings (weekly proxy)
      - regimen: {"meds":..., "lifestyle_intensity":...}
    Outputs:
      - resistance signal (mock): no improvement over 8+ weeks on therapy
    """
    meds = regimen.get("meds", {}) or {}
    weeks_on = 0
    for _drug, meta in meds.items():
        weeks_on = max(weeks_on, int(meta.get("weeks_on", 0)))

    if weeks_on < min_weeks_on_regimen or len(recent_hba1c) < 6:
        return ResistanceSignal(resistant=False, reason="insufficient_history", confidence=0.2)

    # if last ~6 weeks trend is flat or worsening while on meds
    start = float(recent_hba1c[-6])
    end = float(recent_hba1c[-1])
    delta = end - start
    if delta > -0.05:  # not improving
        conf = min(0.9, 0.4 + 0.2 * len(meds))
        return ResistanceSignal(resistant=True, reason="hba1c_not_improving_on_therapy", confidence=conf)

    return ResistanceSignal(resistant=False, reason="responding", confidence=0.6)


def suggest_alternatives(*, current_regimen: Dict[str, Any], ckd: bool, cvd: bool, obese: bool) -> List[Dict[str, Any]]:
    """
    Returns candidate action dicts (Phase 4 schema) as alternatives.
    """
    meds = set((current_regimen.get("meds") or {}).keys())
    candidates: List[Dict[str, Any]] = []

    if "metformin" not in meds and not ckd:
        candidates.append({"type": "start", "drug": "metformin", "dose": 1.0, "lifestyle": 0.6})
    if ("glp1" not in meds) and (obese or cvd):
        candidates.append({"type": "add", "drug": "glp1", "dose": 1.0})
    if ("sglt2" not in meds) and (cvd or ckd is False):
        candidates.append({"type": "add", "drug": "sglt2", "dose": 1.0})
    if "insulin" not in meds:
        candidates.append({"type": "start", "drug": "insulin", "dose": 0.7})

    if not candidates:
        candidates.append({"type": "dose_adjust", "drug": next(iter(meds), "metformin"), "dose": 1.0})
    return candidates

