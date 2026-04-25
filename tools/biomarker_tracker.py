from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class BiomarkerTrend:
    name: str
    recent_values: List[float]
    slope: float
    improving: bool


def track_biomarkers(*, trajectory: List[Dict[str, Any]]) -> Dict[str, BiomarkerTrend]:
    """
    Inputs:
      - trajectory: list of state dicts (e.g., env.state()['state'] per step)
    Outputs:
      - per-biomarker trend summary (mock linear slope over last 4 points)
    """
    if not trajectory:
        return {}

    def _trend(key: str) -> BiomarkerTrend:
        vals = [float(s.get(key, 0.0)) for s in trajectory][-4:]
        if len(vals) < 2:
            slope = 0.0
        else:
            slope = (vals[-1] - vals[0]) / (len(vals) - 1)
        improving = slope < 0.0 if key in {"hba1c", "fasting_glucose", "bmi", "systolic_bp"} else slope > 0.0
        return BiomarkerTrend(name=key, recent_values=vals, slope=float(slope), improving=bool(improving))

    return {
        "hba1c": _trend("hba1c"),
        "fasting_glucose": _trend("fasting_glucose"),
        "bmi": _trend("bmi"),
        "systolic_bp": _trend("systolic_bp"),
        "egfr": _trend("egfr"),
    }

