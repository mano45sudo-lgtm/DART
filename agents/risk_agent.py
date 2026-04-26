"""Safety and polypharmacy heuristics (hypoglycemia, overload)."""

from __future__ import annotations

from typing import Any, Dict

from .base_agent import BaseAgent


def _f(state: Dict[str, Any], key: str, default: float) -> float:
    return float(state.get(key, default))


class RiskAgent(BaseAgent):
    name = "risk"

    def __init__(self) -> None:
        pass

    def propose(self, state: Dict[str, Any]) -> Dict[str, Any]:
        fg = _f(state, "fasting_glucose", 120.0)
        n_meds = int(state.get("_n_meds", 0) or 0)
        if fg < 85:
            if n_meds > 0:
                return {"type": "dose_adjust", "drug": "insulin", "dose": 0.2, "lifestyle": 0.65}
            return {"type": "noop", "lifestyle": 0.7}
        if n_meds >= 4:
            return {"type": "dose_adjust", "drug": "metformin", "dose": 0.45, "lifestyle": 0.55}
        if _f(state, "egfr", 80.0) < 35 and n_meds >= 2:
            return {"type": "dose_adjust", "drug": "metformin", "dose": 0.4, "lifestyle": 0.55}
        return {"type": "noop", "lifestyle": 0.5}

    def evaluate(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        fg = _f(state, "fasting_glucose", 120.0)
        n_meds = int(state.get("_n_meds", 0) or 0)
        t = str(action.get("type", "noop"))
        drug = str(action.get("drug", "none")).lower()
        dose = _f(action, "dose", 0.0) if "dose" in action else 0.0
        score = 0.0
        ins_or_su = drug in {"insulin", "sulfonylurea"}
        if ins_or_su and fg < 100:
            score -= 1.0 + (100 - fg) * 0.02
        if t in {"add", "start"} and n_meds >= 4:
            score -= 0.45
        if t == "dose_adjust" and drug == "insulin" and dose > 0.75 and fg < 120:
            score -= 0.5
        if t == "noop" and 70 <= fg <= 180:
            score += 0.15
        if t in {"dose_adjust", "stop"} and fg < 90 and ins_or_su:
            score += 0.3
        if n_meds >= 3 and t == "add":
            score -= 0.25
        return float(score)
