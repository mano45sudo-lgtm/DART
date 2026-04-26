"""Proposes glucocentric treatment steps (heuristic, CPU-only)."""

from __future__ import annotations

import random
from typing import Any, Dict

from .base_agent import BaseAgent


def _f(state: Dict[str, Any], key: str, default: float) -> float:
    return float(state.get(key, default))


class TreatmentAgent(BaseAgent):
    name = "treatment"
    _rng: random.Random

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def propose(self, state: Dict[str, Any]) -> Dict[str, Any]:
        fg = _f(state, "fasting_glucose", 120.0)
        hba1c = _f(state, "hba1c", 7.0)
        n_meds = int(state.get("_n_meds", 0) or 0)
        j = 0.02 * self._rng.random()  # tiny jitter for disagreement stories

        if fg < 100:
            return {"type": "noop", "lifestyle": min(0.80, 0.55 + j)}
        if fg > 250 or hba1c > 9.5:
            if n_meds == 0:
                return {"type": "start", "drug": "metformin", "dose": 0.9 + j, "lifestyle": 0.65}
            if n_meds < 3 and self._rng.random() < 0.55:
                return {"type": "add", "drug": "glp1", "dose": 0.75, "lifestyle": 0.6}
            return {"type": "dose_adjust", "drug": "insulin", "dose": 0.55 + j, "lifestyle": 0.6}
        if fg > 160 or hba1c > 7.2:
            if n_meds == 0:
                return {"type": "start", "drug": "metformin", "dose": 0.85, "lifestyle": 0.6 + j}
            return {"type": "dose_adjust", "drug": "metformin", "dose": 0.9, "lifestyle": 0.6}
        if fg > 130:
            return {
                "type": "dose_adjust" if n_meds else "start",
                "drug": "metformin",
                "dose": 0.75,
                "lifestyle": 0.55 + j,
            }
        return {"type": "noop", "lifestyle": 0.5 + 0.1 * j}

    def evaluate(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        fg = _f(state, "fasting_glucose", 120.0)
        hba1c = _f(state, "hba1c", 7.0)
        t = str(action.get("type", "noop"))
        drug = str(action.get("drug", "none")).lower()
        score = 0.0
        if fg > 140 and t in {"start", "add", "dose_adjust", "switch"} and drug not in {"none", ""}:
            score += 0.5
        if fg > 180 and t != "noop":
            score += 0.3
        if hba1c > 7.5 and t != "noop":
            score += 0.25
        if fg < 100 and t == "noop":
            score += 0.2
        if t == "noop" and fg > 160:
            score -= 0.4
        return float(score)
