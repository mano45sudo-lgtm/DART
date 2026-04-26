"""Encourages monitoring / 'labs' through conservative noop and lifestyle (info proxy)."""

from __future__ import annotations

import random
from typing import Any, Dict

from .base_agent import BaseAgent


def _f(state: Dict[str, Any], key: str, default: float) -> float:
    return float(state.get(key, default))


class DiagnosticAgent(BaseAgent):
    name = "diagnostic"
    _rng: random.Random

    def __init__(self, seed: int = 1) -> None:
        self._rng = random.Random(seed)

    def propose(self, state: Dict[str, Any]) -> Dict[str, Any]:
        week = int(state.get("week", 0) or 0)
        j = 0.03 * self._rng.random()
        if week % 6 == 0 and week > 0:
            return {
                "type": "noop",
                "lifestyle": min(0.75, 0.5 + j),
                "rationale": "periodic review",
            }
        fg = _f(state, "fasting_glucose", 120.0)
        if 95 <= fg <= 125:
            return {"type": "noop", "lifestyle": 0.5 + j, "rationale": "tight range watch"}
        return {"type": "noop", "lifestyle": 0.45 + j}

    def evaluate(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        t = str(action.get("type", "noop"))
        lv = _f(action, "lifestyle", 0.5) if "lifestyle" in action else 0.5
        fg = _f(state, "fasting_glucose", 120.0)
        score = 0.0
        if 90 <= fg <= 140 and t == "noop":
            score += 0.35
        if t == "noop" and 0.45 <= lv <= 0.7:
            score += 0.2
        if t in {"add", "start", "dose_adjust"} and 95 <= fg <= 115 and str(action.get("drug", "")) == "insulin":
            score -= 0.3
        if t == "noop":
            score += 0.1
        return float(score)
