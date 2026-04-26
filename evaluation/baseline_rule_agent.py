"""Simple glucocentric rule policy (valid JSON actions, no LLM)."""

from __future__ import annotations

from typing import Any, Dict


def _f(obs: Dict[str, Any], k: str, default: float) -> float:
    return float(obs.get(k, default))


class RuleBasedAgent:
    name = "rule"

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        fg = _f(obs, "fasting_glucose", 150.0)
        hba1c = _f(obs, "hba1c", 8.0)
        egfr = _f(obs, "egfr", 90.0)
        if fg < 80:
            return {"type": "noop", "lifestyle": 0.65}
        if fg < 100:
            if hba1c < 6.5:
                return {"type": "noop", "lifestyle": 0.55}
            return {"type": "dose_adjust", "drug": "insulin", "dose": 0.2, "lifestyle": 0.55}
        if fg > 250 or hba1c > 9.0:
            if egfr < 30:
                return {"type": "dose_adjust", "drug": "insulin", "dose": 0.45, "lifestyle": 0.55}
            if fg > 280 or hba1c > 9.5:
                return {"type": "start", "drug": "metformin", "dose": 0.88, "lifestyle": 0.6}
            return {"type": "start", "drug": "metformin", "dose": 0.85, "lifestyle": 0.58}
        if fg > 160 or hba1c > 7.5:
            if egfr >= 30:
                return {"type": "start", "drug": "metformin", "dose": 0.8, "lifestyle": 0.6}
            return {"type": "add", "drug": "glp1", "dose": 0.7, "lifestyle": 0.55}
        if fg > 130:
            return {"type": "dose_adjust", "drug": "metformin", "dose": 0.75, "lifestyle": 0.5}
        return {"type": "noop", "lifestyle": 0.45}
