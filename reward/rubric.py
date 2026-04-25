from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class RewardResult:
    total: float
    components: Dict[str, float]
    terminal: bool = False


class RewardRubric:
    """
    Multi-component reward:
      - clinical improvement (HbA1c, FPG)
      - biomarker trends (dense shaping)
      - side effects penalty (severity-weighted)
      - cost penalty
      - terminal reward (remission / failure)
    """

    def compute(self, *, prev_state: Dict[str, Any], next_state: Dict[str, Any], info: Dict[str, Any]) -> RewardResult:
        p = prev_state
        n = next_state

        # Extract
        h0 = float(p.get("hba1c", 8.0))
        h1 = float(n.get("hba1c", 8.0))
        g0 = float(p.get("fasting_glucose", 160.0))
        g1 = float(n.get("fasting_glucose", 160.0))
        egfr1 = float(n.get("egfr", 90.0))
        bmi0 = float(p.get("bmi", 30.0))
        bmi1 = float(n.get("bmi", 30.0))

        side_effects: List[Dict[str, Any]] = list(info.get("side_effects") or [])
        weekly_cost = float(info.get("weekly_cost_usd", 0.0))

        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))

        # 1) clinical improvement: reward reductions in HbA1c/FPG
        dh = h0 - h1
        dg = (g0 - g1) / 50.0
        r_clin = 2.5 * dh + 1.2 * dg

        # 2) biomarker shaping: distance to target zone + trend bonus
        # target HbA1c ~6.5-7; target FPG ~ 90-110
        dist_h = abs(h1 - 6.6)
        dist_g = abs(g1 - 105.0) / 55.0
        r_trend = -0.45 * dist_h - 0.35 * dist_g
        # small BMI encouragement
        r_trend += 0.05 * (bmi0 - bmi1)

        # 3) side effects penalty: severity-weighted
        se_pen = 0.0
        for ev in side_effects:
            sev = float(ev.get("severity", 1))
            # hypoglycemia is more severe in this task
            if str(ev.get("label", "")) == "hypoglycemia":
                se_pen -= 0.9 * sev
            else:
                se_pen -= 0.35 * sev

        # 4) cost penalty (soft)
        r_cost = -0.02 * weekly_cost

        # 5) terminal rewards / failures
        terminal = False
        r_terminal = 0.0
        remission = (h1 < 7.0) and (g1 < 126.0)
        failure = (egfr1 < 12.0) or (g1 > 380.0) or bool(info.get("cvd_event", False))

        if failure:
            terminal = True
            r_terminal -= 15.0
        elif remission and (bool(truncated) or bool(terminated) or int(n.get("week", 0)) >= 12):
            # sustained-ish control by week 12+ gets a terminal bonus
            terminal = True
            r_terminal += 12.0

        total = float(r_clin + r_trend + se_pen + r_cost + r_terminal)
        comps = {
            "clinical": float(r_clin),
            "trend": float(r_trend),
            "side_effects": float(se_pen),
            "cost": float(r_cost),
            "terminal": float(r_terminal),
        }
        return RewardResult(total=total, components=comps, terminal=terminal)

