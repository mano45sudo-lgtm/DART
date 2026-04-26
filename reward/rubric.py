from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RewardResult:
    total: float
    components: Dict[str, float]
    terminal: bool = False


@dataclass
class RewardConfig:
    """
    Tunable terms (all >= 0 where applicable; signs applied in compute()).
    Keep scales modest so the twin remains trainable on CPU.
    """

    w_glucose_improve: float = 1.8  # step delta FPG (scaled)
    w_hba1c_improve: float = 2.2
    w_hypo: float = 1.6  # low glucose + hypos in side_effects
    w_instability: float = 0.22  # (Δg)^2 and acceleration
    w_accel: float = 0.12
    w_cost: float = 0.02
    w_inactivity: float = 0.45  # noop streak
    w_repeat: float = 0.2  # same action signature as last step
    w_time: float = 0.04  # mild pressure to learn early (per week index)
    w_hyper_cumulative: float = 0.11  # exposure to high glucose
    w_tool: float = 0.35  # safety / forced interventions (e.g. fall path)
    tool_free: int = 1  # allow one free safety episode before penalizing
    hypoglycemia_fpg: float = 70.0
    target_g: float = 105.0
    target_hba1c: float = 6.6


@dataclass
class RewardRubric:
    """
    Decomposed, anti-hacking-friendly reward (dense + terminal).

    - Rewards improvement in glucose/HbA1c at each step (not just terminal).
    - Penalizes hypos, glucose swings, no-op stalling, repeated actions, cost,
      cumulative hyperglycemia time-in-range damage, and excess safety tooling.
    """

    config: RewardConfig = field(default_factory=RewardConfig)

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config or RewardConfig()

    def compute(self, *, prev_state: Dict[str, Any], next_state: Dict[str, Any], info: Dict[str, Any]) -> RewardResult:
        c = self.config
        p = prev_state
        n = next_state
        wk = int(n.get("week", 0) or 0)

        h0, h1 = float(p.get("hba1c", 8.0)), float(n.get("hba1c", 8.0))
        g0, g1 = float(p.get("fasting_glucose", 160.0)), float(n.get("fasting_glucose", 160.0))
        egfr1 = float(n.get("egfr", 90.0))
        bmi0, bmi1 = float(p.get("bmi", 30.0)), float(n.get("bmi", 30.0))

        dg = (g0 - g1) / 40.0
        dh = h0 - h1
        r_glu = c.w_glucose_improve * max(0.0, dg) + c.w_hba1c_improve * max(0.0, dh)
        r_glu -= 0.15 * max(0.0, -dg) * c.w_glucose_improve
        r_glu -= 0.2 * max(0.0, -dh) * c.w_hba1c_improve

        dist_h = abs(h1 - c.target_hba1c)
        dist_g = abs(g1 - c.target_g) / 50.0
        r_trend = -0.4 * dist_h - 0.3 * dist_g
        r_trend += 0.06 * (bmi0 - bmi1)

        hypo_pen = 0.0
        if g1 < c.hypoglycemia_fpg:
            hypo_pen += c.w_hypo * (c.hypoglycemia_fpg - g1) / 25.0
        side_effects: List[Dict[str, Any]] = list(info.get("side_effects") or [])
        se_non_hypo = 0.0
        for ev in side_effects:
            sev = float(ev.get("severity", 1))
            if str(ev.get("label", "")) == "hypoglycemia":
                hypo_pen += c.w_hypo * 0.5 * sev
            else:
                se_non_hypo -= 0.3 * sev

        dg_step = g1 - g0
        dd = float(info.get("dd_glucose", 0.0))
        r_inst = -c.w_instability * (dg_step**2) / 900.0
        r_inst -= c.w_accel * (dd**2) / 900.0

        weekly_cost = float(info.get("weekly_cost_usd", 0.0))
        r_cost = -c.w_cost * weekly_cost

        noop_s = int(info.get("noop_streak", 0) or 0)
        r_inact = -c.w_inactivity * max(0, noop_s - 1) * 0.2
        if int(info.get("action_repeat", 0) or 0):
            r_inact -= c.w_repeat

        r_time = -c.w_time * (wk**0.5) * 0.15

        hyper_cum = float(info.get("hyper_cumulative", 0.0) or 0.0)
        r_cum = -c.w_hyper_cumulative * min(hyper_cum, 24.0) / 12.0

        ivc = int(info.get("intervention_count", 0) or 0)
        r_tool = 0.0
        if ivc > c.tool_free:
            r_tool = -c.w_tool * (ivc - c.tool_free) ** 0.7

        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        terminal = False
        r_terminal = 0.0
        remission = (h1 < 7.0) and (g1 < 126.0)
        failure = (egfr1 < 12.0) or (g1 > 380.0) or bool(info.get("cvd_event", False))
        if failure:
            terminal = True
            r_terminal -= 15.0
        elif remission and (bool(truncated) or bool(terminated) or wk >= 12):
            terminal = True
            r_terminal += 12.0

        total = float(
            r_glu
            + r_trend
            - hypo_pen
            + r_inst
            + r_cost
            + r_inact
            + r_time
            + r_cum
            + r_tool
            + r_terminal
            + se_non_hypo
        )
        comps: Dict[str, float] = {
            "glucose_improvement": float(r_glu),
            "trend_shaping": float(r_trend),
            "hypoglycemia_penalty": float(-hypo_pen),
            "side_effects": float(se_non_hypo),
            "instability": float(r_inst),
            "treatment_cost": float(r_cost),
            "inactivity": float(r_inact),
            "time_decay": float(r_time),
            "cumulative_hyper": float(r_cum),
            "tool_intervention": float(r_tool),
            "terminal": float(r_terminal),
        }
        return RewardResult(total=total, components=comps, terminal=terminal)
