from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class EpisodeMetrics:
    ep_return: float
    steps: int
    remission: bool
    failure: bool
    mean_hba1c: float
    mean_fpg: float
    side_effect_steps: int
    total_cost_usd: float


def compute_episode_metrics(trajectory_obs: List[Dict[str, Any]], trajectory_info: List[Dict[str, Any]], rewards: List[float]) -> EpisodeMetrics:
    h = [float(o.get("hba1c", 0.0)) for o in trajectory_obs]
    g = [float(o.get("fasting_glucose", 0.0)) for o in trajectory_obs]
    side = sum(int(bool((inf.get("side_effects") or []))) for inf in trajectory_info)
    cost = float(sum(float(inf.get("weekly_cost_usd", 0.0)) for inf in trajectory_info))

    last = trajectory_obs[-1] if trajectory_obs else {}
    remission = bool(float(last.get("hba1c", 99.0)) < 7.0 and float(last.get("fasting_glucose", 999.0)) < 126.0)
    failure = bool(any(bool(inf.get("cvd_event", False)) for inf in trajectory_info)) or bool(float(last.get("fasting_glucose", 0.0)) > 380.0) or bool(float(last.get("egfr", 99.0)) < 12.0)

    return EpisodeMetrics(
        ep_return=float(np.sum(rewards)),
        steps=int(len(rewards)),
        remission=remission,
        failure=failure,
        mean_hba1c=float(np.mean(h)) if h else 0.0,
        mean_fpg=float(np.mean(g)) if g else 0.0,
        side_effect_steps=int(side),
        total_cost_usd=float(cost),
    )


def summarize(metrics: List[EpisodeMetrics]) -> Dict[str, float]:
    if not metrics:
        return {}
    return {
        "episodes": float(len(metrics)),
        "avg_return": float(np.mean([m.ep_return for m in metrics])),
        "avg_steps": float(np.mean([m.steps for m in metrics])),
        "remission_rate": float(np.mean([1.0 if m.remission else 0.0 for m in metrics])),
        "failure_rate": float(np.mean([1.0 if m.failure else 0.0 for m in metrics])),
        "avg_hba1c": float(np.mean([m.mean_hba1c for m in metrics])),
        "avg_fpg": float(np.mean([m.mean_fpg for m in metrics])),
        "avg_side_effect_steps": float(np.mean([m.side_effect_steps for m in metrics])),
        "avg_cost_usd": float(np.mean([m.total_cost_usd for m in metrics])),
    }

