from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class TrialOutcome:
    mean_reward: float
    mean_hba1c: float
    side_effect_rate: float
    samples: int


def simulate_trial(
    *,
    env_factory,
    candidate_actions: List[Dict[str, Any]],
    horizon_weeks: int = 12,
    n_rollouts: int = 8,
    seed: int = 0,
) -> Dict[str, TrialOutcome]:
    """
    Inputs:
      - env_factory: callable -> fresh env
      - candidate_actions: list of fixed actions applied every step (mock)
    Outputs:
      - per-action aggregate outcomes

    Mock logic: uses environment step() and reads `reward` (Phase 5 will make real).
    """
    rng = np.random.default_rng(seed)
    results: Dict[str, TrialOutcome] = {}

    for a in candidate_actions:
        key = str(a)
        rewards: List[float] = []
        hba1cs: List[float] = []
        se: List[float] = []
        for _ in range(n_rollouts):
            env = env_factory()
            obs, _info = env.reset(seed=int(rng.integers(0, 1_000_000)))
            se_events = 0
            for _t in range(horizon_weeks):
                obs, r, term, trunc, info = env.step(a)
                rewards.append(float(r))
                hba1cs.append(float(obs.get("hba1c", 0.0)))
                se_events += int(bool((info.get("side_effects") or [])))
                if term or trunc:
                    break
            se.append(se_events / max(1, horizon_weeks))

        results[key] = TrialOutcome(
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            mean_hba1c=float(np.mean(hba1cs)) if hba1cs else 0.0,
            side_effect_rate=float(np.mean(se)) if se else 0.0,
            samples=n_rollouts,
        )

    return results

