"""Council + self-improvement rollouts: no LLM, rich logging for demos."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

from env.digital_twin_env import DigitalTwinDiabetesEnv
from council import Council, build_default_council


def run_council_episode(
    env_seed: int,
    max_steps: int,
    council: Council,
    controller,  # SelfImprovementController
    *,
    log: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    env = DigitalTwinDiabetesEnv(seed=env_seed, max_steps=max_steps)
    obs, _ = env.reset(seed=env_seed)
    total = 0.0
    glucose_traj: List[float] = [float(obs["fasting_glucose"])]
    steps_log: List[Dict[str, Any]] = []
    done = False
    while not done:
        ex: Dict[str, Any] = council.decide(
            obs,
            env,
            use_fallback=bool(controller.use_fallback),
            exploration=float(controller.exploration),
        )
        action = ex["final_action"]
        obs, r, term, trunc, _info = env.step(action)
        total += float(r)
        glucose_traj.append(float(obs["fasting_glucose"]))
        done = bool(term or trunc)
        if log:
            steps_log.append(
                {
                    "step_return": r,
                    "council": {
                        "final_action": ex.get("final_action"),
                        "agent_proposals": ex.get("agent_proposals"),
                        "per_action_scores": ex.get("per_action_scores"),
                        "weights_used": ex.get("weights_used"),
                        "use_fallback": ex.get("use_fallback"),
                        "agent_votes": ex.get("agent_votes"),
                        "reasoning": ex.get("reasoning"),
                    },
                }
            )
    g_std = float(pstdev(glucose_traj)) if len(glucose_traj) > 1 else 0.0
    return total, {
        "steps": steps_log,
        "ep_return": total,
        "glucose_fasting_stdev": g_std,
    }


def run_council_training(
    *,
    max_steps: int,
    updates: int,
    episodes_per_update: int,
    train_seed_base: int,
    out_path,
    log_steps: int = 2,
) -> None:
    """Many short council + self-improvement loops; no transformers (CPU-friendly)."""
    import json
    from pathlib import Path

    from self_improvement import SelfImprovementController

    council = build_default_council(seed=42)
    ctrl = SelfImprovementController()
    train_rets: list[float] = []
    by_update: list[dict] = []
    for u in range(1, updates + 1):
        batch: list[float] = []
        gstds: list[float] = []
        for e in range(episodes_per_update):
            seed = train_seed_base + u * 1000 + e * 19
            total, detail = run_council_episode(
                seed, max_steps, council, ctrl, log=(e < log_steps and u <= 2)
            )
            gstds.append(detail["glucose_fasting_stdev"])
            batch.append(total)
        ctrl.update(float(sum(batch) / len(batch)), last_episode_fasting_std=float(max(gstds) if gstds else 0.0))
        ctrl.adjust_council(council)
        ctrl.adjust_exploration()
        train_rets.append(float(mean(batch)))
        by_update.append(
            {
                "update": u,
                "mean_episode_return": train_rets[-1],
                "self_improvement": ctrl.snapshot(),
                "council_weights": {**council.weights},
            }
        )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "mode": "council_rule",
                "protocol": {
                    "max_steps": max_steps,
                    "updates": updates,
                    "episodes_per_update": episodes_per_update,
                },
                "by_update": by_update,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("council run wrote", p)
