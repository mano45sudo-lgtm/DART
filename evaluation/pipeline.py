"""
Unified evaluation: mean / std return and optional action + glucose trajectories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from env.digital_twin_env import DigitalTwinDiabetesEnv  # noqa: E402
from evaluation.eval_metrics import compute_episode_metrics, summarize  # noqa: E402


@dataclass
class Trajectory:
    step_rewards: List[float]
    actions: List[Dict[str, Any]]
    fasting_glucose: List[float]
    action_types: List[str]


def evaluate(
    agent: Any,
    *,
    n_episodes: int = 20,
    seed: int = 0,
    max_steps: int = 24,
    capture_trajectories: bool = False,
) -> Dict[str, Any]:
    """
    Run `n_episodes` in fresh envs. Agent must have `.act(obs) -> action_dict`.
    """
    act_fn = getattr(agent, "act", None)
    if act_fn is None:
        raise TypeError("agent must implement .act(obs)")

    ep_returns: List[float] = []
    metrics_list = []
    trajs: List[Trajectory] = []
    for ep in range(n_episodes):
        env = DigitalTwinDiabetesEnv(seed=seed + ep, max_steps=max_steps)
        obs, _ = env.reset(seed=seed + ep)
        traj_obs = [obs]
        traj_info: List[Dict[str, Any]] = []
        rewards: List[float] = []
        actions: List[Dict[str, Any]] = []
        glucose: List[float] = [float(obs["fasting_glucose"])]
        types: List[str] = []
        done = False
        while not done:
            a = act_fn(obs)
            obs, r, term, trunc, info = env.step(a)
            traj_obs.append(obs)
            traj_info.append(info)
            rewards.append(float(r))
            if capture_trajectories:
                act_clean = info.get("action", a)
                actions.append(dict(act_clean) if isinstance(act_clean, dict) else {"type": "unknown"})
                glucose.append(float(obs["fasting_glucose"]))
                types.append(str(act_clean.get("type", a.get("type", "noop")) if isinstance(act_clean, dict) else "noop"))
            done = bool(term or trunc)
        er = float(np.sum(rewards))
        ep_returns.append(er)
        metrics_list.append(compute_episode_metrics(traj_obs, traj_info, rewards))
        if capture_trajectories:
            trajs.append(
                Trajectory(step_rewards=list(rewards), actions=actions, fasting_glucose=glucose, action_types=types)
            )
    m = float(np.mean(ep_returns))
    s = float(np.std(ep_returns, ddof=1) if len(ep_returns) > 1 else 0.0)
    out: Dict[str, Any] = {
        "mean_reward": m,
        "std_reward": s,
        "n_episodes": n_episodes,
        "per_episode_return": [float(x) for x in ep_returns],
        "summary": summarize(metrics_list),
    }
    if capture_trajectories and trajs:
        out["trajectories"] = [asdict(t) for t in trajs]
    return out


def compare_random_rule_trained(
    *,
    n_episodes: int = 20,
    seed: int = 0,
    max_steps: int = 24,
) -> Dict[str, Any]:
    """Quick comparison using built-in baselines (random + rule) only."""
    from evaluation.baseline_random_agent import RandomAgent
    from evaluation.baseline_rule_agent import RuleBasedAgent

    r = evaluate(RandomAgent(seed=seed + 1), n_episodes=n_episodes, seed=seed, max_steps=max_steps, capture_trajectories=True)
    u = evaluate(RuleBasedAgent(), n_episodes=n_episodes, seed=seed + 7, max_steps=max_steps, capture_trajectories=True)
    return {
        "random": r,
        "rule": u,
    }
