from __future__ import annotations

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from env.digital_twin_env import DigitalTwinDiabetesEnv  # noqa: E402
from evaluation.baseline_random_agent import RandomAgent  # noqa: E402
from evaluation.eval_metrics import compute_episode_metrics, summarize  # noqa: E402


def run(agent, *, episodes: int = 50, seed: int = 0):
    metrics = []
    for ep in range(episodes):
        env = DigitalTwinDiabetesEnv(seed=seed + ep, max_steps=24)
        obs, _ = env.reset(seed=seed + ep)
        traj_obs = [obs]
        traj_info = []
        rewards = []
        done = False
        while not done:
            a = agent.act(obs)
            obs, r, term, trunc, info = env.step(a)
            traj_obs.append(obs)
            traj_info.append(info)
            rewards.append(float(r))
            done = bool(term or trunc)
        metrics.append(compute_episode_metrics(traj_obs, traj_info, rewards))
    return metrics


def main():
    agent = RandomAgent(seed=0)
    metrics = run(agent, episodes=40, seed=0)
    print(json.dumps(summarize(metrics), indent=2))


if __name__ == "__main__":
    main()

