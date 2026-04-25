from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from env.digital_twin_env import DigitalTwinDiabetesEnv  # noqa: E402
from evaluation.baseline_random_agent import RandomAgent  # noqa: E402


def rollout(agent, *, episodes: int = 30, seed: int = 0):
    returns = []
    for ep in range(episodes):
        env = DigitalTwinDiabetesEnv(seed=seed + ep, max_steps=24)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        G = 0.0
        while not done:
            a = agent.act(obs)
            obs, r, term, trunc, _info = env.step(a)
            G += float(r)
            done = bool(term or trunc)
        returns.append(G)
    return np.array(returns, dtype=float)


def main():
    out_dir = repo_root / "logs"
    out_dir.mkdir(exist_ok=True)

    # Avoid blocking in headless runs (do not show by default)
    plt.switch_backend("Agg")

    agent = RandomAgent(seed=0)
    rets = rollout(agent, episodes=60, seed=0)

    plt.figure(figsize=(9, 3.5))
    plt.plot(np.arange(1, len(rets) + 1), rets, alpha=0.75, linewidth=1.0, color="#64748b")
    plt.title("Random baseline: episode returns (DigitalTwinDiabetesEnv rollouts)")
    plt.xlabel("Episode index (sequential rollouts, 1 … N)")
    plt.ylabel("Episode return (sum of weekly step rewards; same units as training plots)")
    plt.grid(True, alpha=0.35)
    png = out_dir / "baseline_returns.png"
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    plt.close()

    summary = {
        "episodes": int(len(rets)),
        "avg_return": float(np.mean(rets)),
        "std_return": float(np.std(rets)),
        "min_return": float(np.min(rets)),
        "max_return": float(np.max(rets)),
    }
    (out_dir / "baseline_returns.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote:", str(png))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

