#!/usr/bin/env python3
"""
From evaluation_baseline.json: bar (random vs rule) + one glucose trace.
From training_last.json (optional): overlay training/eval means on a second figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", type=Path, help="Output of scripts/evaluate_agents.py")
    p.add_argument("--training-json", type=Path, help="Optional logs/training_last.json")
    p.add_argument("--out-baseline", type=Path, default=repo_root / "docs" / "figures" / "validation_baselines.png")
    p.add_argument("--out-training", type=Path, default=repo_root / "docs" / "figures" / "validation_training_curve.png")
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()
    if not args.eval_json and not args.training_json:
        p.error("pass --eval-json and/or --training-json")

    if args.eval_json and args.eval_json.is_file():
        d = json.loads(args.eval_json.read_text(encoding="utf-8"))
        args.out_baseline.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        r_m = d.get("random", {}).get("mean_reward", 0.0)
        u_m = d.get("rule", {}).get("mean_reward", 0.0)
        r_s = d.get("random", {}).get("std_reward", 0.0)
        u_s = d.get("rule", {}).get("std_reward", 0.0)
        axes[0].bar(
            [0, 1],
            [r_m, u_m],
            yerr=[r_s, u_s],
            capsize=4,
            color=["#94a3b8", "#22c55e"],
            edgecolor="white",
        )
        axes[0].set_xticks([0, 1], ["Random", "Rule"], rotation=0)
        axes[0].set_ylabel("Mean episode return")
        axes[0].set_title("Baselines (identical env protocol)")
        axes[0].grid(True, axis="y", alpha=0.35)
        trs = d.get("rule", {}).get("trajectories", [])
        if trs and trs[0].get("fasting_glucose"):
            g = trs[0]["fasting_glucose"]
            axes[1].plot(np.arange(len(g)), g, "o-", color="#16a34a", linewidth=1.2, markersize=3)
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Fasting glucose (rule, ep 0)")
        else:
            axes[1].text(0.5, 0.5, "no trajectories in JSON", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Sample glucose path")
        axes[1].grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(args.out_baseline, dpi=args.dpi)
        plt.close(fig)
        print("wrote", args.out_baseline)

    if args.training_json and args.training_json.is_file():
        tj = json.loads(args.training_json.read_text(encoding="utf-8"))
        tr = tj.get("training", {})
        uidx = np.array(tr.get("update_index", []), dtype=float)
        train_m = np.array(tr.get("train_mean_episode_return", []), dtype=float)
        ev_x = np.array(tr.get("eval_at_update", []), dtype=float)
        ev_m = np.array(tr.get("eval_llm_mean_return", []), dtype=float)
        br = tj.get("baseline_random", {}).get("avg_return", np.nan)
        args.out_training.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if uidx.size and train_m.size:
            ax.plot(uidx, train_m, color="#2563eb", label="Train batch mean return", linewidth=1.6, alpha=0.85)
        if ev_x.size and ev_m.size:
            ev_s = list(tr.get("eval_llm_std_return", []))
            if len(ev_s) != len(ev_m):
                yerr = None
            else:
                yerr = np.array(ev_s, dtype=float)
            ax.errorbar(
                ev_x,
                ev_m,
                yerr=yerr,
                fmt="o-",
                color="#ea580c",
                capsize=3,
                label="Eval mean (held-out seeds)",
            )
        if not np.isnan(br):
            ax.axhline(br, color="#64748b", linestyle="--", label=f"Random baseline ≈{br:.1f}", linewidth=1.2)
        ax.set_xlabel("Update index")
        ax.set_ylabel("Return")
        ax.set_title("Training validation (REINFORCE run)")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out_training, dpi=args.dpi)
        plt.close(fig)
        print("wrote", args.out_training)


if __name__ == "__main__":
    main()
