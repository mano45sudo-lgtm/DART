#!/usr/bin/env python3
"""
Overlay several `training_last*.json` logs (one per trained model run) for judges.

Each run should use the **same** `--eval-seed-base` / `--eval-seeds` / `max_steps` so curves are comparable.

Example:
  python scripts/train_reinforce_twin.py --judge-schedule --model distilgpt2 \\
      --out-json logs/training_last_distil.json
  python scripts/train_reinforce_twin.py --judge-schedule --load-in-4bit \\
      --model unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit \\
      --out-json logs/training_last_8b.json
  python scripts/plot_compare_training_runs.py \\
      --run logs/training_last_distil.json:DistilGPT2 \\
      --run logs/training_last_8b.json:8B-Instruct-4bit \\
      --out docs/figures/compare_small_vs_8b.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def _parse_run(s: str) -> Tuple[Path, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"Expected path:Label, got: {s!r}")
    path_s, label = s.split(":", 1)
    path = Path(path_s)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Missing file: {path}")
    return path, label.strip() or path.name


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    plt.switch_backend("Agg")
    p = argparse.ArgumentParser(description="Compare multiple training JSON logs on one figure.")
    p.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        type=_parse_run,
        metavar="PATH:Label",
        help="Repeat for each model (e.g. logs/a.json:DistilGPT2)",
    )
    p.add_argument("--out", type=Path, default=repo_root / "docs" / "figures" / "compare_models.png")
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()

    runs: List[Tuple[Path, str, Dict[str, Any]]] = [(path, lab, _load(path)) for path, lab in args.runs]

    # Baseline: final_eval.random from first run (runs should match eval protocol)
    br = runs[0][2]["final_eval"]["random"]["avg_return"]
    br_std = runs[0][2]["final_eval"]["random"].get("std_return", 0.0)
    n_rand = runs[0][2]["final_eval"]["random"].get("episodes", 0)

    colors = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#ca8a04"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))

    ax1.axhline(br, color="#64748b", linestyle="--", linewidth=1.6, label=f"Random baseline mean ({br:.2f})")
    if br_std:
        ax1.axhspan(br - br_std, br + br_std, color="#64748b", alpha=0.12, label="Random ±1 std")

    for i, (_path, label, payload) in enumerate(runs):
        c = colors[i % len(colors)]
        model_id = payload.get("protocol", {}).get("model", label)
        t = payload["training"]
        ex = np.array(t["eval_at_update"], dtype=int)
        ey = np.array(t["eval_llm_mean_return"], dtype=float)
        es = np.array(t["eval_llm_std_return"], dtype=float)
        ax1.errorbar(ex, ey, yerr=es, fmt="o-", color=c, capsize=3, linewidth=1.4, markersize=4, label=f"{label}\n({model_id})")

    ax1.set_xlabel("REINFORCE update index (on-policy rollouts in DigitalTwinDiabetesEnv)")
    ax1.set_ylabel("Held-out eval: mean episode return ± std (same eval seeds across runs)")
    ax1.set_title("Learning trend: small vs large LM (eval checkpoints)")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best", fontsize=7)

    # Bars: random + each trained final
    labels_b = ["Random\n(baseline)"]
    means = [br]
    stds = [br_std if br_std else 0.0]
    bar_colors = ["#64748b"]

    for i, (_path, label, payload) in enumerate(runs):
        fe = payload["final_eval"]["llm"]
        labels_b.append(label.replace(" ", "\n"))
        means.append(float(fe["mean_return"]))
        stds.append(float(fe.get("std_return", 0.0)))
        bar_colors.append(colors[i % len(colors)])

    x = np.arange(len(labels_b))
    ax2.bar(x, means, yerr=stds, capsize=5, color=bar_colors, edgecolor="white", linewidth=0.7, alpha=0.9)
    ax2.set_xticks(x, labels_b)
    ax2.set_xlabel("Agent")
    ax2.set_ylabel(f"Mean episode return ± std (final held-out eval; random n≈{n_rand})")
    ax2.set_title("Final quantitative comparison")
    ax2.grid(True, axis="y", alpha=0.35)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print("wrote", args.out.resolve())


if __name__ == "__main__":
    main()
