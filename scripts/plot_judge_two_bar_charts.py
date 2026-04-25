#!/usr/bin/env python3
"""
Two judge-facing bar charts from REINFORCE logs (same protocol in each JSON):

  1) Random baseline vs trained small LM (e.g. DistilGPT2)
  2) Random baseline vs trained Unsloth / large LM

Baseline bars use `final_eval.random` from the **distil** JSON so both charts
share the same random-policy reference when schedules match.

Example:
  python scripts/plot_judge_two_bar_charts.py \\
    --distil-json logs/training_last_distil.json \\
    --unsloth-json logs/training_last_unsloth.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _one_bar(ax, labels: list, means: list, stds: list, title: str, ylabel: str) -> None:
    x = np.arange(len(labels))
    colors = ["#64748b", "#ea580c"]
    ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="white", linewidth=0.8, alpha=0.92)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.35)


def main() -> None:
    plt.switch_backend("Agg")
    p = argparse.ArgumentParser()
    p.add_argument("--distil-json", type=Path, default=repo_root / "logs" / "training_last_distil.json")
    p.add_argument("--unsloth-json", type=Path, required=True)
    p.add_argument("--small-bar-label", default=None, help="Short name under bar 2 in first figure (default: model id)")
    p.add_argument("--large-bar-label", default=None, help="Short name under bar 2 in second figure (default: model id)")
    p.add_argument("--out-dir", type=Path, default=repo_root / "docs" / "figures")
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()

    if not args.distil_json.is_file():
        p.error(f"missing {args.distil_json}")
    if not args.unsloth_json.is_file():
        p.error(f"missing {args.unsloth_json}")

    d = _load(args.distil_json)
    u = _load(args.unsloth_json)

    br = float(d["final_eval"]["random"]["avg_return"])
    br_std = float(d["final_eval"]["random"].get("std_return", 0.0))
    n_rand = int(d["final_eval"]["random"].get("episodes", 0))

    dm = float(d["final_eval"]["llm"]["mean_return"])
    ds = float(d["final_eval"]["llm"].get("std_return", 0.0))
    d_model = d.get("protocol", {}).get("model", "small LM")
    d_bar = args.small_bar_label or d_model.split("/")[-1][:22]

    um = float(u["final_eval"]["llm"]["mean_return"])
    us = float(u["final_eval"]["llm"].get("std_return", 0.0))
    u_model = u.get("protocol", {}).get("model", "large LM")
    u_bar = args.large_bar_label or u_model.split("/")[-1][:22]

    n_eval_d = int(d["final_eval"]["llm"].get("eval_episodes", 0))
    n_eval_u = int(u["final_eval"]["llm"].get("eval_episodes", 0))

    ylabel = f"Mean episode return ± std (DigitalTwinDiabetesEnv; random n={n_rand})"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p1 = args.out_dir / "judge_bar_baseline_vs_small.png"
    p2 = args.out_dir / "judge_bar_baseline_vs_unsloth.png"

    fig1, ax1 = plt.subplots(figsize=(5.5, 4.2))
    _one_bar(
        ax1,
        ["Random\n(baseline)", f"Trained\n{d_bar}"],
        [br, dm],
        [br_std, ds],
        "Baseline vs small LM (same held-out eval seeds)",
        ylabel,
    )
    fig1.tight_layout()
    fig1.savefig(p1, dpi=args.dpi)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(5.5, 4.2))
    _one_bar(
        ax2,
        ["Random\n(baseline)", f"Trained\n{u_bar}"],
        [br, um],
        [br_std, us],
        "Baseline vs Unsloth / large LM (same protocol)",
        ylabel,
    )
    fig2.tight_layout()
    fig2.savefig(p2, dpi=args.dpi)
    plt.close(fig2)

    print("=== Numbers for judges (from logs) ===")
    print(f"Random baseline mean return:     {br:.4f} ± {br_std:.4f}")
    print(f"Small LM ({d_model}) final:      {dm:.4f} ± {ds:.4f}  (eval episodes={n_eval_d})")
    print(f"Large LM ({u_model}) final:       {um:.4f} ± {us:.4f}  (eval episodes={n_eval_u})")
    print("wrote:", p1.resolve())
    print("wrote:", p2.resolve())


if __name__ == "__main__":
    main()
