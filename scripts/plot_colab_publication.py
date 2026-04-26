#!/usr/bin/env python3
"""
Publication-style figures from `logs/colab_experiment.json` (notebook export).
No custom line colors; matplotlib defaults only. Uses Agg backend.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def _savefig_multi(fig, path: Path, *, dpi: int, also_svg: bool) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    if also_svg:
        fig.savefig(path.with_suffix(".svg"), format="svg")


def _ma(x: Sequence[float], window: int) -> np.ndarray:
    a = np.array(x, dtype=float)
    if a.size < window or window < 1:
        return a.copy()
    return np.convolve(a, np.ones(window) / window, mode="valid")


def _group_by_model(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        m = str(r.get("model", "unknown"))
        g.setdefault(m, []).append(r)
    for m in g:
        g[m].sort(key=lambda x: int(x.get("episode", 0)))
    return g


def plot_training_curves(
    ax,
    groups: Dict[str, List[Dict[str, Any]]],
    order: List[str],
    ma_window: int,
    label_map: Dict[str, str] | None = None,
) -> None:
    label_map = label_map or {}
    for key in order:
        if key not in groups:
            continue
        rs = [float(x["reward"]) for x in groups[key]]
        ep = [int(x["episode"]) for x in groups[key]]
        y = _ma(rs, min(ma_window, max(1, len(rs) // 3 + 1)))
        x0 = max(0, len(ep) - len(y))
        ax.plot(
            [ep[x0 + i] for i in range(len(y))],
            y,
            linewidth=1.5,
            label=label_map.get(key, key),
        )
    ax.set_xlabel("Episode (training order)")
    ax.set_ylabel("Return (smoothed)" if ma_window > 1 else "Return")
    ax.set_title("Training curves (environment rollouts)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)


def main() -> None:
    plt.switch_backend("Agg")
    p = argparse.ArgumentParser()
    p.add_argument("--in-json", type=Path, default=repo_root / "logs" / "colab_experiment.json")
    p.add_argument("--out-dir", type=Path, default=repo_root / "docs" / "figures")
    p.add_argument("--ma-window", type=int, default=5)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--also-svg",
        action="store_true",
        help="Also write .svg (text) next to .png; safe for git push to Hugging Face.",
    )
    args = p.parse_args()
    also_svg: bool = bool(args.also_svg)
    d = json.loads(args.in_json.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = list(d.get("episodes", []))
    ma_use = int(d.get("ma_window", args.ma_window))
    groups = _group_by_model(rows)
    label_map: Dict[str, str] = {
        "random": "Random baseline",
        "distilgpt2": "Small LM (distilgpt2)",
        "llama-8b-4bit": "Large LM (4-bit)",
        "council_self_repair": "Council + self-repair",
    }
    order = d.get("plot_order", ["random", "distilgpt2", "llama-8b-4bit"])
    bar_order = d.get("bar_models", order)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Primary filename (README / paper); same figure also referenced in Colab
    p1 = args.out_dir / "training_curve.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    plot_training_curves(ax, groups, order, ma_use, label_map=label_map)
    fig.tight_layout()
    _savefig_multi(fig, p1, dpi=args.dpi, also_svg=also_svg)
    plt.close(fig)

    # Bar: mean and std of reward in last K episodes per model
    tail_k = int(d.get("bar_tail_episodes", 20))
    fig2, ax2 = plt.subplots(figsize=(5.0, 4.0))
    means: List[float] = []
    stds: List[float] = []
    labels: List[str] = []
    for key in bar_order:
        if key not in groups:
            continue
        g = [float(x["reward"]) for x in groups[key]]
        tail = g[-tail_k:] if len(g) >= tail_k else g
        if not tail:
            continue
        means.append(float(np.mean(tail)))
        stds.append(float(np.std(tail, ddof=1) if len(tail) > 1 else 0.0))
        labels.append(label_map.get(key, key))
    xb = np.arange(len(labels))
    ax2.bar(xb, means, yerr=stds, capsize=4, alpha=0.85)
    ax2.set_xticks(xb, labels, rotation=12, ha="right")
    ax2.set_ylabel("Mean return (last episodes)")
    ax2.set_title("Final comparison (tail mean ± std)")
    ax2.grid(True, axis="y", alpha=0.35)
    fig2.tight_layout()
    p2 = args.out_dir / "final_comparison_bars.png"
    _savefig_multi(fig2, p2, dpi=args.dpi, also_svg=also_svg)
    plt.close(fig2)

    # Glucose behavior
    gl = d.get("glucose", {})
    p3 = args.out_dir / "behavior_glucose.png"
    if gl.get("random") and gl.get("trained"):
        fig3, ax3 = plt.subplots(figsize=(6.5, 3.5))
        r = gl["random"]
        t = gl["trained"]
        ax3.plot(np.arange(len(r)), r, marker="o", markersize=2, linewidth=1.2, label="Random (1 ep)")
        ax3.plot(np.arange(len(t)), t, marker="o", markersize=2, linewidth=1.2, label="Trained small LM (1 ep)")
        ax3.set_xlabel("Time step (weeks)")
        ax3.set_ylabel("Fasting glucose (simulation)")
        ax3.set_title("Example glucose trajectories (same max_steps)")
        ax3.grid(True, alpha=0.35)
        ax3.legend(loc="best", fontsize=8)
        fig3.tight_layout()
        _savefig_multi(fig3, p3, dpi=args.dpi, also_svg=also_svg)
        plt.close(fig3)
    else:
        p3 = None

    # Self-repair: council rows + vertical lines
    p4 = args.out_dir / "self_repair_episodes.png"
    marks = d.get("self_repair_episodes", [])
    c_rows = [r for r in rows if r.get("model") == "council_self_repair"]
    if c_rows:
        fig4, ax4 = plt.subplots(figsize=(7.0, 3.5))
        epn = [int(x["episode"]) for x in c_rows]
        rew = [float(x["reward"]) for x in c_rows]
        y4 = _ma(rew, min(3, len(rew) or 1))
        off = max(0, len(epn) - len(y4))
        ax4.plot([epn[off + i] for i in range(len(y4))], y4, linewidth=1.4, label="Episode return (MA)")
        for m in marks:
            ax4.axvline(m, linestyle="--", linewidth=0.9, alpha=0.7)
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Return")
        ax4.set_title("Council + self-repair (dashed = repair signal)")
        ax4.grid(True, alpha=0.35)
        ax4.legend(loc="best", fontsize=8)
        fig4.tight_layout()
        _savefig_multi(fig4, p4, dpi=args.dpi, also_svg=also_svg)
        plt.close(fig4)
    else:
        p4 = None

    print("wrote", p1, p2, p3, p4)


if __name__ == "__main__":
    main()
