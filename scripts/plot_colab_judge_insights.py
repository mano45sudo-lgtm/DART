#!/usr/bin/env python3
"""
Rich judge-facing figures from `logs/colab_experiment.json` (requires `traces` + `endpoints` keys).

Expects the notebook to save per-policy clinical traces and multi-episode endpoint rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def _weeks_for_steps(trace: Dict[str, Any]) -> Optional[np.ndarray]:
    w = trace.get("week", [])
    if not w or len(w) < 2:
        return None
    return np.array(w[1:], dtype=int)


def _plot_clinical_panels(
    axs: np.ndarray,
    traces: Dict[str, Dict[str, Any]],
    order: List[str],
    label_map: Dict[str, str],
) -> None:
    keys = [("fasting_glucose", "Fasting glucose (mg/dL)"), ("hba1c", "HbA1c (%)"), ("egfr", "eGFR"), ("weekly_cost_usd", "Weekly cost (USD)")]
    for j, (field, yl) in enumerate(keys):
        ax = axs.flat[j]
        for k in order:
            tr = traces.get(k)
            if not tr or not tr.get("week"):
                continue
            w = np.array(tr["week"], dtype=int)
            y = np.array(tr.get(field) or [], dtype=float)
            if field == "weekly_cost_usd" and len(y) < len(w):
                y = np.concatenate([np.array([0.0]), y])
            m = min(len(w), len(y))
            if m < 1:
                continue
            ax.plot(
                w[:m],
                y[:m],
                marker="o",
                markersize=2,
                linewidth=1.3,
                label=label_map.get(k, k),
            )
        ax.set_xlabel("Simulated week (env)")
        ax.set_ylabel(yl)
        ax.set_title(yl.split(" (")[0])
        ax.grid(True, alpha=0.35)
        if j == 0:
            ax.legend(loc="best", fontsize=7)


def _plot_reward_dynamics(
    axa,
    axb,
    traces: Dict[str, Dict[str, Any]],
    order: List[str],
    label_map: Dict[str, str],
) -> None:
    for k in order:
        tr = traces.get(k)
        if not tr or not tr.get("step_reward"):
            continue
        wx = _weeks_for_steps(tr)
        if wx is None:
            continue
        m = min(len(wx), len(tr["step_reward"]))
        if m < 1:
            continue
        r = np.array(tr["step_reward"][:m], dtype=float)
        c = np.array(tr["cumulative_return"][:m], dtype=float)
        axa.plot(
            wx[:m],
            r,
            marker="o",
            markersize=1.5,
            linewidth=1.1,
            label=label_map.get(k, k),
        )
        axb.plot(wx[:m], c, marker="o", markersize=1.5, linewidth=1.1, label=label_map.get(k, k))
    axa.set_xlabel("Simulated week (after each step)")
    axa.set_ylabel("Per-step return")
    axa.set_title("Dense reward (rubric sum each week)")
    axa.grid(True, alpha=0.35)
    axa.legend(loc="best", fontsize=7)
    axb.set_xlabel("Simulated week (after each step)")
    axb.set_ylabel("Cumulative return")
    axb.set_title("Cumulative return (same example episode per policy)")
    axb.grid(True, alpha=0.35)
    axb.legend(loc="best", fontsize=7)


def _action_frequencies(traces: Dict[str, Dict[str, Any]], order: List[str]) -> Tuple[List[str], np.ndarray]:
    all_types: set[str] = set()
    for k in order:
        tr = traces.get(k) or {}
        for t in tr.get("action_type") or []:
            all_types.add(str(t))
    labels = sorted(all_types) if all_types else ["noop"]
    mat = np.zeros((len(order), len(labels)))
    for i, k in enumerate(order):
        tr = traces.get(k) or {}
        c = Counter(str(t) for t in (tr.get("action_type") or []))
        for j, lab in enumerate(labels):
            mat[i, j] = float(c.get(lab, 0))
    return labels, mat


def _sum_components(trace: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not trace:
        return {}
    parts: Dict[str, float] = {}
    for d in trace.get("reward_components") or []:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if isinstance(v, (int, float)):
                parts[k] = parts.get(k, 0.0) + float(v)
    return parts


def _plot_rubric_sums(
    ax,
    traces: Dict[str, Dict[str, Any]],
    order: List[str],
    label_map: Dict[str, str],
) -> None:
    per_model: List[Dict[str, float]] = []
    for k in order:
        per_model.append(_sum_components(traces.get(k)))
    all_k = set()
    for p in per_model:
        all_k |= set(p.keys())
    all_k = sorted(all_k) if all_k else list(RewardRubricKeys.ALL)
    w = 0.8 / max(len(order), 1)
    xb = np.arange(len(all_k))
    for i, k in enumerate(order):
        p = per_model[i] if i < len(per_model) else {}
        heights = [float(p.get(ak, 0.0)) for ak in all_k]
        xoff = (i - (len(order) - 1) / 2) * w
        ax.bar(
            xb + xoff,
            heights,
            width=w * 0.95,
            label=label_map.get(k, k),
        )
    ax.set_xticks(xb, [s.replace("_", " ") for s in all_k], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Sum over the traced episode")
    ax.set_title("Rubric: summed components (see reward/rubric.py)")
    ax.axhline(0, color="k", linewidth=0.4, alpha=0.4)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="best", fontsize=7)


class RewardRubricKeys:
    """Fallback x-order if a trace has no component dicts (should not happen)."""
    ALL = (
        "glucose_improvement",
        "trend_shaping",
        "hypoglycemia_penalty",
        "side_effects",
        "instability",
        "treatment_cost",
        "inactivity",
        "time_decay",
        "cumulative_hyper",
        "tool_intervention",
        "terminal",
    )


def _plot_outcomes(
    axs: np.ndarray,
    endpoints: Dict[str, List[Dict[str, Any]]],
    order: List[str],
    label_map: Dict[str, str],
) -> None:
    for ax, key, title in zip(
        axs.flat,
        ("final_hba1c", "final_fpg", "return"),
        ("Final HbA1c (N episodes)", "Final fasting glucose (N episodes)", "Episode return (N episodes)"),
    ):
        data: List[np.ndarray] = []
        names: List[str] = []
        for k in order:
            rows = endpoints.get(k) or []
            if not rows:
                continue
            vals: List[float] = []
            for r in rows:
                if key not in r:
                    continue
                vals.append(float(r[key]))
            if not vals:
                continue
            data.append(np.array(vals, dtype=float))
            names.append(label_map.get(k, k))
        if not data:
            ax.set_title(title + " (no data)")
            continue
        ax.boxplot(data, showmeans=True, meanline=True)
        ax.set_xticklabels(names, rotation=12, ha="right", fontsize=7)
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.35)


def main() -> None:
    plt.switch_backend("Agg")
    p = argparse.ArgumentParser()
    p.add_argument("--in-json", type=Path, default=repo_root / "logs" / "colab_experiment.json")
    p.add_argument("--out-dir", type=Path, default=repo_root / "docs" / "figures")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    d = json.loads(args.in_json.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    traces: Dict[str, Dict[str, Any]] = dict(d.get("traces") or {})
    endpoints: Dict[str, List[Dict[str, Any]]] = dict(d.get("endpoints") or {})
    label_map: Dict[str, str] = {
        "random": "Random policy",
        "distilgpt2": "Trained small LM (distilgpt2)",
        "llama-8b-4bit": "4-bit 8B (one-episode trace if present)",
        "council_self_repair": "Council + self-repair (one example ep)",
    }
    trace_order = d.get("judge_trace_order", ["random", "distilgpt2", "council_self_repair"])
    trace_order = [k for k in trace_order if k in traces and traces.get(k, {}).get("step_reward")]

    written: List[Path] = []

    if trace_order and any(traces.get(k) for k in trace_order):
        fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.5))
        _plot_clinical_panels(axs, traces, trace_order, label_map)
        fig.suptitle("Partial observability + stochastic twin: key vitals and treatment cost (one ep / policy where logged)")
        fig.tight_layout()
        p0 = args.out_dir / "judge_clinical_state.png"
        fig.savefig(p0, dpi=args.dpi)
        plt.close(fig)
        written.append(p0)

        fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.0, 3.8))
        _plot_reward_dynamics(axa, axb, traces, trace_order, label_map)
        fig.suptitle("How the rubric accrues over simulated weeks (example trajectories)")
        fig.tight_layout()
        p1 = args.out_dir / "judge_step_and_cumulative_return.png"
        fig.savefig(p1, dpi=args.dpi)
        plt.close(fig)
        written.append(p1)

        alabels, amat = _action_frequencies(traces, trace_order)
        if alabels:
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            n_models, n_labs = amat.shape
            xb = np.arange(n_labs)
            w = 0.8 / max(n_models, 1)
            for i in range(n_models):
                xoff = (i - (n_models - 1) / 2) * w
                ax.bar(
                    xb + xoff,
                    amat[i],
                    width=w * 0.9,
                    label=label_map.get(trace_order[i], trace_order[i]),
                )
            ax.set_xticks(xb, alabels, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel("Count (same example episode as clinical plots)")
            ax.set_title("Action types taken (from env action_parser)")
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, axis="y", alpha=0.35)
            fig.tight_layout()
            p2 = args.out_dir / "judge_action_mix.png"
            fig.savefig(p2, dpi=args.dpi)
            plt.close(fig)
            written.append(p2)

        if trace_order:
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            _plot_rubric_sums(ax, traces, trace_order, label_map)
            fig.tight_layout()
            p3 = args.out_dir / "judge_rubric_episode_totals.png"
            fig.savefig(p3, dpi=args.dpi)
            plt.close(fig)
            written.append(p3)

    ep_order = d.get("judge_endpoint_order", ["random", "distilgpt2"])
    ep_order = [k for k in ep_order if k in endpoints and len(endpoints[k] or []) > 0]
    if ep_order:
        fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.5))
        _plot_outcomes(axs, endpoints, ep_order, label_map)
        fig.suptitle("Stochasticity across episodes (same max_steps, different patient seeds per episode)")
        fig.tight_layout()
        p4 = args.out_dir / "judge_outcome_distributions.png"
        fig.savefig(p4, dpi=args.dpi)
        plt.close(fig)
        written.append(p4)

    if traces.get("council_self_repair") and traces["council_self_repair"].get("step_reward"):
        fig = plt.figure(figsize=(6.0, 3.5))
        t = traces["council_self_repair"]
        wx = _weeks_for_steps(t)
        if wx is not None and t.get("fasting_glucose") and t.get("week"):
            w = np.array(t["week"], dtype=int)
            y = np.array(t["fasting_glucose"], dtype=float)
            m = min(len(w), len(y))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(w[:m], y[:m], "o-", markersize=2, linewidth=1.2, label=label_map["council_self_repair"])
            ax.set_xlabel("Week")
            ax.set_ylabel("Fasting glucose (example ep)")
            ax.set_title("Council: example glucose path (no LLM; rule fusion + self-repair loop elsewhere)")
            ax.grid(True, alpha=0.35)
            ax.legend(loc="best", fontsize=7)
            fig.tight_layout()
            p5 = args.out_dir / "judge_council_glucose_example.png"
            fig.savefig(p5, dpi=args.dpi)
            plt.close(fig)
            written.append(p5)

    print("judge_plots", "wrote" if written else "skipped (no traces/endpoints):", *written, sep=" ")


if __name__ == "__main__":
    main()
