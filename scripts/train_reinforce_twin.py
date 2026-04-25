#!/usr/bin/env python3
"""
End-to-end REINFORCE training on DigitalTwinDiabetesEnv (no static dataset).

Requires: pip install -r requirements_hackathon.txt  (or at least torch + transformers)

Example (CPU demo, ~few minutes):
  python scripts/train_reinforce_twin.py --quick

Stronger run for judges (GPU recommended):
  python scripts/train_reinforce_twin.py --updates 120 --episodes-per-update 4 --eval-seeds 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "scripts"))

from evaluation.baseline_random_agent import RandomAgent  # noqa: E402
from evaluation.eval_metrics import summarize  # noqa: E402
from run_evaluation import evaluate_agent  # noqa: E402
from training.llm_reinforce import (  # noqa: E402
    eval_mean_return,
    reinforce_loss_on_episode,
    rollout_episode,
)


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _plot_results(payload: Dict[str, Any], fig_dir: Path) -> None:
    plt.switch_backend("Agg")
    fig_dir.mkdir(parents=True, exist_ok=True)
    t = payload["training"]
    updates = np.array(t["update_index"], dtype=int)
    train_ret = np.array(t["train_mean_episode_return"], dtype=float)
    eval_x = np.array(t["eval_at_update"], dtype=int)
    eval_y = np.array(t["eval_llm_mean_return"], dtype=float)
    eval_std = np.array(t["eval_llm_std_return"], dtype=float)
    br = payload["baseline_random"]["avg_return"]
    br_std = payload["baseline_random"].get("std_return", 0.0)
    bu = payload.get("baseline_untrained_llm", {})
    bu_m = bu.get("mean_return")
    bu_s = bu.get("std_return", 0.0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(updates, train_ret, color="#2563eb", alpha=0.85, linewidth=1.8, label="Training batch mean episode return")
    ax.axhline(br, color="#64748b", linestyle="--", linewidth=1.5, label=f"Random policy mean return ({br:.2f})")
    ax.fill_between(
        updates,
        br - br_std,
        br + br_std,
        color="#64748b",
        alpha=0.12,
        label="Random ±1 std (eval episodes)",
    )
    if bu_m is not None:
        ax.axhline(bu_m, color="#94a3b8", linestyle=":", linewidth=1.4, label=f"Untrained LM mean return ({bu_m:.2f})")
        if bu_s:
            ax.fill_between(updates, bu_m - bu_s, bu_m + bu_s, color="#94a3b8", alpha=0.08)

    ax.errorbar(
        eval_x,
        eval_y,
        yerr=eval_std,
        fmt="o-",
        color="#ea580c",
        capsize=3,
        markersize=5,
        linewidth=1.4,
        label="Held-out eval: LM mean ± std",
    )
    ax.set_xlabel("Training update (batch of rollouts on the twin env)")
    ax.set_ylabel("Total reward per episode (sum of weekly step rewards)")
    ax.set_title("REINFORCE on digital twin: random baseline vs learning curve")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p1 = fig_dir / "training_vs_baselines.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # Bar: final comparison
    fr = payload["final_eval"]["random"]["avg_return"]
    frs = payload["final_eval"]["random"].get("std_return", 0.0)
    fl = payload["final_eval"]["llm"]["mean_return"]
    fls = payload["final_eval"]["llm"]["std_return"]
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    labels = ["Random\n(untrained)", "LM after\nREINFORCE"]
    means = [fr, fl]
    stds = [frs, fls]
    colors = ["#64748b", "#ea580c"]
    x = np.arange(len(labels))
    ax2.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="white", linewidth=0.8, alpha=0.9)
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("Mean episode return (same eval seeds)")
    ax2.set_title("Post-training evaluation on held-out seeds")
    ax2.grid(True, axis="y", alpha=0.35)
    fig2.tight_layout()
    p2 = fig_dir / "final_random_vs_trained.png"
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilgpt2", help="HF causal LM (public, small default)")
    p.add_argument("--max-steps", type=int, default=24, help="Max simulated weeks per episode")
    p.add_argument("--updates", type=int, default=48)
    p.add_argument("--episodes-per-update", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-new-tokens", type=int, default=56)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--eval-every", type=int, default=6)
    p.add_argument("--eval-seeds", type=int, default=20, help="Count of fixed seeds starting at --eval-seed-base")
    p.add_argument("--eval-seed-base", type=int, default=9000)
    p.add_argument("--train-seed-base", type=int, default=101)
    p.add_argument("--random-eval-episodes", type=int, default=48)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--fig-dir", type=Path, default=repo_root / "docs" / "figures")
    p.add_argument("--save-model", type=Path, default=None, help="Optional path to save final model dir")
    p.add_argument("--quick", action="store_true", help="Short run for smoke tests")
    args = p.parse_args()

    if args.quick:
        # Small public model so CPU smoke runs finish in minutes, not hours.
        args.model = "sshleifer/tiny-gpt2"
        args.updates = 14
        args.episodes_per_update = 1
        args.eval_every = 4
        args.eval_seeds = 8
        args.random_eval_episodes = 20
        args.max_new_tokens = 40
        args.max_steps = 16

    device_s = _device()
    device = torch.device(device_s)
    eval_seed_list = [args.eval_seed_base + i for i in range(args.eval_seeds)]

    print("device:", device_s)
    print("loading model:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    rng_agent = RandomAgent(seed=0)
    random_metrics = evaluate_agent(rng_agent, episodes=args.random_eval_episodes, seed=0, max_steps=args.max_steps)
    random_summary = summarize(random_metrics)
    random_returns = np.array([m.ep_return for m in random_metrics], dtype=float)

    train_update_idx: List[int] = []
    train_mean_ret: List[float] = []
    eval_at: List[int] = []
    eval_means: List[float] = []
    eval_stds: List[float] = []
    eval_parse: List[float] = []

    def snapshot_eval(u: int) -> None:
        model.eval()
        m, s, pr = eval_mean_return(
            model,
            tokenizer,
            seeds=eval_seed_list,
            max_steps=args.max_steps,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        model.train()
        eval_at.append(u)
        eval_means.append(m)
        eval_stds.append(s)
        eval_parse.append(pr)
        print(f"  eval@{u}: mean_return={m:.3f} std={s:.3f} parse_ok_rate={pr:.3f}")

    snapshot_eval(0)
    u_mean = eval_means[0]
    u_std = eval_stds[0]
    u_parse = eval_parse[0]

    for u in range(1, args.updates + 1):
        batch_returns: List[float] = []
        opt.zero_grad(set_to_none=True)
        ep_losses: List[torch.Tensor] = []
        for e in range(args.episodes_per_update):
            env_seed = args.train_seed_base + u * 1000 + e * 17
            steps, ep_ret = rollout_episode(
                model,
                tokenizer,
                env_seed=env_seed,
                max_steps=args.max_steps,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            batch_returns.append(ep_ret)
            loss_ep = reinforce_loss_on_episode(steps, gamma=1.0)
            if loss_ep is not None:
                ep_losses.append(loss_ep)
        if ep_losses:
            loss = torch.stack(ep_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        train_update_idx.append(u)
        train_mean_ret.append(float(np.mean(batch_returns)))
        print(f"update {u}/{args.updates} train_batch_mean_return={train_mean_ret[-1]:.3f}")
        if u % args.eval_every == 0 or u == args.updates:
            snapshot_eval(u)

    model.eval()
    final_llm_m, final_llm_s, final_pr = eval_mean_return(
        model,
        tokenizer,
        seeds=eval_seed_list,
        max_steps=args.max_steps,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    random_metrics_final = evaluate_agent(rng_agent, episodes=args.random_eval_episodes, seed=1, max_steps=args.max_steps)
    random_summary_final = summarize(random_metrics_final)

    payload: Dict[str, Any] = {
        "protocol": {
            "model": args.model,
            "max_steps_per_episode": args.max_steps,
            "updates": args.updates,
            "episodes_per_update": args.episodes_per_update,
            "lr": args.lr,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "eval_seed_base": args.eval_seed_base,
            "eval_seed_count": len(eval_seed_list),
            "device": device_s,
        },
        "baseline_random": {
            "avg_return": float(random_summary["avg_return"]),
            "std_return": float(np.std(random_returns)),
            "remission_rate": float(random_summary.get("remission_rate", 0.0)),
            "failure_rate": float(random_summary.get("failure_rate", 0.0)),
            "episodes": int(args.random_eval_episodes),
        },
        "baseline_untrained_llm": {
            "mean_return": float(u_mean),
            "std_return": float(u_std),
            "parse_ok_rate": float(u_parse),
            "eval_episodes": len(eval_seed_list),
        },
        "training": {
            "update_index": train_update_idx,
            "train_mean_episode_return": train_mean_ret,
            "eval_at_update": eval_at,
            "eval_llm_mean_return": eval_means,
            "eval_llm_std_return": eval_stds,
            "eval_llm_parse_ok": eval_parse,
        },
        "final_eval": {
            "random": {
                "avg_return": float(random_summary_final["avg_return"]),
                "std_return": float(np.std([m.ep_return for m in random_metrics_final])),
                "episodes": args.random_eval_episodes,
            },
            "llm": {
                "mean_return": float(final_llm_m),
                "std_return": float(final_llm_s),
                "parse_ok_rate": float(final_pr),
                "eval_episodes": len(eval_seed_list),
            },
        },
    }

    out_json = args.out_json or (repo_root / "logs" / "training_last.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out_json)

    _plot_results(payload, args.fig_dir)
    print("wrote plots to", args.fig_dir)

    if args.save_model:
        args.save_model.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print("saved model to", args.save_model)


if __name__ == "__main__":
    main()
