#!/usr/bin/env python3
"""Evaluate random vs rule baselines; optional JSON for plotting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from evaluation.baseline_rule_agent import RuleBasedAgent  # noqa: E402
from evaluation.baseline_random_agent import RandomAgent  # noqa: E402
from evaluation.pipeline import compare_random_rule_trained, evaluate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=24)
    p.add_argument("--out", type=Path, default=repo_root / "logs" / "evaluation_baseline.json")
    args = p.parse_args()
    r = compare_random_rule_trained(n_episodes=args.n_episodes, seed=args.seed, max_steps=args.max_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print("random mean ± approx:", r["random"]["mean_reward"], "std", r["random"]["std_reward"])
    print("rule   mean ± approx:", r["rule"]["mean_reward"], "std", r["rule"]["std_reward"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
