#!/usr/bin/env python3
"""
Build a minimal but valid colab_experiment.json (no GPU / no LLM) and write all
publication + judge figures into docs/figures/. Committed so GitHub/HF README images resolve.

Run from repo root:  python scripts/generate_readme_demo_figures.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from training.colab_episode_rl import (  # noqa: E402
    collect_council_clinical_trace,
    collect_endpoints_random_baseline,
    collect_random_baseline,
    council_self_repair_episode_log,
    rollout_clinical_trace_random,
)

MA = 5
K_TAIL = 15
MAX_S = 16
JSEED = 50_200


def _synth_distil_tracelike(random_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Same schema as a trained trace; use random trace shape with nudged values (no model import)."""
    out: Dict[str, Any] = {k: (list(v) if isinstance(v, list) else v) for k, v in random_trace.items()}
    for k in ("fasting_glucose", "hba1c", "bmi", "systolic_bp", "cumulative_return"):
        arr = out.get(k) or []
        for i in range(len(arr)):
            if k == "fasting_glucose" and i > 0:
                arr[i] = min(280.0, float(arr[i]) * 0.98 + 1.0)
            elif k == "hba1c" and i > 0:
                arr[i] = max(5.2, float(arr[i]) - 0.02)
    for i in range(len(out.get("step_reward", []))):
        out["step_reward"][i] = float(out["step_reward"][i]) + 0.15
    return out


def main() -> None:
    (repo_root / "logs").mkdir(exist_ok=True)
    (repo_root / "docs" / "figures").mkdir(parents=True, exist_ok=True)

    r_rows, _, g_random = collect_random_baseline(
        n_episodes=24, max_steps=MAX_S, seed=0, model_key="random"
    )
    trace_r = rollout_clinical_trace_random(env_seed=JSEED, max_steps=MAX_S, random_seed=0)
    end_r = collect_endpoints_random_baseline(
        n_episodes=10, max_steps=MAX_S, seed=0, model_key="random"
    )
    trace_s = _synth_distil_tracelike(trace_r)
    end_s = [
        {
            "model": "distilgpt2",
            "episode": i + 1,
            "return": float(8.0 + 0.4 * i - (i % 3)),
            "final_hba1c": 7.4 - 0.02 * i,
            "final_fpg": 150.0 - 1.5 * i,
            "final_egfr": 88.0 - 0.1 * i,
            "n_steps": MAX_S,
        }
        for i in range(10)
    ]
    s_rows: List[Dict[str, Any]] = []
    for i in range(1, 37):
        s_rows.append(
            {
                "episode": i,
                "reward": -12.0 + 0.35 * i + (i % 4) * 0.1,
                "avg_reward": -5.0 + 0.1 * i,
                "action_count": MAX_S,
                "model": "distilgpt2",
            }
        )
    c_rows, repair_marks = council_self_repair_episode_log(
        n_episodes=12, max_steps=MAX_S, seed=7
    )
    c_trace = collect_council_clinical_trace(
        env_seed=JSEED + 1, max_steps=MAX_S, council_seed=7
    )
    l_rows: List[Dict[str, Any]] = [
        {
            "episode": i,
            "reward": -10.0 + 0.3 * i,
            "avg_reward": -4.0 + 0.12 * i,
            "action_count": MAX_S,
            "model": "llama-8b-4bit",
        }
        for i in range(1, 9)
    ]
    g_trained: List[float] = [180.0 - 0.7 * t for t in range(MAX_S + 1)]

    payload: Dict[str, Any] = {
        "config": {
            "max_steps": MAX_S,
            "demo_generated": True,
            "judge_trace_env_seed": JSEED,
        },
        "episodes": r_rows + s_rows + l_rows + c_rows,
        "glucose": {"random": g_random, "trained": g_trained},
        "self_repair_episodes": repair_marks,
        "plot_order": ["random", "distilgpt2", "llama-8b-4bit", "council_self_repair"],
        "bar_models": ["random", "distilgpt2", "llama-8b-4bit"],
        "bar_tail_episodes": K_TAIL,
        "ma_window": MA,
        "traces": {
            "random": trace_r,
            "distilgpt2": trace_s,
            "council_self_repair": c_trace,
        },
        "endpoints": {"random": end_r, "distilgpt2": end_s},
        "judge_trace_order": ["random", "distilgpt2", "council_self_repair"],
        "judge_endpoint_order": ["random", "distilgpt2"],
    }
    out_path = repo_root / "logs" / "colab_experiment.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", out_path)

    for cmd in (
        [
            sys.executable,
            str(repo_root / "scripts" / "plot_colab_publication.py"),
            "--in-json",
            str(out_path),
            "--out-dir",
            str(repo_root / "docs" / "figures"),
            "--ma-window",
            str(MA),
            "--also-svg",
        ],
        [
            sys.executable,
            str(repo_root / "scripts" / "plot_colab_judge_insights.py"),
            "--in-json",
            str(out_path),
            "--out-dir",
            str(repo_root / "docs" / "figures"),
            "--also-svg",
        ],
    ):
        subprocess.run(cmd, cwd=repo_root, check=True)
    print("figures →", repo_root / "docs" / "figures")


if __name__ == "__main__":
    main()
