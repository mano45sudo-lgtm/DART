from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from env.digital_twin_env import DigitalTwinDiabetesEnv

    env = DigitalTwinDiabetesEnv(seed=0, max_steps=24)
    obs, info = env.reset()
    print("patient_id:", info["patient_id"])
    print("obs0:", json.dumps(obs, indent=2))

    actions = [
        {"type": "start", "drug": "metformin", "dose": 1.0, "lifestyle": 0.7, "rationale": "first-line"},
        {"type": "noop"},
        {"type": "noop"},
        {"type": "add", "drug": "glp1", "dose": 1.0, "rationale": "weight + glycemic"},
    ]

    total = 0.0
    for t in range(12):
        a = actions[t] if t < len(actions) else {"type": "noop"}
        obs, r, term, trunc, info = env.step(a)
        total += r
        print(
            f"t={t+1:02d} hba1c={obs['hba1c']:.2f} fpg={obs['fasting_glucose']:.0f} "
            f"egfr={obs['egfr']:.0f} r={r:+.2f} fd={info['fall_detection']['category']}"
        )
        if term or trunc:
            break
    print("total_reward:", round(total, 3))


if __name__ == "__main__":
    main()

