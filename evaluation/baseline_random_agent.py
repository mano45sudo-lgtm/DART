from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class RandomAgent:
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.action_templates: List[Dict[str, Any]] = [
            {"type": "noop"},
            {"type": "start", "drug": "metformin", "dose": 1.0, "lifestyle": 0.6},
            {"type": "add", "drug": "glp1", "dose": 1.0},
            {"type": "add", "drug": "sglt2", "dose": 1.0},
            {"type": "add", "drug": "dpp4", "dose": 1.0},
            {"type": "start", "drug": "insulin", "dose": 0.7},
        ]

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        _ = obs
        return dict(self.action_templates[int(self.rng.integers(0, len(self.action_templates)))])

