"""Track recent returns, nudge council weights, exploration, and safe fallback (CPU, no LLM)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from council import Council


@dataclass
class SelfImprovementController:
    window: int = 20
    reward_low: float = -12.0
    reward_stdev_high: float = 28.0
    glucose_stdev_high: float = 45.0
    exploration: float = 0.0
    exploration_max: float = 0.45
    use_fallback: bool = False
    _recent_rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    _recent_glucose_std: Deque[float] = field(default_factory=lambda: deque(maxlen=20))

    def __post_init__(self) -> None:
        self._recent_rewards = deque(maxlen=self.window)
        self._recent_glucose_std = deque(maxlen=self.window)

    def update(self, episode_return: float, last_episode_fasting_std: float = 0.0) -> None:
        self._recent_rewards.append(float(episode_return))
        if last_episode_fasting_std > 0:
            self._recent_glucose_std.append(float(last_episode_fasting_std))
        n = len(self._recent_rewards)
        r = self._roll_mean(self._recent_rewards)
        rs = self._roll_std(self._recent_rewards)
        gs = self._roll_std(self._recent_glucose_std) if self._recent_glucose_std else 0.0
        instable = n >= 3 and ((r < self.reward_low) or (rs > self.reward_stdev_high) or (gs > self.glucose_stdev_high))
        if instable and self.exploration < self.exploration_max - 1e-6:
            self.exploration = min(self.exploration + 0.1, self.exploration_max)
        if n >= 3 and (r < self.reward_low - 5.0 or (gs > self.glucose_stdev_high * 1.1 and n >= 2)):
            self.use_fallback = True
        if self.use_fallback and n >= 4 and r > self.reward_low + 3.0 and not instable:
            self.use_fallback = False

    @staticmethod
    def _roll_mean(d: Deque[float]) -> float:
        if not d:
            return 0.0
        return float(sum(d) / len(d))

    @staticmethod
    def _roll_std(d: Deque[float]) -> float:
        if len(d) < 2:
            return 0.0
        m = sum(d) / len(d)
        v = sum((x - m) ** 2 for x in d) / (len(d) - 1)
        return float(v**0.5)

    def adjust_council(self, council: "Council") -> None:
        r = self._roll_mean(self._recent_rewards)
        rs = self._roll_std(self._recent_rewards)
        w = {**council.weights}
        if r < self.reward_low or rs > self.reward_stdev_high:
            w["risk"] = w.get("risk", 0.3) + 0.04
            w["treatment"] = max(0.12, w.get("treatment", 0.5) - 0.03)
            w["diagnostic"] = w.get("diagnostic", 0.2) - 0.01
        elif r > self.reward_low + 2.0 and not self.use_fallback:
            w["treatment"] = min(0.7, w.get("treatment", 0.5) + 0.01)
        council.set_weights(w)

    def adjust_exploration(self) -> None:
        r = self._roll_mean(self._recent_rewards)
        if r < self.reward_low:
            self.exploration = min(self.exploration + 0.1, self.exploration_max)
        else:
            self.exploration = max(0.0, self.exploration - 0.02)

    def snapshot(self) -> Dict[str, float]:
        return {
            "window_mean_reward": self._roll_mean(self._recent_rewards),
            "window_std_reward": self._roll_std(self._recent_rewards),
            "exploration": float(self.exploration),
            "use_fallback": float(self.use_fallback),
        }
