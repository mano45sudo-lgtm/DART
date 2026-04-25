from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym

from .patient_twin import PatientTwin
from .action_parser import safe_action
from .fall_detection import detect_and_recover
from reward.rubric import RewardRubric


class DigitalTwinDiabetesEnv(gym.Env):
    """
    Gymnasium-compatible scaffold.
    Phase 2+ will implement real dynamics, actions, reward, and fall detection.
    """

    metadata = {"render_modes": []}

    def __init__(self, *, max_steps: int = 52, seed: Optional[int] = None):
        super().__init__()
        self.max_steps = int(max_steps)
        self._seed = seed
        self.twin = PatientTwin(seed=seed)
        self.rewarder = RewardRubric()
        self._trajectory: list[Dict[str, Any]] = []
        self.auto_recover = True
        self._terminated = False
        self._truncated = False

        # Spaces are kept permissive: actual agent interface uses JSON dict actions.
        self.observation_space = gym.spaces.Dict(
            {
                "week": gym.spaces.Box(low=0, high=self.max_steps, shape=(), dtype=int),
                "hba1c": gym.spaces.Box(low=3.0, high=15.0, shape=(), dtype=float),
                "fasting_glucose": gym.spaces.Box(low=50.0, high=450.0, shape=(), dtype=float),
                "bmi": gym.spaces.Box(low=14.0, high=60.0, shape=(), dtype=float),
                "systolic_bp": gym.spaces.Box(low=70.0, high=240.0, shape=(), dtype=float),
                "egfr": gym.spaces.Box(low=5.0, high=160.0, shape=(), dtype=float),
                "ckd": gym.spaces.Discrete(2),
                "cvd": gym.spaces.Discrete(2),
            }
        )
        self.action_space = gym.spaces.Dict({})  # JSON dict externally validated in Phase 4

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        _ = options
        if seed is not None:
            self._seed = seed
        self._terminated = False
        self._truncated = False
        patient_id = "P%04d" % int(gym.utils.seeding.np_random(seed=self._seed)[0].integers(0, 9999))
        self.twin.reset(seed=self._seed, patient_id=patient_id)
        self._trajectory = [self.twin.as_dict()["state"]]
        return self._obs(), {"patient_id": self.twin.profile.patient_id, "profile": self.twin.profile.__dict__}

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self._terminated or self._truncated:
            raise RuntimeError("Episode is done. Call reset().")

        prev_state = self.twin.as_dict()["state"]

        a_dict, parse_info = safe_action(action)
        action_info = self.twin.apply_action(a_dict)

        prog_info = self.twin.step()
        next_state = self.twin.as_dict()["state"]
        if self.twin.state.week >= self.max_steps:
            self._truncated = True

        # Placeholder termination: severe renal failure or extreme hyperglycemia
        if self.twin.state.egfr < 12.0 or self.twin.state.fasting_glucose > 380.0:
            self._terminated = True

        # fall detection + recovery (may override next action in-place)
        self._trajectory.append(next_state)
        fd = detect_and_recover(
            patient_profile=self.twin.as_dict()["profile"],
            trajectory_states=self._trajectory,
            latest_info=prog_info,
        )

        recovery_action = None
        if self.auto_recover and fd.failed and fd.alternatives:
            # pick the first alternative (Phase 6 can be upgraded to trial_simulator selection)
            recovery_action = fd.alternatives[0]
            self.twin.apply_action(recovery_action)

        # reward
        reward_info = {**prog_info, "terminated": self._terminated, "truncated": self._truncated}
        rr = self.rewarder.compute(prev_state=prev_state, next_state=next_state, info=reward_info)
        reward = float(rr.total)

        info: Dict[str, Any] = {
            "action": a_dict,
            "action_parse": parse_info,
            "action_info": action_info,
            "fall_detection": {
                "failed": fd.failed,
                "category": fd.category,
                "reason": fd.reason,
                "severity": fd.severity,
                "alternatives": fd.alternatives,
                "recovery_action": recovery_action,
            },
            "reward": rr.components,
            **prog_info,
        }
        return self._obs(), reward, self._terminated, self._truncated, info

    def state(self) -> Dict[str, Any]:
        """OpenEnv-style state() hook (Phase 7 will finalize)."""
        return self.twin.as_dict()

    def _obs(self) -> Dict[str, Any]:
        # partial observability: only a subset of state is exposed each step
        s = self.twin.state
        return {
            "week": int(s.week),
            "hba1c": float(s.hba1c),
            "fasting_glucose": float(s.fasting_glucose),
            "bmi": float(s.bmi),
            "systolic_bp": float(s.systolic_bp),
            "egfr": float(s.egfr),
            "ckd": int(bool(s.ckd)),
            "cvd": int(bool(s.cvd)),
        }

