from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from dtm_openenv.models import DTMAction, DTMObservation, DTMState


class DigitalTwinMedicineOpenEnv:
    """
    OpenEnv-compatible server-side environment wrapper.

    Notes:
    - We keep this wrapper lightweight and re-use the existing core env logic
      implemented in `env/digital_twin_env.py`.
    - If `openenv` is installed, `server/app.py` will expose this via FastAPI.
    """

    def __init__(self, *, max_steps: int = 52, seed: Optional[int] = None):
        from env.digital_twin_env import DigitalTwinDiabetesEnv

        self._env = DigitalTwinDiabetesEnv(max_steps=max_steps, seed=seed)
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._history: list[dict[str, Any]] = []

    def reset(self, *, seed: Optional[int] = None) -> DTMObservation:
        obs, info = self._env.reset(seed=seed)
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._history = []
        self._history.append({"t": 0, "obs": obs, "info": info})
        return DTMObservation(observation=obs, info=info)

    def step(self, action: DTMAction) -> tuple[DTMObservation, float, bool]:
        # Convert action model -> dict for core env
        action_dict: Dict[str, Any] = action.model_dump(exclude_none=True)
        obs, reward, terminated, truncated, info = self._env.step(action_dict)
        done = bool(terminated or truncated)
        self._step_count += 1
        self._history.append({"t": self._step_count, "action": action_dict, "reward": reward, "obs": obs, "info": info})
        return DTMObservation(observation=obs, info=info), float(reward), done

    @property
    def state(self) -> DTMState:
        s = self._env.state()
        return DTMState(
            episode_id=self._episode_id,
            step_count=int(self._step_count),
            patient_profile=dict(s.get("profile", {})),
            patient_state=dict(s.get("state", {})),
            history=list(self._history),
        )

