from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


class OpenEnvDigitalTwinAdapter:
    """
    Minimal adapter that exposes the same reset/step/state but can be used
    from OpenEnv-style runners if `openenv` is installed.

    This avoids breaking Windows installs if openenv isn't present.
    """

    def __init__(self, *, max_steps: int = 52, seed: Optional[int] = None):
        from env.digital_twin_env import DigitalTwinDiabetesEnv

        self._env = DigitalTwinDiabetesEnv(max_steps=max_steps, seed=seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        return self._env.step(action)

    def state(self) -> Dict[str, Any]:
        return self._env.state()

