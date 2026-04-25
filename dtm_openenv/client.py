from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from dtm_openenv.models import DTMAction, DTMObservation, DTMState


class DigitalTwinMedicineClient:
    """
    Minimal HTTP client for the FastAPI server (Colab-friendly).
    This is NOT the official OpenEnv client, but it makes the environment usable
    even without Docker/WebSocket tooling.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, *, seed: Optional[int] = None) -> DTMObservation:
        r = requests.post(f"{self.base_url}/reset", json={"seed": seed})
        r.raise_for_status()
        payload = r.json()
        return DTMObservation(**payload["observation"])

    def step(self, action: DTMAction) -> tuple[DTMObservation, float, bool]:
        r = requests.post(f"{self.base_url}/step", json=action.model_dump(exclude_none=True))
        r.raise_for_status()
        payload = r.json()
        return DTMObservation(**payload["observation"]), float(payload["reward"]), bool(payload["done"])

    def state(self) -> DTMState:
        r = requests.get(f"{self.base_url}/state")
        r.raise_for_status()
        return DTMState(**r.json())

