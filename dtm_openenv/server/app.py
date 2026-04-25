from __future__ import annotations

"""
FastAPI entrypoint for OpenEnv-style serving.

This file intentionally supports two modes:
1) If `openenv` is installed, expose the environment using OpenEnv server utilities.
2) Otherwise, provide a minimal FastAPI app so the repo remains importable.
"""

from fastapi import FastAPI

from dtm_openenv.server.digital_twin_environment import DigitalTwinMedicineOpenEnv

app = FastAPI(title="Digital Twin Medicine (OpenEnv)")

env = DigitalTwinMedicineOpenEnv(max_steps=52, seed=0)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/reset")
def reset(payload: dict | None = None):
    seed = None if payload is None else payload.get("seed")
    obs = env.reset(seed=seed)
    return {"observation": obs.model_dump(), "state": env.state.model_dump()}


@app.post("/step")
def step(payload: dict):
    from dtm_openenv.models import DTMAction

    action = DTMAction(**payload)
    obs, reward, done = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "state": env.state.model_dump()}


@app.get("/state")
def state():
    return env.state.model_dump()

