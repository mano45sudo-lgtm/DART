from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DTMAction(BaseModel):
    """
    OpenEnv Action model.
    The LLM should output JSON matching this structure.
    """

    type: str = Field(default="noop")
    drug: Optional[str] = None
    dose: Optional[float] = None
    lifestyle: Optional[float] = None
    from_drug: Optional[str] = None
    to_drug: Optional[str] = None
    rationale: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class DTMObservation(BaseModel):
    """
    OpenEnv Observation model.
    """

    observation: Dict[str, Any]
    info: Dict[str, Any] = Field(default_factory=dict)


class DTMState(BaseModel):
    """
    OpenEnv State model (episode metadata + full patient state).
    """

    episode_id: str
    step_count: int
    patient_profile: Dict[str, Any]
    patient_state: Dict[str, Any]
    history: List[Dict[str, Any]] = Field(default_factory=list)

