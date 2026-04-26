"""Lightweight base interface for rule-based council agents (no LLM)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def propose(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return one structured env action (before validation)."""

    @abstractmethod
    def evaluate(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        """
        Return a unbounded or normalized score; council combines via weighted sum.
        Higher = more alignment with this agent's objective.
        """


def copy_action(a: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in a.items() if not str(k).startswith("_")}
