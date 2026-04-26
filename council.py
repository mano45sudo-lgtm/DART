"""Rule-based multi-agent council: propose, cross-score, pick action (no LLM)."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from env.digital_twin_env import DigitalTwinDiabetesEnv
from env.action_parser import parse_action

from agents import BaseAgent, DiagnosticAgent, RiskAgent, TreatmentAgent


def _enrich_obs(obs: Dict[str, Any], env: Optional[DigitalTwinDiabetesEnv] = None) -> Dict[str, Any]:
    s = {**obs}
    if env is not None:
        st = env.state()["state"]
        meds = st.get("meds") or {}
        s["_n_meds"] = len(meds) if isinstance(meds, dict) else 0
        s["_lifestyle_st"] = float(st.get("lifestyle_intensity", 0.4))
    return s


def _action_key(a: Dict[str, Any]) -> str:
    clean = {k: v for k, v in a.items() if k not in {"rationale", "meta"}}
    return json.dumps(clean, sort_keys=True)


def safe_baseline_action(_state: Dict[str, Any]) -> Dict[str, Any]:
    """Conservative metformin + lifestyle when council is in fallback mode."""
    return {"type": "start", "drug": "metformin", "dose": 0.45, "lifestyle": 0.5}


def build_default_council(
    seed: int = 0, *, self_improvement_weights: Optional[Dict[str, float]] = None
) -> "Council":
    w = self_improvement_weights or {
        "treatment": 0.5,
        "risk": 0.3,
        "diagnostic": 0.2,
    }
    agents: List[BaseAgent] = [
        TreatmentAgent(seed=seed),
        RiskAgent(),
        DiagnosticAgent(seed=seed + 7),
    ]
    return Council(agents, weights=dict(w), rng_seed=seed + 13)


class Council:
    def __init__(
        self,
        agents: List[BaseAgent],
        weights: Optional[Dict[str, float]] = None,
        *,
        rng_seed: int = 0,
    ) -> None:
        if len(agents) != 3:
            raise ValueError("this Council expects 3 agents (treatment, risk, diagnostic order)")
        self.agents = agents
        w = weights or {
            "treatment": 0.5,
            "risk": 0.3,
            "diagnostic": 0.2,
        }
        self.weights: Dict[str, float] = {k: float(w[k]) for k in w}
        self._rng = random.Random(rng_seed)
        self._normalize_weights()

    def set_weights(self, w: Dict[str, float]) -> None:
        self.weights.update({k: float(v) for k, v in w.items() if k in self.weights})
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        t = self.weights.get("treatment", 0.0)
        r = self.weights.get("risk", 0.0)
        d = self.weights.get("diagnostic", 0.0)
        s = max(t + r + d, 1e-8)
        self.weights = {"treatment": t / s, "risk": r / s, "diagnostic": d / s}

    def score_action(self, state: Dict[str, Any], action: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        by_agent: Dict[str, float] = {}
        for ag in self.agents:
            by_agent[ag.name] = float(ag.evaluate(state, action))
        wsum = sum(self.weights[name] * by_agent.get(name, 0.0) for name in self.weights)
        return float(wsum), by_agent

    def decide(
        self,
        obs: Dict[str, Any],
        env: Optional[DigitalTwinDiabetesEnv] = None,
        *,
        use_fallback: bool = False,
        exploration: float = 0.0,
    ) -> Dict[str, Any]:
        st = _enrich_obs(obs, env)
        if use_fallback:
            a = safe_baseline_action(st)
            pa = parse_action(a)
            ssum, b_ag = self.score_action(st, pa.action)
            return {
                "final_action": pa.action,
                "agent_proposals": {},
                "per_action_scores": [],
                "winning_key": "fallback",
                "weights_used": {**self.weights},
                "use_fallback": True,
                "agent_votes": b_ag,
                "reasoning": f"safe baseline (score={ssum:.3f}) per self-improvement controller",
            }

        seen: Dict[str, Dict[str, Any]] = {}
        prop_map: Dict[str, Any] = {}
        for ag in self.agents:
            p = ag.propose(st)
            prop_map[ag.name] = p
            pa = parse_action(p)
            act = pa.action
            k = _action_key(act)
            if k not in seen:
                seen[k] = act

        for act in list(seen.values()):
            if self._rng.random() < exploration and len(seen) < 6:
                nb = {**act, "dose": min(0.95, float(act.get("dose", 0.5) or 0.5) + 0.05 * self._rng.random())}
                if "type" in nb and str(nb["type"]) in {"dose_adjust", "start", "add"} and "dose" in nb:
                    k2 = _action_key(parse_action(nb).action)
                    if k2 not in seen:
                        seen[k2] = parse_action(nb).action

        scored: List[Tuple[str, float, Dict[str, float], Dict[str, Any]]] = []
        for act in seen.values():
            s, by_ag = self.score_action(st, act)
            noise = 0.02 * self._rng.random() * max(0.0, exploration)
            scored.append((_action_key(act), s + noise, by_ag, act))
        scored.sort(key=lambda x: -x[1])
        if not scored:
            fa = safe_baseline_action(st)
            pa = parse_action(fa)
            ssum, b_ag = self.score_action(st, pa.action)
            return {
                "final_action": pa.action,
                "agent_proposals": prop_map,
                "per_action_scores": [],
                "winning_key": "empty_set",
                "weights_used": {**self.weights},
                "use_fallback": True,
                "agent_votes": b_ag,
                "reasoning": f"degenerate scoring; using safe action score={ssum:.3f}",
            }

        best = scored[0][3]
        k0, s0, by_win, _b = scored[0]
        best_pa = parse_action(best)
        return {
            "final_action": best_pa.action,
            "agent_proposals": prop_map,
            "per_action_scores": [
                {
                    "action_key": k,
                    "total_score": sc,
                    "by_agent": by_ag,
                }
                for k, sc, by_ag, _a in scored
            ],
            "winning_key": scored[0][0],
            "weights_used": {**self.weights},
            "use_fallback": False,
            "agent_votes": by_win,
            "reasoning": f"council max weighted score={s0:.3f} key={k0[:48]}",
        }
