from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


ALLOWED_DRUGS = {"none", "metformin", "sulfonylurea", "dpp4", "glp1", "sglt2", "insulin"}
ALLOWED_TYPES = {"noop", "start", "add", "stop", "switch", "dose_adjust"}


@dataclass
class ParsedAction:
    action: Dict[str, Any]
    ok: bool
    error: Optional[str] = None
    raw: Any = None


def _as_float01(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        v = float(x)
        return max(0.0, min(1.0, v))
    except Exception:
        return default


def parse_action(raw: Any) -> ParsedAction:
    """
    Accepts:
      - dict (already structured)
      - JSON string (LLM output)
    Produces a normalized dict compatible with env/patient_twin.apply_treatment_action().
    """
    original = raw
    if isinstance(raw, str):
        s = raw.strip()
        # tolerate surrounding ```json fences
        if s.startswith("```"):
            s = "\n".join([ln for ln in s.splitlines() if not ln.strip().startswith("```")]).strip()
        try:
            raw = json.loads(s)
        except Exception as e:
            return ParsedAction(action={"type": "noop"}, ok=False, error=f"json_parse_error:{e}", raw=original)

    if not isinstance(raw, dict):
        return ParsedAction(action={"type": "noop"}, ok=False, error="action_not_object", raw=original)

    a_type = str(raw.get("type", "noop"))
    if a_type not in ALLOWED_TYPES:
        return ParsedAction(action={"type": "noop"}, ok=False, error="invalid_type", raw=original)

    out: Dict[str, Any] = {"type": a_type}

    # shared fields
    lifestyle = _as_float01(raw.get("lifestyle", None), default=None)
    if lifestyle is not None:
        out["lifestyle"] = lifestyle

    if a_type in {"start", "add", "stop", "dose_adjust"}:
        drug = str(raw.get("drug", "none"))
        if drug not in ALLOWED_DRUGS:
            return ParsedAction(action={"type": "noop"}, ok=False, error="invalid_drug", raw=original)
        out["drug"] = drug

        dose = _as_float01(raw.get("dose", 1.0), default=1.0)
        if a_type in {"start", "add", "dose_adjust"}:
            out["dose"] = float(dose if dose is not None else 1.0)

    if a_type == "switch":
        fd = str(raw.get("from_drug", "none"))
        td = str(raw.get("to_drug", "none"))
        if fd not in ALLOWED_DRUGS or td not in ALLOWED_DRUGS or td == "none":
            return ParsedAction(action={"type": "noop"}, ok=False, error="invalid_switch", raw=original)
        out["from_drug"] = fd
        out["to_drug"] = td
        dose = _as_float01(raw.get("dose", 1.0), default=1.0)
        out["dose"] = float(dose if dose is not None else 1.0)

    # free text (kept but ignored by env)
    if "rationale" in raw:
        out["rationale"] = str(raw.get("rationale", ""))[:500]
    if "meta" in raw and isinstance(raw["meta"], dict):
        out["meta"] = raw["meta"]

    return ParsedAction(action=out, ok=True, error=None, raw=original)


def safe_action(raw: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (action_dict, parse_info) where action_dict is always valid (fallback noop).
    """
    pa = parse_action(raw)
    parse_info = {"ok": pa.ok, "error": pa.error}
    return pa.action, parse_info

