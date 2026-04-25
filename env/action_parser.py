from __future__ import annotations

import json
import re
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


def _extract_first_json_object(text: str) -> Optional[str]:
    """Best-effort extraction of first balanced {...} object from free-form text."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _heuristic_action_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse coarse text when strict JSON fails.
    Keeps training/eval alive for weak LMs by mapping obvious intent words.
    """
    s = text.strip().lower()
    if not s:
        return None

    # Action type intent.
    a_type = "noop"
    if "dose" in s and ("adjust" in s or "increase" in s or "decrease" in s):
        a_type = "dose_adjust"
    elif "switch" in s:
        a_type = "switch"
    elif "stop" in s or "discontinue" in s:
        a_type = "stop"
    elif "add" in s:
        a_type = "add"
    elif "start" in s or "begin" in s or "initiate" in s:
        a_type = "start"

    drugs = [d for d in ALLOWED_DRUGS if d != "none" and d in s]
    out: Dict[str, Any] = {"type": a_type}

    if a_type == "switch" and len(drugs) >= 2:
        out["from_drug"] = drugs[0]
        out["to_drug"] = drugs[1]
    elif a_type in {"start", "add", "stop", "dose_adjust"} and drugs:
        out["drug"] = drugs[0]
    elif a_type == "noop" and drugs:
        # If model mentions a drug without schema, prefer "start drug".
        out["type"] = "start"
        out["drug"] = drugs[0]

    dm = re.search(r"\bdose\b[^0-9]*([0-9]*\.?[0-9]+)", s)
    if dm:
        out["dose"] = _as_float01(dm.group(1), default=1.0)
    elif out["type"] in {"start", "add", "dose_adjust"} and "drug" in out:
        out["dose"] = 1.0

    lm = re.search(r"\blifestyle\b[^0-9]*([0-9]*\.?[0-9]+)", s)
    if lm:
        lv = _as_float01(lm.group(1), default=None)
        if lv is not None:
            out["lifestyle"] = lv

    if out["type"] == "switch" and ("from_drug" not in out or "to_drug" not in out):
        return None
    if out["type"] in {"start", "add", "stop", "dose_adjust"} and "drug" not in out:
        return None
    return out


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
        s_candidate = _extract_first_json_object(s) or s
        try:
            raw = json.loads(s_candidate)
        except Exception as e:
            h = _heuristic_action_from_text(s)
            if h is not None:
                raw = h
            else:
                return ParsedAction(action={"type": "noop"}, ok=False, error=f"json_parse_error:{e}", raw=original)

    if not isinstance(raw, dict):
        return ParsedAction(action={"type": "noop"}, ok=False, error="action_not_object", raw=original)

    a_type = str(raw.get("type", "noop")).strip().lower()
    if a_type not in ALLOWED_TYPES:
        return ParsedAction(action={"type": "noop"}, ok=False, error="invalid_type", raw=original)

    out: Dict[str, Any] = {"type": a_type}

    # shared fields
    lifestyle = _as_float01(raw.get("lifestyle", None), default=None)
    if lifestyle is not None:
        out["lifestyle"] = lifestyle

    if a_type in {"start", "add", "stop", "dose_adjust"}:
        drug = str(raw.get("drug", "none")).strip().lower()
        if drug not in ALLOWED_DRUGS:
            return ParsedAction(action={"type": "noop"}, ok=False, error="invalid_drug", raw=original)
        out["drug"] = drug

        dose = _as_float01(raw.get("dose", 1.0), default=1.0)
        if a_type in {"start", "add", "dose_adjust"}:
            out["dose"] = float(dose if dose is not None else 1.0)

    if a_type == "switch":
        fd = str(raw.get("from_drug", "none")).strip().lower()
        td = str(raw.get("to_drug", "none")).strip().lower()
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

