from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tools.biomarker_tracker import track_biomarkers
from tools.resistance_detector import detect_resistance, suggest_alternatives
from tools.risk_predictor import predict_risk


@dataclass
class FailureDetection:
    failed: bool
    category: str  # "toxicity"|"resistance"|"deterioration"|"none"
    reason: str
    severity: float
    alternatives: List[Dict[str, Any]]


def detect_and_recover(
    *,
    patient_profile: Dict[str, Any],
    trajectory_states: List[Dict[str, Any]],
    latest_info: Dict[str, Any],
) -> FailureDetection:
    """
    Pipeline:
      1) detect failure signals (toxicity, resistance, deterioration)
      2) classify
      3) generate alternative actions
      4) return recommendation; env may auto-switch if configured
    """
    if not trajectory_states:
        return FailureDetection(False, "none", "no_history", 0.0, [])

    cur = trajectory_states[-1]
    prof = patient_profile

    # toxicity: high severity side effect or repeated events
    side_effects = list(latest_info.get("side_effects") or [])
    if side_effects:
        max_sev = max(float(ev.get("severity", 1)) for ev in side_effects)
        if max_sev >= 3:
            alts = suggest_alternatives(
                current_regimen={"meds": cur.get("meds", {}), "lifestyle_intensity": cur.get("lifestyle_intensity", 0.2)},
                ckd=bool(cur.get("ckd")),
                cvd=bool(cur.get("cvd")),
                obese=float(cur.get("bmi", 30.0)) >= 30.0,
            )
            return FailureDetection(True, "toxicity", "high_severity_side_effect", max_sev, alts)

    # deterioration: runaway biomarkers
    if float(cur.get("fasting_glucose", 160.0)) > 320.0 or float(cur.get("hba1c", 8.0)) > 11.0:
        alts = suggest_alternatives(
            current_regimen={"meds": cur.get("meds", {}), "lifestyle_intensity": cur.get("lifestyle_intensity", 0.2)},
            ckd=bool(cur.get("ckd")),
            cvd=bool(cur.get("cvd")),
            obese=float(cur.get("bmi", 30.0)) >= 30.0,
        )
        return FailureDetection(True, "deterioration", "glycemic_out_of_control", 0.9, alts)

    # resistance: HbA1c not improving on regimen
    recent_hba1c = [float(s.get("hba1c", 8.0)) for s in trajectory_states]
    regimen = {"meds": cur.get("meds", {}), "lifestyle_intensity": cur.get("lifestyle_intensity", 0.2)}
    rs = detect_resistance(recent_hba1c=recent_hba1c, regimen=regimen)
    if rs.resistant and rs.confidence >= 0.6:
        alts = suggest_alternatives(
            current_regimen=regimen,
            ckd=bool(cur.get("ckd")),
            cvd=bool(cur.get("cvd")),
            obese=float(cur.get("bmi", 30.0)) >= 30.0,
        )
        return FailureDetection(True, "resistance", rs.reason, float(rs.confidence), alts)

    # risk-based caution: if overall risk too high, suggest safer escalation
    rr = predict_risk(patient_profile=prof, patient_state=cur)
    if rr.overall_risk > 0.75:
        alts = suggest_alternatives(
            current_regimen=regimen,
            ckd=bool(cur.get("ckd")),
            cvd=bool(cur.get("cvd")),
            obese=float(cur.get("bmi", 30.0)) >= 30.0,
        )
        return FailureDetection(True, "toxicity", "overall_risk_high", float(rr.overall_risk), alts)

    return FailureDetection(False, "none", "stable", 0.0, [])

