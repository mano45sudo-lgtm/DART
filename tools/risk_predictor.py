from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RiskReport:
    hypoglycemia_risk: float
    renal_risk: float
    cvd_risk: float
    overall_risk: float


def predict_risk(*, patient_profile: Dict[str, Any], patient_state: Dict[str, Any]) -> RiskReport:
    """
    Inputs:
      - patient_profile/state dicts
    Outputs:
      - risk scores 0..1 (mock)
    """
    age = float(patient_profile.get("age", 55))
    egfr = float(patient_state.get("egfr", 90))
    hba1c = float(patient_state.get("hba1c", 8.0))
    systolic = float(patient_state.get("systolic_bp", 130.0))
    meds = patient_state.get("meds", {}) or {}

    on_su_or_insulin = float(("sulfonylurea" in meds) or ("insulin" in meds))

    hypo = min(max(0.05 + 0.004 * max(age - 60, 0) + 0.18 * on_su_or_insulin, 0.0), 0.95)
    renal = min(max(0.05 + 0.012 * max(60 - egfr, 0), 0.0), 0.95)
    cvd = min(max(0.06 + 0.005 * max(age - 50, 0) + 0.07 * float(systolic > 140) + 0.08 * float(hba1c > 9.0), 0.0), 0.95)

    overall = min(0.98, 0.34 * hypo + 0.33 * renal + 0.33 * cvd)
    return RiskReport(hypoglycemia_risk=float(hypo), renal_risk=float(renal), cvd_risk=float(cvd), overall_risk=float(overall))

