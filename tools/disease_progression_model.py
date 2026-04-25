from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ProgressionForecast:
    expected_hba1c_12w: float
    expected_fpg_12w: float
    risk_of_ckd_1y: float
    risk_of_cvd_1y: float


def forecast_progression(
    *,
    patient_profile: Dict[str, Any],
    patient_state: Dict[str, Any],
    regimen: Dict[str, Any],
) -> ProgressionForecast:
    """
    Inputs:
      - patient_profile/state (dict)
      - regimen: meds + lifestyle
    Output:
      - rough forecast numbers used for planning / tool-augmented decisions

    Mock logic: uses simple heuristics, not clinical-grade.
    """
    age = int(patient_profile.get("age", 55))
    hba1c = float(patient_state.get("hba1c", 8.0))
    fpg = float(patient_state.get("fasting_glucose", 160.0))
    egfr = float(patient_state.get("egfr", 90.0))
    bmi = float(patient_state.get("bmi", 30.0))

    lifestyle = float(regimen.get("lifestyle_intensity", 0.2))
    meds = regimen.get("meds", {}) or {}
    potency = 0.0
    potency += 0.6 * lifestyle
    potency += 1.0 * float("metformin" in meds)
    potency += 1.0 * float("glp1" in meds)
    potency += 0.7 * float("sglt2" in meds)
    potency += 1.2 * float("insulin" in meds)

    expected_hba1c_12w = max(5.5, hba1c - 0.4 * potency + 0.15 * (bmi > 35))
    expected_fpg_12w = max(80.0, fpg - 18.0 * potency + 6.0 * (hba1c > 9.0))

    # crude annual risks
    risk_ckd = 1 / (1 + pow(2.71828, -(-2.6 + 0.18 * (hba1c - 7) + 0.02 * (age - 50) - 0.015 * (egfr - 60))))
    risk_cvd = 1 / (1 + pow(2.71828, -(-3.0 + 0.22 * (hba1c - 7) + 0.03 * (age - 50) + 0.6 * float(egfr < 60))))
    return ProgressionForecast(
        expected_hba1c_12w=float(expected_hba1c_12w),
        expected_fpg_12w=float(expected_fpg_12w),
        risk_of_ckd_1y=float(min(max(risk_ckd, 0.0), 0.9)),
        risk_of_cvd_1y=float(min(max(risk_cvd, 0.0), 0.9)),
    )

