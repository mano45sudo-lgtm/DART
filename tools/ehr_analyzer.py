from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EHRSummary:
    problems: List[str]
    current_meds: List[str]
    allergies: List[str]
    key_risks: List[str]
    notes: str


def analyze_ehr(
    *,
    patient_profile: Dict[str, Any],
    patient_state: Dict[str, Any],
    prior_actions: Optional[List[Dict[str, Any]]] = None,
) -> EHRSummary:
    """
    Inputs:
      - patient_profile/state dicts (from env.state())
      - optional prior action history
    Output:
      - concise EHR-like summary for the agent
    """
    _ = prior_actions
    prof = patient_profile
    st = patient_state

    problems = ["type2_diabetes"]
    if bool(st.get("ckd")):
        problems.append("ckd")
    if bool(st.get("cvd")):
        problems.append("cvd")
    if bool(st.get("hypertension")):
        problems.append("hypertension")

    meds = []
    for drug in (st.get("meds") or {}).keys():
        meds.append(str(drug))

    allergies = []  # mock
    key_risks = []
    if float(st.get("egfr", 90.0)) < 30.0:
        key_risks.append("renal_impairment_severe")
    if float(st.get("hba1c", 8.0)) > 9.0:
        key_risks.append("poor_glycemic_control")
    if int(prof.get("age", 55)) > 70:
        key_risks.append("elderly_hypoglycemia_risk")

    notes = (
        f"Age {prof.get('age')} sex {prof.get('sex')}. "
        f"HbA1c {st.get('hba1c'):.2f}%, FPG {st.get('fasting_glucose'):.0f} mg/dL. "
        f"eGFR {st.get('egfr'):.0f}."
    )

    return EHRSummary(problems=problems, current_meds=meds, allergies=allergies, key_risks=key_risks, notes=notes)

