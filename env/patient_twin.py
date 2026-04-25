from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


DRUG_COST_WEEKLY_USD = {
    "none": 0.0,
    "lifestyle": 5.0,
    "metformin": 2.0,
    "sulfonylurea": 3.0,
    "dpp4": 12.0,
    "glp1": 45.0,
    "sglt2": 25.0,
    "insulin": 30.0,
}


DRUG_HBA1C_DELTA_12W = {
    # expected HbA1c change over ~12 weeks at standard dose (negative improves)
    "none": 0.0,
    "lifestyle": -0.6,
    "metformin": -1.0,
    "sulfonylurea": -1.2,
    "dpp4": -0.6,
    "glp1": -1.0,
    "sglt2": -0.7,
    "insulin": -1.6,
}


DRUG_WEIGHT_DELTA_12W = {
    "none": 0.0,
    "lifestyle": -1.2,
    "metformin": -0.3,
    "sulfonylurea": +0.7,
    "dpp4": 0.0,
    "glp1": -1.5,
    "sglt2": -0.8,
    "insulin": +0.8,
}


DRUG_SIDE_EFFECTS = {
    # (prob_per_week, severity 1-5, label)
    "metformin": (0.06, 2, "gi_upset"),
    "sulfonylurea": (0.05, 3, "hypoglycemia"),
    "dpp4": (0.01, 1, "headache"),
    "glp1": (0.07, 2, "nausea"),
    "sglt2": (0.02, 2, "uti_genital"),
    "insulin": (0.08, 3, "hypoglycemia"),
}


@dataclass
class PatientProfile:
    """Static patient factors."""

    patient_id: str = "P0000"
    age: int = 55
    sex: str = "U"  # M/F/U
    genetics: Dict[str, Any] = field(default_factory=dict)
    comorbidities: Dict[str, bool] = field(default_factory=dict)

    # baseline latent factors
    insulin_resistance: float = 0.7  # 0-1
    beta_cell_capacity: float = 0.55  # 0-1


@dataclass
class PatientState:
    """Dynamic state variables (weekly timestep)."""

    week: int = 0
    hba1c: float = 8.5
    fasting_glucose: float = 160.0
    bmi: float = 32.0
    systolic_bp: float = 135.0
    diastolic_bp: float = 85.0
    egfr: float = 90.0

    # severity / toxicity summary signals (0..1)
    disease_stage: float = 0.55
    side_effect_load: float = 0.0

    # derived / comorbidity flags (may flip with progression)
    hypertension: bool = True
    ckd: bool = False
    cvd: bool = False

    # regimen
    lifestyle_intensity: float = 0.3  # 0..1
    meds: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # drug -> {"dose":0..1,"weeks_on":int}
    time_on_treatment: int = 0  # weeks since current regimen started (max weeks_on)
    treatment_history: List[Dict[str, Any]] = field(default_factory=list)

    # event tracking
    last_side_effects: List[Dict[str, Any]] = field(default_factory=list)
    cumulative_cost_usd: float = 0.0
    adherence: float = 0.85  # 0..1


def compute_disease_stage(state: PatientState) -> float:
    """
    Map multi-biomarker status into a 0..1 severity scalar.
    0.0 ~ remission/normal, 1.0 ~ critical.
    """
    # Normalize
    h = _clip((state.hba1c - 5.6) / (12.5 - 5.6), 0.0, 1.0)
    g = _clip((state.fasting_glucose - 90.0) / (320.0 - 90.0), 0.0, 1.0)
    renal = _clip((60.0 - state.egfr) / 60.0, 0.0, 1.0)  # egfr <60 worsens
    bp = _clip((state.systolic_bp - 120.0) / 60.0, 0.0, 1.0)

    stage = 0.45 * h + 0.25 * g + 0.20 * renal + 0.10 * bp
    # hard bumps for complications
    if state.cvd:
        stage += 0.10
    if state.egfr < 30.0:
        stage += 0.10
    return float(_clip(stage, 0.0, 1.0))


def sample_patient_profile(rng: np.random.Generator, patient_id: str) -> PatientProfile:
    age = int(rng.integers(35, 81))
    sex = rng.choice(["M", "F"])

    bmi = float(rng.normal(31.0, 4.0))
    bmi = _clip(bmi, 20.0, 45.0)

    hypertension = bool(rng.random() < _sigmoid((bmi - 28.0) / 4.0))
    ckd = bool(rng.random() < (0.08 + 0.02 * (age > 65)))
    cvd = bool(rng.random() < (0.10 + 0.05 * (age > 65)))

    # genetics: simple SNP flags affecting response / risk
    genetics = {
        "PPARG_Pro12Ala": int(rng.random() < 0.18),
        "KCNJ11_E23K": int(rng.random() < 0.35),
    }

    # insulin resistance and beta cell capacity correlate with BMI/age
    insulin_resistance = _clip(_sigmoid((bmi - 27.0) / 4.5) + rng.normal(0, 0.05), 0.1, 0.98)
    beta_cell_capacity = _clip(0.75 - 0.0035 * (age - 35) - 0.12 * (insulin_resistance - 0.5) + rng.normal(0, 0.05), 0.12, 0.9)

    profile = PatientProfile(
        patient_id=patient_id,
        age=age,
        sex=sex,
        genetics=genetics,
        comorbidities={"hypertension": hypertension, "ckd": ckd, "cvd": cvd},
        insulin_resistance=insulin_resistance,
        beta_cell_capacity=beta_cell_capacity,
    )
    return profile


def initialize_patient_state(rng: np.random.Generator, profile: PatientProfile) -> PatientState:
    bmi = float(rng.normal(31.0, 4.0))
    bmi = _clip(bmi, 20.0, 45.0)

    # HbA1c sampling: mix of prediabetes + overt T2DM
    if rng.random() < 0.25:
        hba1c = float(rng.normal(6.1, 0.25))
    else:
        hba1c = float(rng.normal(8.4, 1.1))
    hba1c = _clip(hba1c, 5.6, 12.5)

    fasting_glucose = float(rng.normal(28.7 * hba1c - 46.7, 12.0))
    fasting_glucose = _clip(fasting_glucose, 85.0, 320.0)

    systolic_bp = float(rng.normal(128.0 + 0.8 * (bmi - 25.0) + 0.25 * (profile.age - 50), 10.0))
    diastolic_bp = float(rng.normal(80.0 + 0.35 * (bmi - 25.0), 7.0))
    systolic_bp = _clip(systolic_bp, 95.0, 190.0)
    diastolic_bp = _clip(diastolic_bp, 55.0, 120.0)

    # eGFR lower with age + CKD flag
    egfr = float(rng.normal(95.0 - 0.55 * (profile.age - 40), 10.0))
    if profile.comorbidities.get("ckd", False):
        egfr -= float(rng.normal(22.0, 8.0))
    egfr = _clip(egfr, 15.0, 125.0)

    state = PatientState(
        week=0,
        hba1c=hba1c,
        fasting_glucose=fasting_glucose,
        bmi=bmi,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        egfr=egfr,
        hypertension=profile.comorbidities.get("hypertension", False),
        ckd=profile.comorbidities.get("ckd", False) or egfr < 60.0,
        cvd=profile.comorbidities.get("cvd", False),
        lifestyle_intensity=float(_clip(rng.normal(0.25, 0.12), 0.0, 1.0)),
        meds={},
        last_side_effects=[],
        cumulative_cost_usd=0.0,
        adherence=float(_clip(rng.normal(0.85, 0.07), 0.55, 0.98)),
    )
    state.disease_stage = compute_disease_stage(state)
    state.side_effect_load = 0.0
    state.time_on_treatment = 0
    state.treatment_history = []
    return state


def _treatment_effect_weekly(
    rng: np.random.Generator, profile: PatientProfile, state: PatientState
) -> Tuple[float, float, float, List[Dict[str, Any]], float]:
    """
    Returns:
      dhba1c_week, dfpg_week, dbmi_week, side_effect_events, weekly_cost
    """
    side_effects: List[Dict[str, Any]] = []
    weekly_cost = 0.0

    # lifestyle as continuous intensity
    lifestyle = float(_clip(state.lifestyle_intensity, 0.0, 1.0))
    weekly_cost += DRUG_COST_WEEKLY_USD["lifestyle"] * lifestyle
    dhba1c = (DRUG_HBA1C_DELTA_12W["lifestyle"] / 12.0) * lifestyle
    dbmi = (DRUG_WEIGHT_DELTA_12W["lifestyle"] / 12.0) * lifestyle

    # medications (dose 0..1)
    for drug, meta in state.meds.items():
        dose = float(_clip(float(meta.get("dose", 1.0)), 0.0, 1.0))
        dhba1c += (DRUG_HBA1C_DELTA_12W.get(drug, 0.0) / 12.0) * dose
        dbmi += (DRUG_WEIGHT_DELTA_12W.get(drug, 0.0) / 12.0) * dose
        weekly_cost += DRUG_COST_WEEKLY_USD.get(drug, 0.0) * max(dose, 0.25)

        if drug in DRUG_SIDE_EFFECTS:
            p, severity, label = DRUG_SIDE_EFFECTS[drug]
            # intolerance higher if older/CKD for some classes
            risk_boost = 1.0 + 0.35 * (profile.age > 65) + 0.35 * (state.ckd and drug in {"metformin", "sglt2"})
            if rng.random() < _clip(p * risk_boost * max(dose, 0.4), 0.0, 0.4):
                side_effects.append({"drug": drug, "label": label, "severity": int(severity)})

    # genetics response modulation (~10% variance): simple scaling
    geno_scale = 1.0
    if profile.genetics.get("PPARG_Pro12Ala", 0) == 1:
        geno_scale *= 1.05
    if profile.genetics.get("KCNJ11_E23K", 0) == 1:
        geno_scale *= 0.98
    dhba1c *= geno_scale

    # adherence applies to net effect (and cost still incurred partially)
    effective_adherence = float(_clip(state.adherence + rng.normal(0, 0.03), 0.35, 0.99))
    dhba1c *= effective_adherence
    dbmi *= effective_adherence
    weekly_cost *= 0.6 + 0.4 * effective_adherence

    # map HbA1c change to fasting glucose change (rough)
    dfpg = float(rng.normal(28.0 * dhba1c, 4.0))
    return float(dhba1c), float(dfpg), float(dbmi), side_effects, float(weekly_cost)


def progression_step(
    rng: np.random.Generator, profile: PatientProfile, state: PatientState
) -> Tuple[PatientState, Dict[str, Any]]:
    """
    One-week stochastic progression with treatment effects + natural history.
    Returns updated state + info.
    """
    prev = PatientState(**{**state.__dict__})

    # Natural disease drift: insulin resistance + beta cell decline cause upward pressure on HbA1c.
    ir = profile.insulin_resistance
    beta = profile.beta_cell_capacity
    baseline_drift = 0.015 + 0.025 * ir + 0.012 * (1.0 - beta)  # HbA1c points/week
    baseline_drift += 0.008 * state.ckd + 0.006 * (profile.age > 65)

    dhba1c_tx, dfpg_tx, dbmi_tx, side_effects, weekly_cost = _treatment_effect_weekly(rng, profile, state)

    # stochasticity: HbA1c noise (measurement + physiology)
    noise_hba1c = float(rng.normal(0.0, 0.05))
    noise_bmi = float(rng.normal(0.0, 0.08))

    # update
    state.week += 1
    state.hba1c = _clip(state.hba1c + baseline_drift + dhba1c_tx + noise_hba1c, 4.8, 14.5)
    state.fasting_glucose = _clip(
        state.fasting_glucose + 28.7 * (baseline_drift + dhba1c_tx) + dfpg_tx + float(rng.normal(0.0, 6.0)),
        70.0,
        400.0,
    )
    state.bmi = _clip(state.bmi + dbmi_tx + noise_bmi, 16.0, 55.0)

    # BP drifts with BMI and control
    bp_noise = float(rng.normal(0.0, 1.5))
    state.systolic_bp = _clip(state.systolic_bp + 0.10 * (state.bmi - prev.bmi) + bp_noise, 85.0, 210.0)
    state.diastolic_bp = _clip(state.diastolic_bp + 0.06 * (state.bmi - prev.bmi) + 0.6 * bp_noise, 45.0, 130.0)

    # eGFR decline: faster with poor glycemic control + HTN
    egfr_decline = 0.04 + 0.015 * (state.hba1c - 7.0) + 0.01 * (state.systolic_bp > 140)
    egfr_decline += 0.05 * (profile.age > 70)
    state.egfr = _clip(state.egfr - egfr_decline + float(rng.normal(0, 0.15)), 10.0, 140.0)

    # comorbidity flags
    state.hypertension = bool(state.systolic_bp >= 130.0 or state.diastolic_bp >= 80.0)
    state.ckd = bool(state.egfr < 60.0)

    # CVD risk: increases with age + HbA1c + HTN + CKD
    cvd_p = 0.0005 + 0.0008 * max(profile.age - 40, 0) + 0.0012 * max(state.hba1c - 7.0, 0)
    cvd_p += 0.0010 * state.hypertension + 0.0015 * state.ckd
    cvd_p = _clip(cvd_p, 0.0, 0.05)
    cvd_event = bool((not state.cvd) and (rng.random() < cvd_p))
    state.cvd = bool(state.cvd or cvd_event)

    state.last_side_effects = side_effects
    state.cumulative_cost_usd = float(state.cumulative_cost_usd + weekly_cost)

    # update medication weeks_on
    for drug in list(state.meds.keys()):
        state.meds[drug]["weeks_on"] = int(state.meds[drug].get("weeks_on", 0) + 1)

    # time_on_treatment: longest-running active med (or 0)
    state.time_on_treatment = int(max([int(m.get("weeks_on", 0)) for m in state.meds.values()], default=0))

    # toxicity accumulation (side_effect_load): add severity, with small decay
    state.side_effect_load = float(_clip(state.side_effect_load * 0.92, 0.0, 1.0))
    if side_effects:
        add = sum(float(ev.get("severity", 1)) for ev in side_effects) / 5.0
        state.side_effect_load = float(_clip(state.side_effect_load + 0.10 * add, 0.0, 1.0))

    # update disease stage scalar
    state.disease_stage = compute_disease_stage(state)

    info = {
        "weekly_cost_usd": weekly_cost,
        "side_effects": side_effects,
        "cvd_event": cvd_event,
        "prev_state": prev,
    }
    return state, info


def apply_treatment_action(
    rng: np.random.Generator, profile: PatientProfile, state: PatientState, action: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply a (parsed) treatment action to the patient's regimen.
    This is designed to be compatible with Phase 4 action schema.
    """
    info: Dict[str, Any] = {"action_applied": True, "action_errors": []}

    a_type = str(action.get("type", "noop"))
    lifestyle = action.get("lifestyle", None)
    if lifestyle is not None:
        try:
            state.lifestyle_intensity = float(_clip(float(lifestyle), 0.0, 1.0))
        except Exception:
            info["action_errors"].append("invalid_lifestyle")

    def _start_or_adjust(drug: str, dose: float):
        dose = float(_clip(dose, 0.0, 1.0))
        if drug not in state.meds:
            state.meds[drug] = {"dose": dose, "weeks_on": 0}
        else:
            state.meds[drug]["dose"] = dose

    if a_type in {"start", "add", "dose_adjust"}:
        drug = str(action.get("drug", "none"))
        dose = float(action.get("dose", 1.0))

        # contraindications (soft enforce): block if unsafe
        if drug == "metformin" and state.egfr < 30.0:
            info["action_errors"].append("contra_metformin_egfr<30")
        elif drug == "sglt2" and state.egfr < 30.0:
            info["action_errors"].append("contra_sglt2_egfr<30")
        else:
            _start_or_adjust(drug, dose)

    elif a_type == "stop":
        drug = str(action.get("drug", "none"))
        if drug in state.meds:
            state.meds.pop(drug, None)

    elif a_type == "switch":
        from_drug = str(action.get("from_drug", "none"))
        to_drug = str(action.get("to_drug", "none"))
        dose = float(action.get("dose", 1.0))
        if from_drug in state.meds:
            state.meds.pop(from_drug, None)
        _start_or_adjust(to_drug, dose)

    elif a_type == "noop":
        pass
    else:
        info["action_errors"].append("unknown_action_type")

    # small adherence perturbation with regimen complexity / side effects expectation
    complexity = len(state.meds)
    state.adherence = float(_clip(state.adherence - 0.01 * max(complexity - 1, 0) + rng.normal(0, 0.01), 0.35, 0.99))

    # record in treatment history (high-level; keeps the blueprint requirement)
    state.treatment_history.append(
        {
            "week": int(state.week),
            "action": dict(action),
            "active_meds": {k: {"dose": float(v.get("dose", 0.0)), "weeks_on": int(v.get("weeks_on", 0))} for k, v in state.meds.items()},
            "lifestyle": float(state.lifestyle_intensity),
        }
    )
    return info


class PatientTwin:
    """Patient twin with stochastic progression."""

    def __init__(self, profile: Optional[PatientProfile] = None, state: Optional[PatientState] = None, *, seed: Optional[int] = None):
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.profile = profile or sample_patient_profile(self.rng, patient_id="P0000")
        self.state = state or initialize_patient_state(self.rng, self.profile)

    def reset(self, *, seed: Optional[int] = None, patient_id: str = "P0000") -> PatientState:
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        self.profile = sample_patient_profile(self.rng, patient_id=patient_id)
        self.state = initialize_patient_state(self.rng, self.profile)
        return self.state

    def apply_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return apply_treatment_action(self.rng, self.profile, self.state, action)

    def step(self) -> Dict[str, Any]:
        _, info = progression_step(self.rng, self.profile, self.state)
        return info

    def as_dict(self) -> Dict[str, Any]:
        return {
            "profile": {
                "patient_id": self.profile.patient_id,
                "age": self.profile.age,
                "sex": self.profile.sex,
                "genetics": self.profile.genetics,
                "comorbidities": self.profile.comorbidities,
                "insulin_resistance": self.profile.insulin_resistance,
                "beta_cell_capacity": self.profile.beta_cell_capacity,
            },
            "state": {
                "week": self.state.week,
                "hba1c": self.state.hba1c,
                "fasting_glucose": self.state.fasting_glucose,
                "bmi": self.state.bmi,
                "systolic_bp": self.state.systolic_bp,
                "diastolic_bp": self.state.diastolic_bp,
                "egfr": self.state.egfr,
                "disease_stage": self.state.disease_stage,
                "side_effect_load": self.state.side_effect_load,
                "hypertension": self.state.hypertension,
                "ckd": self.state.ckd,
                "cvd": self.state.cvd,
                "lifestyle_intensity": self.state.lifestyle_intensity,
                "meds": self.state.meds,
                "time_on_treatment": self.state.time_on_treatment,
                "treatment_history": self.state.treatment_history[-50:],
                "last_side_effects": self.state.last_side_effects,
                "cumulative_cost_usd": self.state.cumulative_cost_usd,
                "adherence": self.state.adherence,
            },
        }

