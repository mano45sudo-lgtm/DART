from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class InteractionFinding:
    severity: str  # "low"|"moderate"|"high"
    message: str


@dataclass
class InteractionReport:
    ok: bool
    findings: List[InteractionFinding]


def check_drug_interactions(
    *,
    current_meds: Dict[str, Dict[str, Any]],
    proposed_action: Dict[str, Any],
    egfr: float,
    age: int,
) -> InteractionReport:
    """
    Inputs:
      - current_meds: regimen map drug -> meta
      - proposed_action: parsed action dict
      - egfr/age: key contraindication axes
    Outputs:
      - ok: whether action is allowed
      - findings: list of warnings/blocks (mock logic)
    """
    findings: List[InteractionFinding] = []
    ok = True

    a_type = str(proposed_action.get("type", "noop"))
    drug = str(proposed_action.get("drug", proposed_action.get("to_drug", "none")))
    current = set((current_meds or {}).keys())

    if drug == "metformin" and egfr < 30.0:
        ok = False
        findings.append(InteractionFinding("high", "Metformin contraindicated: eGFR < 30."))

    if drug == "sglt2" and egfr < 30.0:
        ok = False
        findings.append(InteractionFinding("high", "SGLT2i contraindicated: eGFR < 30."))

    if a_type in {"start", "add"} and drug == "sulfonylurea" and age >= 70:
        findings.append(InteractionFinding("moderate", "Elderly: sulfonylurea increases hypoglycemia risk."))

    if "insulin" in current and drug == "sulfonylurea" and a_type in {"start", "add"}:
        findings.append(InteractionFinding("moderate", "Insulin + sulfonylurea increases hypoglycemia risk."))

    if not findings:
        findings.append(InteractionFinding("low", "No major interactions detected (mock)."))

    return InteractionReport(ok=ok, findings=findings)

