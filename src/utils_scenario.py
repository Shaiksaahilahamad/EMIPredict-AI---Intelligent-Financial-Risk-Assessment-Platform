from typing import Tuple, List
from src.config import EMI_SCENARIO_RULES

def validate_emi_scenario(emi_scenario: str, requested_amount: float, requested_tenure: int) -> Tuple[bool, List[str]]:
    messages = []
    rules = EMI_SCENARIO_RULES.get(emi_scenario)
    if rules is None:
        return False, [f"Unknown EMI scenario: {emi_scenario}"]

    ok = True

    if not (rules["amount_min"] <= requested_amount <= rules["amount_max"]):
        ok = False
        messages.append(
            f"Requested amount must be between {rules['amount_min']} and {rules['amount_max']} for {emi_scenario}."
        )

    if not (rules["tenure_min"] <= requested_tenure <= rules["tenure_max"]):
        ok = False
        messages.append(
            f"Requested tenure must be between {rules['tenure_min']} and {rules['tenure_max']} months for {emi_scenario}."
        )

    if ok:
        messages.append("Requested amount and tenure are within allowed limits for this scenario.")
    return ok, messages
