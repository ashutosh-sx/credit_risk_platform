from typing import Dict, Any, List

def evaluate_credit_policy_rules(client_data: Dict[str, Any], ml_probability: float = None) -> Dict[str, Any]:
    """
    Evaluates bank credit policy rules on applicant attributes and ML predictions.
    Bridges ML predictions and credit policies with deterministic business rules.
    """
    triggered_rules = []
    decision = "APPROVED"
    risk_band = "Low"
    policy_recommendations = []
    
    # Extract applicant metrics
    ext_1 = client_data.get('EXT_SOURCE_1', 0.5)
    ext_2 = client_data.get('EXT_SOURCE_2', 0.5)
    ext_3 = client_data.get('EXT_SOURCE_3', 0.5)
    income = client_data.get('AMT_INCOME_TOTAL', 150000.0)
    credit = client_data.get('AMT_CREDIT', 500000.0)
    annuity = client_data.get('AMT_ANNUITY', 25000.0)
    
    # Convert age (DAYS_BIRTH is negative days)
    days_birth = client_data.get('DAYS_BIRTH', -15000)
    age = abs(days_birth) / 365.25 if days_birth else 40.0
    
    # Calculate key ratios
    debt_to_income_ratio = (annuity * 12.0) / income if income > 0 else 1.0
    
    # Rule 1: ML Score Guardrail
    if ml_probability is not None:
        if ml_probability >= 0.40:
            decision = "REJECTED"
            risk_band = "High"
            triggered_rules.append("Rule ML-01: ML Default Probability exceeds maximum risk tolerance threshold (>= 40%).")
            policy_recommendations.append("Applicant classified in High Risk Band. Decline credit request.")
        elif ml_probability >= 0.15:
            decision = "MANUAL_REVIEW"
            risk_band = "Medium"
            triggered_rules.append("Rule ML-02: ML Default Probability is in elevated alert tier (15% - 40%).")
            policy_recommendations.append("Refer applicant for manual underwriting review.")

    # Rule 2: Hard Credit Knock-Outs (External ratings)
    if ext_2 < 0.22 or ext_3 < 0.18:
        decision = "REJECTED"
        risk_band = "High"
        triggered_rules.append(f"Rule CR-01: Critical External Credit Rating knock-out triggered (EXT_2={ext_2:.2f}, EXT_3={ext_3:.2f}).")
        policy_recommendations.append("Immediate auto-rejection due to substandard external bureau scores.")

    # Rule 3: Debt Service Coverage (Affordability)
    if debt_to_income_ratio > 0.45:
        decision = "REJECTED"
        risk_band = "High"
        triggered_rules.append(f"Rule AF-01: Debt-to-Income (DTI) ratio is critically high ({debt_to_income_ratio * 100:.1f}%).")
        policy_recommendations.append("Decline due to severe affordability constraints (DTI exceeds max policy threshold of 45%).")
    elif debt_to_income_ratio > 0.30 and decision == "APPROVED":
        decision = "MANUAL_REVIEW"
        risk_band = "Medium"
        triggered_rules.append(f"Rule AF-02: Elevated Debt-to-Income (DTI) ratio ({debt_to_income_ratio * 100:.1f}%).")
        policy_recommendations.append("Refer to underwriter to verify stable secondary income sources.")

    # Rule 4: Youth & Thin Credit File Policy
    if age < 23.0 and (ext_1 is None or ext_1 < 0.30):
        if decision == "APPROVED":
            decision = "MANUAL_REVIEW"
            risk_band = "Medium"
            triggered_rules.append(f"Rule PL-01: Young applicant (<23 yrs) with thin bureau history file.")
            policy_recommendations.append("Refer to underwriter to request co-signer or proof of employment stability.")

    # Rule 5: Auto-Approval Super Prime Policy
    if ext_2 >= 0.70 and ext_3 >= 0.65 and debt_to_income_ratio <= 0.15 and ml_probability < 0.05:
        if decision == "APPROVED":
            risk_band = "Low"
            triggered_rules.append("Rule SP-01: Super Prime Applicant status achieved.")
            policy_recommendations.append("Auto-approved under Fast-Track Super Prime policy. Pre-approved for credit limit increases.")

    # Format output dictionary
    return {
        'Decision': decision,
        'Risk Band': risk_band,
        'Triggered Rules': triggered_rules if triggered_rules else ["No policy rule violations. Standard approval criteria met."],
        'DTI Ratio': debt_to_income_ratio,
        'Recommendations': policy_recommendations if policy_recommendations else ["Process standard contract approval."]
    }
