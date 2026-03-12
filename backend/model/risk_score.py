# ─────────────────────────────────────────────
#  model/risk_score.py
#  Composite Risk Score (Equations 12 & 13)
#  R = α1*P(M|x) + α2*C_age + α3*C_smoke + α4*C_hist
# ─────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ALPHA_1, ALPHA_2, ALPHA_3, ALPHA_4, RISK_LOW, RISK_MEDIUM, RISK_HIGH


# ── Age Risk Factor ───────────────────────────
def compute_age_factor(age: int) -> float:
    """
    Normalize age to [0,1] risk coefficient.
    Higher age = higher lung cancer risk.
    """
    if age < 40:  return 0.1
    if age < 50:  return 0.3
    if age < 60:  return 0.5
    if age < 70:  return 0.7
    return 0.9


# ── Smoking History Factor ────────────────────
def compute_smoking_factor(smoking_status: str) -> float:
    """
    Map smoking history to risk coefficient.
    """
    mapping = {
        "never":   0.1,
        "former":  0.5,
        "current": 0.9
    }
    return mapping.get(smoking_status.lower(), 0.1)


# ── Clinical History Factor ───────────────────
def compute_history_factor(symptoms: list) -> float:
    """
    Map reported symptoms to clinical history risk factor.
    symptoms: list of strings e.g. ['cough', 'chest_pain']
    """
    high_risk_symptoms = {
        "hemoptysis":        0.9,   # Coughing blood
        "weight_loss":       0.7,
        "chest_pain":        0.6,
        "persistent_cough":  0.5,
        "shortness_breath":  0.5,
        "fatigue":           0.3,
        "none":              0.0
    }
    if not symptoms:
        return 0.0

    scores = [high_risk_symptoms.get(s.lower(), 0.2) for s in symptoms]
    return min(max(scores), 1.0)   # Take max symptom risk, cap at 1


# ── Composite Risk Score — Equation 12 ───────
def compute_risk_score(
    malignancy_prob: float,
    age: int,
    smoking_status: str,
    symptoms: list
) -> dict:
    """
    R = α1*P(M|x) + α2*C_age + α3*C_smoke + α4*C_hist

    Args:
        malignancy_prob: CNN output probability for malignant class
        age:             Patient age
        smoking_status:  'never' | 'former' | 'current'
        symptoms:        List of symptom strings

    Returns:
        dict with score, tier, and individual factors
    """
    c_age   = compute_age_factor(age)
    c_smoke = compute_smoking_factor(smoking_status)
    c_hist  = compute_history_factor(symptoms)

    # Equation 12
    R = (ALPHA_1 * malignancy_prob +
         ALPHA_2 * c_age +
         ALPHA_3 * c_smoke +
         ALPHA_4 * c_hist)

    R = round(min(R, 1.0), 4)

    # Equation 13 — Risk Tier
    if R < RISK_LOW:
        tier  = "Low"
        color = "green"
    elif R < RISK_MEDIUM:
        tier  = "Medium"
        color = "yellow"
    elif R < RISK_HIGH:
        tier  = "High"
        color = "orange"
    else:
        tier  = "Critical"
        color = "red"

    return {
        "risk_score":       R,
        "risk_tier":        tier,
        "risk_color":       color,
        "malignancy_prob":  round(malignancy_prob, 4),
        "age_factor":       round(c_age, 4),
        "smoking_factor":   round(c_smoke, 4),
        "history_factor":   round(c_hist, 4)
    }


if __name__ == "__main__":
    # Quick test
    result = compute_risk_score(
        malignancy_prob = 0.82,
        age             = 65,
        smoking_status  = "current",
        symptoms        = ["persistent_cough", "weight_loss"]
    )
    print("Risk Assessment Result:")
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
