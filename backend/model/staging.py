# ─────────────────────────────────────────────
#  model/staging.py
#  Cancer Stage (I-IV), Tumor Size Estimation,
#  Survival Rate % based on CNN + risk factors
# ─────────────────────────────────────────────

import numpy as np


# ── Tumor Size Estimation from Grad-CAM ──────
def estimate_tumor_size(gradcam_heatmap: np.ndarray, malignancy_prob: float) -> dict:
    """
    Estimate tumor size from Grad-CAM activation area.
    Higher activation area + high malignancy = larger tumor estimate.
    Returns size category and estimated cm range.
    """
    if malignancy_prob < 0.5:
        return {
            "size_category": "N/A",
            "size_range_cm": "N/A",
            "t_classification": "T0",
            "activation_percent": 0.0
        }

    # Threshold heatmap at 50% of max activation
    threshold   = gradcam_heatmap.max() * 0.5
    active_mask = gradcam_heatmap > threshold
    activation_percent = round(float(active_mask.sum()) / active_mask.size * 100, 1)

    # Scale activation % to tumor size using malignancy probability
    weighted = activation_percent * malignancy_prob

    if weighted < 5:
        size_cat = "Small"
        size_cm  = "< 1 cm"
        t_class  = "T1a"
    elif weighted < 10:
        size_cat = "Small-Medium"
        size_cm  = "1 – 2 cm"
        t_class  = "T1b"
    elif weighted < 18:
        size_cat = "Medium"
        size_cm  = "2 – 3 cm"
        t_class  = "T1c"
    elif weighted < 28:
        size_cat = "Medium-Large"
        size_cm  = "3 – 5 cm"
        t_class  = "T2"
    elif weighted < 40:
        size_cat = "Large"
        size_cm  = "5 – 7 cm"
        t_class  = "T3"
    else:
        size_cat = "Very Large"
        size_cm  = "> 7 cm"
        t_class  = "T4"

    return {
        "size_category":      size_cat,
        "size_range_cm":      size_cm,
        "t_classification":   t_class,
        "activation_percent": activation_percent
    }


# ── Cancer Stage (I–IV) ───────────────────────
def determine_cancer_stage(
    malignancy_prob: float,
    risk_score:      float,
    risk_tier:       str,
    t_classification: str,
    age:             int,
    smoking_status:  str
) -> dict:
    """
    Determine cancer stage based on CNN probability,
    risk score, T-classification and clinical factors.
    """
    if malignancy_prob < 0.5:
        return {
            "stage":        "N/A",
            "stage_roman":  "N/A",
            "stage_detail": "No malignancy detected",
            "n_class":      "N0",
            "m_class":      "M0",
            "description":  "Benign — no staging required"
        }

    # N classification (node involvement) from risk factors
    if risk_score < 0.4:
        n_class = "N0"   # No node involvement
    elif risk_score < 0.6:
        n_class = "N1"   # Ipsilateral nodes
    elif risk_score < 0.8:
        n_class = "N2"   # Mediastinal nodes
    else:
        n_class = "N3"   # Contralateral nodes

    # M classification (metastasis) from high risk + age
    if risk_score >= 0.75 and age >= 60:
        m_class = "M1"
    else:
        m_class = "M0"

    # Stage determination from TNM
    t_num = int(t_classification[1]) if t_classification[1].isdigit() else 1

    if m_class == "M1":
        stage, roman, detail = 4, "IV", "Distant metastasis present"
    elif n_class in ("N2", "N3"):
        stage, roman, detail = 3, "III", "Regional lymph node spread"
    elif n_class == "N1" or t_num >= 3:
        stage, roman, detail = 2, "II", "Local spread beyond lung"
    elif t_num <= 2 and n_class == "N0":
        if malignancy_prob < 0.80:
            stage, roman, detail = 1, "IA", "Localized — early stage"
        else:
            stage, roman, detail = 1, "IB", "Localized — slightly advanced"
    else:
        stage, roman, detail = 2, "II", "Local regional involvement"

    return {
        "stage":        stage,
        "stage_roman":  roman,
        "stage_detail": detail,
        "n_class":      n_class,
        "m_class":      m_class,
        "description":  _stage_description(roman)
    }


def _stage_description(roman: str) -> str:
    descriptions = {
        "IA":  "Tumor ≤3cm, confined to lung. Best prognosis.",
        "IB":  "Tumor 3–4cm, confined to lung. Good prognosis.",
        "II":  "Tumor larger or nearby lymph nodes involved.",
        "III": "Spread to mediastinal lymph nodes. Complex treatment needed.",
        "IV":  "Distant metastasis detected. Palliative care focus.",
        "N/A": "No malignancy detected."
    }
    return descriptions.get(roman, "")


# ── Survival Rate % ───────────────────────────
def estimate_survival_rate(
    stage_roman:     str,
    malignancy_prob: float,
    age:             int,
    smoking_status:  str
) -> dict:
    """
    5-year survival rate estimate based on SEER database statistics
    adjusted for patient-specific factors.
    """
    if stage_roman == "N/A":
        return {
            "five_year_survival": 95.0,
            "survival_category":  "Excellent",
            "survival_color":     "green",
            "note": "Benign — normal life expectancy"
        }

    # Base 5-year survival rates (SEER 2023 data)
    base_rates = {
        "IA":  68.0,
        "IB":  60.0,
        "II":  36.0,
        "III": 16.0,
        "IV":   7.0
    }

    base = base_rates.get(stage_roman, 20.0)

    # Adjust for age
    if age < 50:   age_adj = +5.0
    elif age < 60: age_adj = +2.0
    elif age < 70: age_adj = 0.0
    else:          age_adj = -5.0

    # Adjust for smoking
    smoke_adj = {
        "never":   +4.0,
        "former":  0.0,
        "current": -6.0
    }.get(smoking_status.lower(), 0.0)

    # Adjust for CNN confidence (higher confidence = more certain diagnosis)
    confidence_adj = -round((malignancy_prob - 0.5) * 10, 1)

    final = round(max(2.0, min(95.0, base + age_adj + smoke_adj + confidence_adj)), 1)

    if final >= 60:
        category, color = "Good", "green"
    elif final >= 35:
        category, color = "Moderate", "yellow"
    elif final >= 15:
        category, color = "Guarded", "orange"
    else:
        category, color = "Poor", "red"

    return {
        "five_year_survival": final,
        "survival_category":  category,
        "survival_color":     color,
        "note": f"Estimated 5-year survival based on SEER data (Stage {stage_roman})"
    }


# ── Full Staging Report ───────────────────────
def generate_staging_report(
    malignancy_prob: float,
    risk_score:      float,
    risk_tier:       str,
    gradcam_heatmap,
    age:             int,
    smoking_status:  str
) -> dict:
    """
    Master function — returns complete staging report.
    """
    tumor   = estimate_tumor_size(gradcam_heatmap, malignancy_prob)
    stage   = determine_cancer_stage(
                malignancy_prob, risk_score, risk_tier,
                tumor["t_classification"], age, smoking_status)
    survival = estimate_survival_rate(
                stage["stage_roman"], malignancy_prob, age, smoking_status)

    return {
        "tumor_size":    tumor,
        "cancer_stage":  stage,
        "survival_rate": survival
    }