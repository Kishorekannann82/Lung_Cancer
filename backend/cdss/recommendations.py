# ─────────────────────────────────────────────
#  cdss/recommendations.py
#  Clinical Decision Support using Groq API
#  Generates treatment, home remedies, 30-day plan
# ─────────────────────────────────────────────

from groq import Groq
import sys
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
from dotenv import load_dotenv
load_dotenv(dotenv_path=env_path)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)


# ── Build Clinical Prompt ─────────────────────
def build_prompt(patient_data: dict) -> str:
    classification  = patient_data.get('classification', 'Unknown')
    is_malignant    = classification == "Malignant"
    stage           = patient_data.get('cancer_stage', 'N/A')
    tumor_size      = patient_data.get('tumor_size', 'N/A')
    survival        = patient_data.get('survival_rate', 'N/A')

    return f"""
You are an expert oncology clinical decision support system providing a comprehensive personal care plan.

PATIENT PROFILE:
- Age: {patient_data.get('age')} years
- Smoking Status: {patient_data.get('smoking_status')}
- Symptoms: {', '.join(patient_data.get('symptoms', [])) or 'None reported'}

AI DIAGNOSIS RESULTS:
- Classification: {classification}
- Malignancy Probability: {float(patient_data.get('malignancy_prob', 0)) * 100:.1f}%
- Risk Tier: {patient_data.get('risk_tier')}
- Composite Risk Score: {patient_data.get('risk_score')}
- Cancer Stage: {stage}
- Estimated Tumor Size: {tumor_size}
- Estimated 5-Year Survival: {survival}%

Provide a comprehensive structured clinical report with ALL of these sections:

## DIAGNOSIS SUMMARY
Brief interpretation of findings including stage and prognosis.

## RECOMMENDED DIAGNOSTIC TESTS
List 4-5 specific follow-up tests with reasons.

## TREATMENT OPTIONS
{'Stage-specific cancer treatments including surgery, chemotherapy, radiation, targeted therapy, immunotherapy.' if is_malignant else 'Monitoring plan, lifestyle modifications, preventive treatments.'}

## PRESCRIBED MEDICINES
List specific medicines appropriate for this {'cancer stage' if is_malignant else 'benign condition'}:
- Medicine name, dosage, timing, purpose
- Include both prescription and OTC medications
- Include supplements (Vitamin D, antioxidants, etc.)
Note: These are suggestions to discuss with a physician.

## HOME REMEDIES & NATURAL SUPPORT
List practical daily home remedies:
- Morning routine (e.g., warm turmeric water, breathing exercises)
- Dietary recommendations (foods to eat and avoid)
- Herbal teas and natural drinks with timing
- Breathing exercises (Pranayama, pursed lip breathing)
- Steam inhalation schedule
- Sleep position recommendations

## 30-DAY PERSONAL TREATMENT SCHEDULE
Create a detailed daily schedule for 30 days:

WEEK 1 (Days 1-7): [Focus: Stabilization]
- Morning (6-8 AM): specific activities
- Breakfast: specific foods/drinks
- Mid-morning: medicines/exercises
- Lunch: dietary guidelines
- Evening: activities/remedies
- Night: medicines/routine

WEEK 2 (Days 8-14): [Focus: Active Treatment]
- Daily routine adjustments

WEEK 3 (Days 15-21): [Focus: Strengthening]
- Progressive routine

WEEK 4 (Days 22-30): [Focus: Monitoring & Recovery]
- Final phase routine

## FOLLOW-UP SCHEDULE
Specific medical appointments timeline.

## ORGAN DONOR SCREENING NOTE
Eligibility note based on findings.

Be specific with times, quantities, and durations. Make it practical and easy to follow at home.
"""


# ── Get CDSS Recommendations ──────────────────
def get_recommendations(patient_data: dict) -> dict:
    try:
        prompt   = build_prompt(patient_data)
        response = client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical oncology decision support AI. "
                        "Provide accurate, evidence-based medical recommendations including "
                        "home remedies, daily schedules, and medicines. "
                        "Always remind that AI recommendations must be reviewed by a licensed physician. "
                        "Be specific with timings, dosages, and practical daily routines."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens  = 2048,
            temperature = 0.3
        )

        recommendation_text = response.choices[0].message.content

        return {
            "success":         True,
            "recommendations": recommendation_text,
            "model_used":      GROQ_MODEL,
            "disclaimer":      "⚠️ AI-generated recommendations. Must be reviewed by a licensed physician before following any treatment plan."
        }

    except Exception as e:
        return {
            "success":         False,
            "recommendations": f"Could not generate recommendations: {str(e)}",
            "model_used":      GROQ_MODEL,
            "disclaimer":      "Please consult a physician directly."
        }


if __name__ == "__main__":
    test_patient = {
        "age":              65,
        "smoking_status":   "current",
        "symptoms":         ["persistent_cough", "weight_loss"],
        "classification":   "Malignant",
        "malignancy_prob":  0.87,
        "risk_tier":        "Critical",
        "risk_score":       0.81,
        "cancer_stage":     "III",
        "tumor_size":       "3-5 cm",
        "survival_rate":    16.0
    }

    print("[TEST] Calling Groq API...")
    result = get_recommendations(test_patient)

    if result["success"]:
        print("\n" + "=" * 60)
        print(result["recommendations"])
        print("=" * 60)
    else:
        print(f"[ERROR] {result['recommendations']}")