# ─────────────────────────────────────────────
#  cdss/recommendations.py
#  Clinical Decision Support using Groq API
#  Generates treatment, prevention, follow-up
# ─────────────────────────────────────────────

from groq import Groq
import sys
from pathlib import Path

# Load .env from backend root regardless of working directory
env_path = Path(__file__).resolve().parent.parent / ".env"
from dotenv import load_dotenv
load_dotenv(dotenv_path=env_path)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import GROQ_API_KEY, GROQ_MODEL


client = Groq(api_key=GROQ_API_KEY)


# ── Build Clinical Prompt ─────────────────────
def build_prompt(patient_data: dict) -> str:
    return f"""
You are an expert oncology clinical decision support system.
A patient has been assessed for pulmonary malignancy with the following results:

PATIENT PROFILE:
- Age: {patient_data.get('age')} years
- Smoking Status: {patient_data.get('smoking_status')}
- Symptoms: {', '.join(patient_data.get('symptoms', []))}

AI DIAGNOSIS RESULTS:
- Classification: {patient_data.get('classification')}
- Malignancy Probability: {patient_data.get('malignancy_prob') * 100:.1f}%
- Risk Tier: {patient_data.get('risk_tier')}
- Composite Risk Score: {patient_data.get('risk_score')}

Based on this, provide a structured clinical report with:

1. DIAGNOSIS SUMMARY
   Brief interpretation of the AI findings.

2. RECOMMENDED DIAGNOSTIC TESTS
   List 3-4 specific follow-up tests (e.g., PET scan, biopsy, bronchoscopy).

3. TREATMENT OPTIONS
   Based on risk tier, suggest appropriate treatments:
   - Surgery (lobectomy/pneumonectomy if applicable)
   - Chemotherapy regimens
   - Radiation therapy
   - Targeted molecular therapy

4. FOLLOW-UP SCHEDULE
   Specific timeline for monitoring (e.g., CT scan every 3 months).

5. PREVENTIVE RECOMMENDATIONS
   Lifestyle and preventive measures for this patient.

6. ORGAN DONOR SCREENING NOTE
   Brief note on lung donor eligibility based on findings.

Keep the response clear, professional, and suitable for a clinician.
Format each section with its heading. Be specific with medical recommendations.
"""


# ── Get CDSS Recommendations ──────────────────
def get_recommendations(patient_data: dict) -> dict:
    """
    Call Groq API with patient + AI diagnosis data.

    Args:
        patient_data: dict with keys:
            - age, smoking_status, symptoms
            - classification, malignancy_prob
            - risk_tier, risk_score

    Returns:
        dict with 'recommendations' text and 'model_used'
    """
    try:
        prompt   = build_prompt(patient_data)
        response = client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical oncology decision support AI. "
                        "Provide accurate, evidence-based medical recommendations. "
                        "Always remind that AI recommendations must be reviewed by a licensed physician."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens  = 1024,
            temperature = 0.3    # Low temp for consistent medical advice
        )

        recommendation_text = response.choices[0].message.content

        return {
            "success":         True,
            "recommendations": recommendation_text,
            "model_used":      GROQ_MODEL,
            "disclaimer":      "⚠️ AI-generated recommendations. Must be reviewed by a licensed physician."
        }

    except Exception as e:
        return {
            "success":         False,
            "recommendations": f"Could not generate recommendations: {str(e)}",
            "model_used":      GROQ_MODEL,
            "disclaimer":      "Please consult a physician directly."
        }


if __name__ == "__main__":
    # Quick test
    test_patient = {
        "age":              65,
        "smoking_status":   "current",
        "symptoms":         ["persistent_cough", "weight_loss"],
        "classification":   "Malignant",
        "malignancy_prob":  0.87,
        "risk_tier":        "Critical",
        "risk_score":       0.81
    }

    print("[TEST] Calling Groq API...")
    result = get_recommendations(test_patient)

    if result["success"]:
        print("\n" + "=" * 60)
        print(result["recommendations"])
        print("=" * 60)
        print(f"\n{result['disclaimer']}")
    else:
        print(f"[ERROR] {result['recommendations']}")
