"""
groq_pipeline.py
Calls Groq's free/fast LLM API to synthesize ECG model output + symptoms +
retrieved clinical guidelines into a structured triage report.
"""

import os
from groq import Groq

MODEL_NAME = "llama-3.3-70b-versatile"  # fast + capable, check Groq console for current free models

SYSTEM_PROMPT = """You are a clinical decision SUPPORT assistant, not a diagnostic authority.
You help triage patients by reasoning over (1) an automated ECG finding with confidence score,
(2) patient-reported symptoms, and (3) retrieved clinical guideline excerpts.

Rules you MUST follow:
- Base your reasoning primarily on the retrieved guideline excerpts provided. Cite them by source name.
- Never claim certainty. Always express appropriate clinical uncertainty.
- Always include a triage urgency level using the ESI scale (1=most urgent, 5=least urgent).
- Always end with: "This is a decision-support suggestion only and must be reviewed by a qualified clinician before any action is taken."
- Keep the tone calm, clear, and professional — this may be read by a non-specialist.

Output in this exact structure:
1. **Summary of Findings**
2. **Possible Differential Considerations** (ranked, with brief reasoning)
3. **Triage Urgency (ESI Level)** with justification
4. **Recommended Next Steps**
5. **Guideline Sources Used**
"""


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
            "and set it as an environment variable or Streamlit secret."
        )
    return Groq(api_key=api_key)


def build_prompt(ecg_label, ecg_confidence, symptoms_text, retrieved_snippets):
    snippets_block = "\n\n".join(retrieved_snippets)
    prompt = f"""
PATIENT DATA:
- ECG automated finding: {ecg_label} (model confidence: {ecg_confidence:.0%})
- Patient-reported symptoms / history: {symptoms_text}

RETRIEVED CLINICAL GUIDELINE EXCERPTS:
{snippets_block}

Based on the above, produce the structured triage report as instructed.
"""
    return prompt


def generate_report(ecg_label, ecg_confidence, symptoms_text, retrieved_snippets):
    client = get_client()
    user_prompt = build_prompt(ecg_label, ecg_confidence, symptoms_text, retrieved_snippets)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=900,
    )
    return response.choices[0].message.content
