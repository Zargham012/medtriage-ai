"""
app.py
MedTriage AI — Multimodal Clinical Decision Support & Triage System
Streamlit app. Deploy free on Streamlit Community Cloud or Hugging Face Spaces.

Flow:
  1. User uploads ECG signal (CSV of raw values) OR picks a sample.
  2. User enters symptoms/history in a form.
  3. ECG model (ecg_model.py) classifies rhythm.
  4. RAG layer (rag_pipeline.py) retrieves relevant guideline snippets.
  5. Groq LLM (groq_pipeline.py) synthesizes everything into a triage report.
"""

import streamlit as st
import numpy as np
import pandas as pd

from ecg_model import predict as ecg_predict
from rag_pipeline import init_rag, retrieve
from groq_pipeline import generate_report

st.set_page_config(page_title="MedTriage AI", page_icon="🩺", layout="wide")

# ---- Sample ECG data so the demo works even without a real upload ----
SAMPLE_SIGNALS = {
    "Sample: Regular rhythm": np.sin(np.linspace(0, 40 * np.pi, 2500)) + np.random.normal(0, 0.05, 2500),
    "Sample: Irregular rhythm": np.sin(np.linspace(0, 40 * np.pi, 2500))
        + 0.6 * np.sin(np.linspace(0, 13 * np.pi, 2500))
        + np.random.normal(0, 0.15, 2500),
}


@st.cache_resource
def setup_rag():
    init_rag()
    return True


st.title("🩺 MedTriage AI")
st.caption(
    "Multimodal clinical decision **support** tool — ECG analysis + guideline-grounded "
    "LLM reasoning. **Not a diagnostic device.** For hackathon demonstration purposes only."
)

setup_rag()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. ECG Input")
    source = st.radio("Choose ECG source:", ["Use a sample signal", "Upload CSV"])

    signal = None
    if source == "Use a sample signal":
        choice = st.selectbox("Sample:", list(SAMPLE_SIGNALS.keys()))
        signal = SAMPLE_SIGNALS[choice]
        st.line_chart(signal[:500])
    else:
        uploaded = st.file_uploader("Upload a CSV with a single column of ECG values", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded, header=None)
            signal = df.iloc[:, 0].values
            st.line_chart(signal[:500])

    st.subheader("2. Patient Symptoms & History")
    symptoms = st.text_area(
        "Describe symptoms, duration, age, relevant history:",
        placeholder="e.g., 58-year-old male, palpitations for 2 hours, mild chest discomfort, "
                    "history of hypertension, no prior cardiac events.",
        height=120,
    )

    run_button = st.button("🔍 Analyze & Generate Triage Report", type="primary", use_container_width=True)

with col2:
    st.subheader("3. Results")

    if run_button:
        if signal is None:
            st.warning("Please provide an ECG signal first (sample or upload).")
        elif not symptoms.strip():
            st.warning("Please enter patient symptoms/history.")
        else:
            with st.spinner("Analyzing ECG signal..."):
                label, confidence, probs = ecg_predict(signal)

            st.markdown(f"**ECG Finding:** `{label}`  (confidence: {confidence:.0%})")
            st.bar_chart(pd.Series(probs))

            with st.spinner("Retrieving relevant clinical guidelines..."):
                query = f"{label} arrhythmia. Symptoms: {symptoms}"
                snippets = retrieve(query, k=3)

            with st.expander("📚 Retrieved guideline evidence (used for grounding)"):
                for s in snippets:
                    st.markdown(s)
                    st.divider()

            with st.spinner("Generating triage report with Groq LLM..."):
                try:
                    report = generate_report(label, confidence, symptoms, snippets)
                    st.markdown("### 📋 Triage Report")
                    st.markdown(report)
                except ValueError as e:
                    st.error(str(e))
    else:
        st.info("Fill in the ECG source and symptoms, then click Analyze.")

st.divider()
st.caption(
    "⚠️ This tool is a clinical decision **support** prototype built for a hackathon. "
    "It does not diagnose and must not be used for real patient care without professional "
    "medical oversight and regulatory clearance."
)
