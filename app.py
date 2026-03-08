"""
app.py — Heart Risk Clinical Decision Support Tool
Streamlit frontend for the heart-risk-mlops pipeline.
Designed for clinical users — no ML jargon, plain English throughout.
"""

import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import streamlit as st
import yaml

# ── Path setup ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Risk Assessment",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .stApp { font-family: 'Segoe UI', sans-serif; }

    .header-box {
        background: linear-gradient(135deg, #1B2A4A 0%, #0D7377 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .header-box h1 { color: white; margin: 0; font-size: 1.8rem; }
    .header-box p  { color: #B2DFDB; margin: 0.3rem 0 0; font-size: 0.95rem; }

    .result-high {
        background: #FEE2E2; border-left: 6px solid #DC2626;
        padding: 1.5rem; border-radius: 8px; margin: 1rem 0;
    }
    .result-low {
        background: #D1FAE5; border-left: 6px solid #059669;
        padding: 1.5rem; border-radius: 8px; margin: 1rem 0;
    }
    .result-moderate {
        background: #FEF3C7; border-left: 6px solid #D97706;
        padding: 1.5rem; border-radius: 8px; margin: 1rem 0;
    }
    .result-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.4rem; }
    .result-sub   { font-size: 0.95rem; color: #374151; }

    .disclaimer {
        background: #F1F5F9; border: 1px solid #CBD5E1;
        padding: 1rem 1.5rem; border-radius: 8px;
        font-size: 0.82rem; color: #64748B; margin-top: 2rem;
    }
    .section-title {
        font-size: 1.1rem; font-weight: 600;
        color: #1B2A4A; margin: 1.5rem 0 0.8rem;
        border-bottom: 2px solid #0D7377; padding-bottom: 0.3rem;
    }
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        font-weight: 500; color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)


# ── Load config ───────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open(ROOT / "config" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


# ── Load model from MLflow registry ──────────────────────────
@st.cache_resource
def load_model():
    cfg = load_config()
    tracking_uri = (ROOT / cfg["mlflow"]["tracking_uri"]).as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{cfg['mlflow']['model_name']}/latest"
    return mlflow.sklearn.load_model(model_uri)


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🫀 Heart Risk Assessment Tool</h1>
    <p>Clinical decision support · Powered by machine learning · For screening purposes only</p>
</div>
""", unsafe_allow_html=True)

# ── Load model (with spinner) ─────────────────────────────────
with st.spinner("Loading clinical model..."):
    try:
        model = load_model()
        st.success("Model loaded successfully", icon="✅")
    except Exception as e:
        st.error(f"Could not load model from registry: {e}")
        st.info("Run `python src/train.py` first to register a model.")
        st.stop()

# ─────────────────────────────────────────────────────────────
# PATIENT INPUT FORM
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Patient Clinical Information</div>',
            unsafe_allow_html=True)
st.caption("Fill in the patient's latest clinical measurements and select the appropriate options.")

with st.form("patient_form"):

    # ── Row 1: Basic demographics ─────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=20, max_value=100, value=55, step=1)
    with col2:
        sex = st.selectbox("Biological Sex", options=["Male", "Female"])
    with col3:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", options=["No", "Yes"])

    # ── Row 2: Cardiovascular measurements ───────────────────
    st.markdown('<div class="section-title">Cardiovascular Measurements</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        trestbps = st.number_input("Resting Blood Pressure (mmHg)",
                                   min_value=80, max_value=220, value=130, step=1)
    with col2:
        chol = st.number_input("Serum Cholesterol (mg/dL)",
                                min_value=100, max_value=600, value=240, step=1)
    with col3:
        thalach = st.number_input("Max Heart Rate Achieved (bpm)",
                                   min_value=60, max_value=220, value=150, step=1)

    # ── Row 3: ECG and stress test ────────────────────────────
    st.markdown('<div class="section-title">ECG and Stress Test Results</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cp = st.selectbox("Chest Pain Type", options=[
            "Typical Angina",
            "Atypical Angina",
            "Non-Anginal Pain",
            "Asymptomatic",
        ])
        restecg = st.selectbox("Resting ECG Result", options=[
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy",
        ])
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[
            "Upsloping",
            "Flat",
            "Downsloping",
        ])
    with col2:
        exang = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
        oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)",
                                   min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                   format="%.1f")
        ca = st.selectbox("Number of Major Vessels Coloured by Fluoroscopy", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia Type", options=[
            "Normal",
            "Fixed Defect",
            "Reversible Defect",
        ])

    # ── Submit ────────────────────────────────────────────────
    submitted = st.form_submit_button(
        "🔍 Assess Heart Disease Risk",
        use_container_width=True,
        type="primary"
    )

# ─────────────────────────────────────────────────────────────
# ENCODE + PREDICT
# ─────────────────────────────────────────────────────────────
if submitted:

    # Map plain English back to numeric codes the model expects
    input_data = {
        "age":      age,
        "sex":      1.0 if sex == "Male" else 0.0,
        "cp":       ["Typical Angina", "Atypical Angina",
                     "Non-Anginal Pain", "Asymptomatic"].index(cp) + 1,
        "trestbps": float(trestbps),
        "chol":     float(chol),
        "fbs":      1.0 if fbs == "Yes" else 0.0,
        "restecg":  ["Normal", "ST-T Wave Abnormality",
                     "Left Ventricular Hypertrophy"].index(restecg),
        "thalach":  float(thalach),
        "exang":    1.0 if exang == "Yes" else 0.0,
        "oldpeak":  float(oldpeak),
        "slope":    ["Upsloping", "Flat", "Downsloping"].index(slope) + 1,
        "ca":       float(ca),
        "thal":     [None, None, None, "Normal",
                     None, None, "Fixed Defect", "Reversible Defect"
                     ].index(thal) if thal != "Normal" else 3.0,
    }

    # Thal encoding (UCI specific): Normal=3, Fixed=6, Reversible=7
    thal_map = {"Normal": 3.0, "Fixed Defect": 6.0, "Reversible Defect": 7.0}
    input_data["thal"] = thal_map[thal]

    df_input = pd.DataFrame([input_data])

    with st.spinner("Analysing patient data..."):
        prediction  = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        prob_pct    = round(probability * 100, 1)

    # ── Risk classification ───────────────────────────────────
    if probability < 0.35:
        risk_level = "Low Risk"
        css_class  = "result-low"
        emoji      = "🟢"
        explanation = (
            f"Based on the provided clinical values, this patient shows a "
            f"{prob_pct}% estimated probability of heart disease — below the clinical "
            f"alert threshold. Routine monitoring is recommended."
        )
    elif probability < 0.65:
        risk_level = "Moderate Risk"
        css_class  = "result-moderate"
        emoji      = "🟡"
        explanation = (
            f"This patient shows a {prob_pct}% estimated probability of heart disease. "
            f"Further clinical investigation is recommended, including additional "
            f"diagnostic workup and close follow-up."
        )
    else:
        risk_level = "High Risk"
        css_class  = "result-high"
        emoji      = "🔴"
        explanation = (
            f"This patient shows a {prob_pct}% estimated probability of heart disease — "
            f"above the high-risk threshold. Prompt clinical evaluation and cardiology "
            f"referral is strongly recommended."
        )

    # ── Display result ────────────────────────────────────────
    st.markdown(f"""
    <div class="{css_class}">
        <div class="result-title">{emoji} {risk_level}</div>
        <div class="result-sub">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability bar ───────────────────────────────────────
    st.markdown("**Estimated Disease Probability**")
    st.progress(probability)
    col1, col2, col3 = st.columns(3)
    col1.metric("Probability", f"{prob_pct}%")
    col2.metric("Risk Level", risk_level)
    col3.metric("Model Confidence", f"{'High' if abs(probability-0.5)>0.25 else 'Moderate'}")

    # ── Key factors ───────────────────────────────────────────
    st.markdown('<div class="section-title">Key Contributing Factors</div>',
                unsafe_allow_html=True)
    st.caption("Factors from this patient record most associated with heart disease risk:")

    # Extract feature importance from the random forest inside the pipeline
    rf_model = model.named_steps["classifier"]
    feature_names = (
        load_config()["features"]["numeric"] +
        load_config()["features"]["categorical"]
    )

    # Map to plain English labels
    plain_labels = {
        "thalach":  "Max Heart Rate",
        "oldpeak":  "ST Depression",
        "age":      "Age",
        "trestbps": "Blood Pressure",
        "chol":     "Cholesterol",
        "ca":       "Vessels Fluoroscopy",
        "cp":       "Chest Pain Type",
        "thal":     "Thalassemia",
        "exang":    "Exercise Angina",
        "slope":    "ST Slope",
        "sex":      "Biological Sex",
        "fbs":      "Fasting Blood Sugar",
        "restecg":  "Resting ECG",
    }

    importances = rf_model.feature_importances_
    # Aggregate OHE-expanded importances back to original features
    n_numeric = len(load_config()["features"]["numeric"])
    numeric_importances = importances[:n_numeric]

    imp_df = pd.DataFrame({
        "Feature":    [plain_labels.get(f, f) for f in feature_names[:n_numeric]],
        "Importance": numeric_importances
    }).sort_values("Importance", ascending=False).head(5)

    for _, row in imp_df.iterrows():
        bar_width = int(row["Importance"] / imp_df["Importance"].max() * 100)
        st.markdown(f"""
        <div style="margin: 6px 0;">
            <span style="font-size:0.9rem;color:#1E293B;font-weight:500;">{row['Feature']}</span>
            <div style="background:#E2E8F0;border-radius:4px;height:10px;margin-top:4px;">
                <div style="background:#0D7377;width:{bar_width}%;height:10px;border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Clinical Disclaimer:</strong> This tool is intended for screening support
        only and does not constitute a medical diagnosis. Results must be interpreted by a
        qualified healthcare professional in conjunction with full clinical assessment.
        Model trained on the UCI Heart Disease dataset (Cleveland, n=303).
    </div>
    """, unsafe_allow_html=True)