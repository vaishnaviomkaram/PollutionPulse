import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Predict Risk", layout="wide")

st.markdown("""
<style>
.result-box {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}
.result-good      { background:#0a2e1e; border: 2px solid #00d4aa; }
.result-moderate  { background:#2e2200; border: 2px solid #ffaa00; }
.result-poor      { background:#2e1200; border: 2px solid #ff6600; }
.result-hazardous { background:#2e0000; border: 2px solid #ff4444; }
.result-label { font-size: 2.5rem; font-weight: 800; }
.result-advice { font-size: 1rem; color: #aaa; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("Health Risk Predictor")
st.markdown("Enter today's pollution readings — get instant health risk prediction from 2 ML models.")

@st.cache_resource
def load_models():
    return (joblib.load("models/svm_model.pkl"),
            joblib.load("models/random_forest.pkl"),
            joblib.load("models/classifier_scaler.pkl"),
            joblib.load("models/label_encoder.pkl"))

svm, rf, scaler, le = load_models()

# ── Presets ────────────────────────────────────────────────
st.markdown("#### Quick Presets")
p1, p2, p3, p4 = st.columns(4)
preset = None
if p1.button("Delhi Winter"):  preset = [180, 250, 80, 30, 3.5, 20, 1]
if p2.button("Mumbai Monsoon"): preset = [35,  60,  25, 10, 1.0, 50, 7]
if p3.button("Clean Day"):      preset = [20,  40,  15,  8, 0.8, 45, 4]
if p4.button("Hazardous Day"):  preset = [300, 400, 120, 60, 8.0, 15, 11]

defaults = preset if preset else [60, 100, 30, 15, 1.5, 40, 6]

st.markdown("---")
st.markdown("#### Enter Values Manually")

col1, col2, col3, col4 = st.columns(4)
with col1:
    pm25  = st.number_input("PM2.5 (μg/m³)", 0.0, 1000.0, float(defaults[0]))
    pm10  = st.number_input("PM10 (μg/m³)",  0.0, 1000.0, float(defaults[1]))
with col2:
    no2   = st.number_input("NO2 (μg/m³)",   0.0, 500.0,  float(defaults[2]))
    so2   = st.number_input("SO2 (μg/m³)",   0.0, 500.0,  float(defaults[3]))
with col3:
    co    = st.number_input("CO (mg/m³)",    0.0,  50.0,  float(defaults[4]))
    o3    = st.number_input("O3 (μg/m³)",    0.0, 500.0,  float(defaults[5]))
with col4:
    month = st.slider("Month", 1, 12, int(defaults[6]))
    st.markdown(" ")
    predict_btn = st.button("Predict Now", type="primary", use_container_width=True)

if predict_btn or preset:
    features   = np.array([[pm25, pm10, no2, so2, co, o3, month]])
    features_s = scaler.transform(features)

    svm_pred  = le.inverse_transform(svm.predict(features_s))[0]
    rf_pred   = le.inverse_transform(rf.predict(features_s))[0]
    svm_proba = svm.predict_proba(features_s)[0]
    rf_proba  = rf.predict_proba(features_s)[0]

    color_hex = {"Good":"#00d4aa","Moderate":"#ffaa00","Poor":"#ff6600","Hazardous":"#ff4444"}
    advice = {
        "Good":      "Air is clean. Safe for jogging, outdoor sports, and all activities.",
        "Moderate":  "Acceptable quality. Children and elderly should reduce long outdoor exposure.",
        "Poor":      "Everyone may feel health effects. Avoid outdoor exercise. Wear a mask.",
        "Hazardous": "Health emergency. Stay indoors. Run air purifiers. Wear N95 if going out."
    }
    css_class = {"Good":"good","Moderate":"moderate","Poor":"poor","Hazardous":"hazardous"}

    st.markdown("---")
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("#### SVM Prediction")
        st.markdown(f"""
        <div class="result-box result-{css_class[svm_pred]}">
            <div class="result-label" style="color:{color_hex[svm_pred]}">{svm_pred}</div>
            <div class="result-advice">{advice[svm_pred]}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(" ")
        fig = go.Figure(go.Bar(
            x=list(le.classes_), y=svm_proba * 100,
            marker_color=[color_hex.get(c,"gray") for c in le.classes_],
            text=[f"{p*100:.1f}%" for p in svm_proba], textposition="outside"
        ))
        fig.update_layout(title="SVM Confidence", paper_bgcolor="#0f1116",
                          plot_bgcolor="#0f1116", font_color="white",
                          height=250, yaxis=dict(range=[0,110], gridcolor="#2a2d35"),
                          margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        st.markdown("#### Random Forest Prediction")
        st.markdown(f"""
        <div class="result-box result-{css_class[rf_pred]}">
            <div class="result-label" style="color:{color_hex[rf_pred]}">{rf_pred}</div>
            <div class="result-advice">{advice[rf_pred]}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(" ")
        fig2 = go.Figure(go.Bar(
            x=list(le.classes_), y=rf_proba * 100,
            marker_color=[color_hex.get(c,"gray") for c in le.classes_],
            text=[f"{p*100:.1f}%" for p in rf_proba], textposition="outside"
        ))
        fig2.update_layout(title="Random Forest Confidence", paper_bgcolor="#0f1116",
                           plot_bgcolor="#0f1116", font_color="white",
                           height=250, yaxis=dict(range=[0,110], gridcolor="#2a2d35"),
                           margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    if svm_pred == rf_pred:
        st.success(f"Both models agree — the air quality is **{svm_pred}**")
    else:
        st.warning(f"Models disagree — SVM says **{svm_pred}**, Random Forest says **{rf_pred}**. This happens in borderline cases.")