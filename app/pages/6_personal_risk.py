import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Personal Risk", layout="wide")

st.markdown("""
<style>
.main { background-color: #0f1116; }
.block-container { padding: 2rem 3rem; }
.result-box {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Personal Health Risk Advisor")
st.markdown("General AQI does not tell the full story. A 60-year-old asthma patient faces much higher risk than a healthy 25-year-old at the same pollution level. Enter your profile to get a personalised assessment.")

@st.cache_resource
def load_models():
    return (joblib.load("models/svm_model.pkl"),
            joblib.load("models/classifier_scaler.pkl"),
            joblib.load("models/label_encoder.pkl"))

svm, scaler, le = load_models()

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Profile")
    age           = st.slider("Your Age", 1, 90, 25)
    has_asthma    = st.checkbox("I have asthma or breathing problems")
    is_pregnant   = st.checkbox("Pregnant")
    outdoor_hours = st.slider("Hours spent outdoors per day", 0, 16, 4)
    exercises_outdoors = st.checkbox("I exercise outdoors regularly")

with col2:
    st.subheader("Today's Pollution Readings")
    pm25  = st.slider("PM2.5 (μg/m³)",  0.0, 500.0, 60.0,  step=5.0)
    pm10  = st.slider("PM10 (μg/m³)",   0.0, 500.0, 100.0, step=5.0)
    no2   = st.slider("NO2 (μg/m³)",    0.0, 200.0, 30.0,  step=2.0)
    so2   = st.slider("SO2 (μg/m³)",    0.0, 200.0, 15.0,  step=2.0)
    co    = st.slider("CO (mg/m³)",     0.0, 20.0,  1.5,   step=0.5)
    o3    = st.slider("O3 (μg/m³)",     0.0, 300.0, 40.0,  step=5.0)
    month = st.slider("Month",          1,   12,    6)

# ── Base ML prediction ─────────────────────────────────────
features   = np.array([[pm25, pm10, no2, so2, co, o3, month]])
features_s = scaler.transform(features)
base_pred  = le.inverse_transform(svm.predict(features_s))[0]

# ── Personal risk adjustment ───────────────────────────────
risk_levels = ["Good", "Moderate", "Poor", "Hazardous"]
risk_idx    = risk_levels.index(base_pred)

extra_risk = 0
reasons    = []

if has_asthma:
    extra_risk += 1
    reasons.append("Asthma increases sensitivity to PM2.5 and O3 by 2-3x")
if age < 12:
    extra_risk += 1
    reasons.append("Children breathe 50% more air per kg of body weight than adults")
if age > 60:
    extra_risk += 1
    reasons.append("Elderly have reduced lung capacity and weaker immune response")
if is_pregnant:
    extra_risk += 1
    reasons.append("PM2.5 exposure during pregnancy is linked to low birth weight")
if outdoor_hours > 6:
    extra_risk += 1
    reasons.append(f"{outdoor_hours} hours outdoors = significantly higher total exposure")
if exercises_outdoors:
    extra_risk += 1
    reasons.append("Exercise increases breathing rate — you inhale 10x more air than at rest")

personal_idx  = min(risk_idx + extra_risk, 3)
personal_pred = risk_levels[personal_idx]

color_map = {
    "Good":      "#00d4aa",
    "Moderate":  "#ffaa00",
    "Poor":      "#ff6600",
    "Hazardous": "#ff4444"
}

advice_map = {
    "Good":      "Air is safe for you today. Normal outdoor activity is fine.",
    "Moderate":  "Take it easy outdoors. Avoid heavy exercise outside.",
    "Poor":      "Limit time outdoors. Wear a mask if going out. Keep windows closed.",
    "Hazardous": "Stay indoors. Close all windows. Use air purifier if available. Wear N95 if going out."
}

st.markdown("---")
st.subheader("Your Personal Risk Assessment")

cola, colb = st.columns(2)

with cola:
    st.markdown(f"""
    <div style="background:#1a1d24; border:2px solid {color_map[personal_pred]};
                border-radius:16px; padding:2rem; text-align:center;">
        <div style="font-size:0.8rem;color:#666;
                    text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;">
            Your Personal Risk
        </div>
        <div style="font-size:3.5rem;font-weight:900;
                    color:{color_map[personal_pred]};line-height:1.1;">
            {personal_pred}
        </div>
        <div style="font-size:0.9rem;color:#888;margin-top:0.8rem;line-height:1.5;">
            {advice_map[personal_pred]}
        </div>
    </div>
    """, unsafe_allow_html=True)

with colb:
    st.markdown(f"""
    <div style="background:#1a1d24; border:1px solid #2a2d35;
                border-radius:16px; padding:2rem;">
        <div style="font-size:0.8rem;color:#666;
                    text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;">
            General Population Risk
        </div>
        <div style="font-size:1.8rem;font-weight:700;color:{color_map[base_pred]};">
            {base_pred}
        </div>
        <div style="font-size:0.85rem;color:#666;margin-top:0.5rem;margin-bottom:1rem;">
            {"Your personal risk is HIGHER than general population because:" if extra_risk > 0 else "Your risk matches the general population."}
        </div>
        {"".join([f'<div style="font-size:0.8rem;color:#888;padding:5px 0;border-bottom:1px solid #2a2d35;">⚠️ {r}</div>' for r in reasons])}
    </div>
    """, unsafe_allow_html=True)

# ── Safety tips based on personal profile ─────────────────
if extra_risk > 0:
    st.markdown("---")
    st.subheader("Personalised Safety Tips")

    tips = []
    if has_asthma:
        tips.append("Keep your inhaler with you at all times on poor air quality days")
    if age < 12:
        tips.append("Avoid outdoor PE classes and sports when PM2.5 is above 60")
    if age > 60:
        tips.append("Morning walks should be skipped on winter days — pollution peaks at dawn")
    if is_pregnant:
        tips.append("Install an air purifier in the room where you sleep")
    if exercises_outdoors:
        tips.append("Switch outdoor workouts to indoor gym on days when AQI is Poor or above")
    if outdoor_hours > 6:
        tips.append("Wear an N95 mask — not just a cloth mask — it filters 95% of PM2.5 particles")

    for tip in tips:
        st.markdown(f"""
        <div style="background:#1a1d24; border-left:3px solid #00d4aa;
                    border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.5rem;">
            <span style="color:#888; font-size:0.85rem;">💡 {tip}</span>
        </div>
        """, unsafe_allow_html=True)