import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="What-If Simulator", layout="wide")

st.markdown("""
<style>
.sim-header {
    font-size: 1rem;
    color: #888;
    margin-bottom: 1.5rem;
    padding: 1rem;
    background: #1a1d24;
    border-radius: 10px;
    border-left: 4px solid #00d4aa;
}
</style>
""", unsafe_allow_html=True)

st.title("What-If Scenario Simulator")
st.markdown("""
<div class="sim-header">
Move any slider and watch the health risk gauge update instantly.
This simulates what would happen if pollution levels changed — 
useful for policy decisions like "what if factories reduced emissions by 30%?"
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return (joblib.load("models/svm_model.pkl"),
            joblib.load("models/random_forest.pkl"),
            joblib.load("models/classifier_scaler.pkl"),
            joblib.load("models/label_encoder.pkl"))

svm, rf, scaler, le = load_models()

color_hex  = {"Good":"#00d4aa","Moderate":"#ffaa00","Poor":"#ff6600","Hazardous":"#ff4444"}
risk_value = {"Good": 12, "Moderate": 37, "Poor": 62, "Hazardous": 87}
advice     = {
    "Good":      "Safe for all outdoor activities. Air is clean.",
    "Moderate":  "Sensitive groups should reduce prolonged outdoor exposure.",
    "Poor":      "Everyone may feel effects. Avoid outdoor exercise.",
    "Hazardous": "Health emergency. Stay indoors. Use N95 mask."
}

# ── Layout: sliders left, output right ────────────────────
left, right = st.columns([1, 2])

with left:
    st.markdown("#### Adjust Pollution Levels")
    st.markdown("*Sliders update the prediction instantly*")
    pm25  = st.slider("PM2.5  (μg/m³)",  0.0, 500.0, 60.0,  step=5.0)
    pm10  = st.slider("PM10   (μg/m³)",  0.0, 500.0, 100.0, step=5.0)
    no2   = st.slider("NO2    (μg/m³)",  0.0, 200.0, 30.0,  step=2.0)
    so2   = st.slider("SO2    (μg/m³)",  0.0, 200.0, 15.0,  step=2.0)
    co    = st.slider("CO     (mg/m³)",  0.0,  20.0,  1.5,  step=0.5)
    o3    = st.slider("O3     (μg/m³)",  0.0, 300.0, 40.0,  step=5.0)
    month = st.slider("Month",           1,    12,    6)

    # Scenario presets
    st.markdown("---")
    st.markdown("**Try a scenario:**")
    s1, s2 = st.columns(2)
    if s1.button("Factory shutdown\n(-40% NO2)"):
        st.session_state["no2_override"] = no2 * 0.6
    if s2.button("Diwali night\n(+300% PM2.5)"):
        st.session_state["pm25_override"] = min(pm25 * 4, 500)

# ── Prediction ─────────────────────────────────────────────
features   = np.array([[pm25, pm10, no2, so2, co, o3, month]])
features_s = scaler.transform(features)

pred      = le.inverse_transform(svm.predict(features_s))[0]
proba     = svm.predict_proba(features_s)[0]
rf_pred   = le.inverse_transform(rf.predict(features_s))[0]
imp       = joblib.load("models/random_forest.pkl").feature_importances_
feat_vals = [pm25/500, pm10/500, no2/200, so2/200, co/20, o3/300]
feat_names= ["PM2.5","PM10","NO2","SO2","CO","O3"]
impact    = [i*v for i,v in zip(imp[:6], feat_vals)]

with right:
    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_value[pred],
        number={"suffix": " / 100", "font": {"size": 28, "color": color_hex[pred]}},
        title={"text": f"<b style='color:{color_hex[pred]}'>{pred}</b><br><span style='font-size:14px;color:#888'>{advice[pred]}</span>",
               "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#888",
                     "tickvals": [12,37,62,87],
                     "ticktext": ["Good","Moderate","Poor","Hazardous"]},
            "bar":  {"color": color_hex[pred], "thickness": 0.3},
            "bgcolor": "#1a1d24",
            "bordercolor": "#2a2d35",
            "steps": [
                {"range": [0,   25], "color": "#0a2e1e"},
                {"range": [25,  50], "color": "#2e2200"},
                {"range": [50,  75], "color": "#2e1200"},
                {"range": [75, 100], "color": "#2e0000"}
            ],
            "threshold": {
                "line":  {"color": color_hex[pred], "width": 4},
                "thickness": 0.8,
                "value": risk_value[pred]
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="#0f1116", font_color="white",
        height=380, margin=dict(l=20,r=20,t=20,b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model agreement
    if pred == rf_pred:
        st.success(f"Both SVM and Random Forest agree: **{pred}**")
    else:
        st.warning(f"SVM → **{pred}**  |  Random Forest → **{rf_pred}**  (borderline case)")

    # Probability bars
    st.markdown("#### Prediction Confidence")
    fig2 = go.Figure()
    for cls, prob in zip(le.classes_, proba):
        fig2.add_trace(go.Bar(
            x=[prob * 100], y=[cls],
            orientation="h",
            marker_color=color_hex.get(cls, "gray"),
            text=f"{prob*100:.1f}%", textposition="outside",
            name=cls
        ))
    fig2.update_layout(
        paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
        font_color="white", height=200, showlegend=False,
        xaxis=dict(range=[0, 110], gridcolor="#2a2d35"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=40, t=10, b=0),
        barmode="overlay"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Which pollutant is causing the most risk
    st.markdown("#### Which Pollutant Is Driving The Risk?")
    sorted_impact = sorted(zip(feat_names, impact), key=lambda x: x[1], reverse=True)
    names_sorted  = [x[0] for x in sorted_impact]
    vals_sorted   = [x[1] for x in sorted_impact]
    max_val       = max(vals_sorted) if max(vals_sorted) > 0 else 1

    fig3 = go.Figure(go.Bar(
        x=vals_sorted,
        y=names_sorted,
        orientation="h",
        marker_color=[color_hex[pred]] * len(names_sorted),
        marker_opacity=[0.4 + 0.6 * (v / max_val) for v in vals_sorted],
        text=[f"{v/max_val*100:.0f}%" for v in vals_sorted],
        textposition="outside"
    ))
    fig3.update_layout(
        paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
        font_color="white", height=220,
        xaxis=dict(gridcolor="#2a2d35", showticklabels=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=60, t=10, b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)