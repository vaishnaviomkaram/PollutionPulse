import streamlit as st
import pandas as pd

st.set_page_config(page_title="PollutionPulse", page_icon="🌿", layout="wide")

st.markdown("""
<style>
    /* Base */
    .main { background-color: #0f1116; }
    .block-container { padding: 2rem 3rem; max-width: 1200px; margin: auto; }

    /* Hero */
    .hero-wrapper {
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
    }
    .hero-badge {
        display: inline-block;
        background: #1a1d24;
        border: 1px solid #00d4aa;
        border-radius: 20px;
        padding: 4px 16px;
        font-size: 0.75rem;
        color: #00d4aa;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4aa 0%, #0088ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-bottom: 1rem;
    }
    .hero-desc {
        font-size: 1.1rem;
        color: #888;
        max-width: 600px;
        margin: 0 auto 2rem auto;
        line-height: 1.7;
    }

    /* Stat cards */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 2rem 0;
    }
    .stat-card {
        background: #1a1d24;
        border: 1px solid #2a2d35;
        border-radius: 14px;
        padding: 1.4rem 1rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .stat-card:hover { border-color: #00d4aa; }
    .stat-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #00d4aa;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #666;
        margin-top: 0.3rem;
        line-height: 1.4;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section heading */
    .section-heading {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 2.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2a2d35;
    }

    /* Risk cards */
    .risk-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 2rem;
    }
    .risk-card {
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .risk-good      { background: #0a2e1e; border: 1px solid #00d4aa33; }
    .risk-moderate  { background: #2a1f00; border: 1px solid #ffaa0033; }
    .risk-poor      { background: #2a1200; border: 1px solid #ff660033; }
    .risk-hazardous { background: #2a0000; border: 1px solid #ff444433; }
    .risk-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .risk-pm { font-size: 0.78rem; color: #888; margin-bottom: 0.4rem; }
    .risk-desc { font-size: 0.78rem; color: #666; line-height: 1.5; }

    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #1a1d24;
        border: 1px solid #2a2d35;
        border-radius: 14px;
        padding: 1.5rem;
        transition: border-color 0.2s;
    }
    .feature-card:hover { border-color: #0088ff55; }
    .feature-icon { font-size: 1.8rem; margin-bottom: 0.8rem; }
    .feature-title {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.4rem;
    }
    .feature-desc { font-size: 0.82rem; color: #666; line-height: 1.6; }

    /* ML models strip */
    .models-strip {
        background: #1a1d24;
        border: 1px solid #2a2d35;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
    }
    .models-title {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }
    .models-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .model-pill {
        background: #0f1116;
        border: 1px solid #2a2d35;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.8rem;
        color: #aaa;
    }
    .model-pill span { color: #00d4aa; font-weight: 600; margin-right: 5px; }

    /* Footer */
    .footer {
        text-align: center;
        color: #444;
        font-size: 0.78rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #1a1d24;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load data ──────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_data.csv")

df = load_data()

total_records  = len(df)
hazardous_days = int((df["Health_Risk"] == "Hazardous").sum())
cities_count   = df["City"].nunique()
worst_city     = df.groupby("City")["PM2.5"].mean().idxmax()
worst_pm25     = df.groupby("City")["PM2.5"].mean().max()
safest_city    = df.groupby("City")["PM2.5"].mean().idxmin()
safest_pm25    = df.groupby("City")["PM2.5"].mean().min()
years_covered  = f"{int(df['Year'].min())}–{int(df['Year'].max())}"


# ── Hero ───────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrapper">
    <div class="hero-badge">🌿 AI-Powered Air Quality Intelligence</div>
    <div class="hero-title">PollutionPulse</div>
    <div class="hero-desc">
        Real CPCB sensor data from {cities_count} Indian cities analysed using
        machine learning — predict health risks, explore pollution patterns,
        and simulate what-if policy scenarios.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Live stats ─────────────────────────────────────────────
st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-icon">📊</div>
        <div class="stat-number">{total_records:,}</div>
        <div class="stat-label">Records<br>Analysed</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🏙️</div>
        <div class="stat-number">{cities_count}</div>
        <div class="stat-label">Indian Cities<br>Covered</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">⚠️</div>
        <div class="stat-number">{hazardous_days:,}</div>
        <div class="stat-label">Hazardous Air<br>Days Detected</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">📅</div>
        <div class="stat-number">{years_covered}</div>
        <div class="stat-label">Years of<br>Real Data</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Worst and safest city highlight ───────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="stat-card" style="text-align:left; padding: 1.2rem 1.5rem;">
        <div style="font-size:0.75rem; color:#666; text-transform:uppercase;
                    letter-spacing:1px; margin-bottom:0.5rem;">
            Most Polluted City
        </div>
        <div style="font-size:1.6rem; font-weight:800; color:#ff4444;">
            {worst_city}
        </div>
        <div style="font-size:0.85rem; color:#888; margin-top:0.3rem;">
            Average PM2.5: <b style="color:#ff6644">{worst_pm25:.1f} μg/m³</b>
            &nbsp;·&nbsp; {int(worst_pm25 / 60 * 100 - 100)}% above CPCB safe limit
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card" style="text-align:left; padding: 1.2rem 1.5rem;">
        <div style="font-size:0.75rem; color:#666; text-transform:uppercase;
                    letter-spacing:1px; margin-bottom:0.5rem;">
            Cleanest City
        </div>
        <div style="font-size:1.6rem; font-weight:800; color:#00d4aa;">
            {safest_city}
        </div>
        <div style="font-size:0.85rem; color:#888; margin-top:0.3rem;">
            Average PM2.5: <b style="color:#00d4aa">{safest_pm25:.1f} μg/m³</b>
            &nbsp;·&nbsp; Within safe limits
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Health risk scale ──────────────────────────────────────
st.markdown('<div class="section-heading">Health Risk Scale Used</div>', unsafe_allow_html=True)
st.markdown("""
<div class="risk-grid">
    <div class="risk-card risk-good">
        <div class="risk-title" style="color:#00d4aa">GOOD</div>
        <div class="risk-pm">PM2.5 below 30 μg/m³</div>
        <div class="risk-desc">Safe for everyone including children and elderly</div>
    </div>
    <div class="risk-card risk-moderate">
        <div class="risk-title" style="color:#ffaa00">MODERATE</div>
        <div class="risk-pm">PM2.5 30–90 μg/m³</div>
        <div class="risk-desc">Sensitive groups should reduce outdoor exposure</div>
    </div>
    <div class="risk-card risk-poor">
        <div class="risk-title" style="color:#ff6600">POOR</div>
        <div class="risk-pm">PM2.5 90–120 μg/m³</div>
        <div class="risk-desc">General population may experience health effects</div>
    </div>
    <div class="risk-card risk-hazardous">
        <div class="risk-title" style="color:#ff4444">HAZARDOUS</div>
        <div class="risk-pm">PM2.5 above 120 μg/m³</div>
        <div class="risk-desc">Health emergency — stay indoors, use N95 mask</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── What each page does ────────────────────────────────────
st.markdown('<div class="section-heading">What You Can Do Here</div>', unsafe_allow_html=True)
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">📈</div>
        <div class="feature-title">City Dashboard</div>
        <div class="feature-desc">
            Explore PM2.5 trends, seasonal patterns, and health risk
            breakdowns for any of the 26 cities across different years.
        </div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <div class="feature-title">Health Risk Predictor</div>
        <div class="feature-desc">
            Enter pollutant readings and get instant predictions from
            two ML models — SVM and Random Forest — with confidence scores.
        </div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🎛️</div>
        <div class="feature-title">What-If Simulator</div>
        <div class="feature-desc">
            Drag sliders to change pollution levels and watch the health
            risk gauge update live. Simulate policy changes like factory
            shutdowns or Diwali night scenarios.
        </div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🗺️</div>
        <div class="feature-title">Hotspot Map</div>
        <div class="feature-desc">
            Geographic bubble map of India showing which cities
            have the worst air quality and which clusters they belong to.
        </div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <div class="feature-title">Model Insights</div>
        <div class="feature-desc">
            See the ML engine — confusion matrix, ROC curves,
            feature importance, and PCA city clusters used to build
            the predictions.
        </div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Real CPCB Data</div>
        <div class="feature-desc">
            All data comes from India's Central Pollution Control Board
            monitoring stations — not synthetic or randomly generated.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── ML models used ─────────────────────────────────────────
st.markdown('<div class="section-heading">ML Models Inside This Project</div>', unsafe_allow_html=True)
st.markdown("""
<div class="models-strip">
    <div class="models-title">Trained Models</div>
    <div class="models-row">
        <div class="model-pill"><span>Unit IV</span> SVM with RBF Kernel</div>
        <div class="model-pill"><span>Unit IV</span> Random Forest + SMOTE</div>
        <div class="model-pill"><span>Unit IV</span> Naive Bayes Baseline</div>
        <div class="model-pill"><span>Unit III</span> Polynomial Regression</div>
        <div class="model-pill"><span>Unit V</span> K-Means Clustering</div>
        <div class="model-pill"><span>Unit V</span> DBSCAN Hotspot Detection</div>
        <div class="model-pill"><span>Unit V</span> PCA Dimensionality Reduction</div>
        <div class="model-pill"><span>Unit VI</span> ROC / AUC Evaluation</div>
        <div class="model-pill"><span>Unit VI</span> Concept Drift Analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    PollutionPulse &nbsp;·&nbsp; Built using Python, Scikit-learn, Streamlit
    &nbsp;·&nbsp; Data source: CPCB India via Kaggle
    &nbsp;·&nbsp; AI/ML Project — 2nd Year CSE
</div>
""", unsafe_allow_html=True)
