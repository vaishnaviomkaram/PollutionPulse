import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Alert System", layout="wide")

st.markdown("""
<style>
.main { background-color: #0f1116; }
.block-container { padding: 2rem 3rem; }
</style>
""", unsafe_allow_html=True)

st.title("Pollution Alert System")
st.markdown("Simulates what an automated government alert system would look like — trigger alerts when pollution crosses dangerous thresholds.")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_data.csv")

df = load_data()

# ── Sidebar settings ───────────────────────────────────────
st.sidebar.header("Alert Settings")
pm25_threshold = st.sidebar.slider(
    "PM2.5 Alert Threshold (μg/m³)", 30, 300, 120,
    help="Alert triggers when PM2.5 crosses this value"
)
selected_cities = st.sidebar.multiselect(
    "Cities to Monitor",
    sorted(df["City"].unique()),
    default=["Delhi", "Amritsar", "Lucknow"]
)

st.markdown("---")

# ── Summary metrics ────────────────────────────────────────
alerts = df[
    (df["City"].isin(selected_cities)) &
    (df["PM2.5"] >= pm25_threshold)
][["City","Date","PM2.5","Health_Risk","Season","Year"]].sort_values(
    "PM2.5", ascending=False
).reset_index(drop=True)

total    = len(alerts)
haz      = len(alerts[alerts["Health_Risk"] == "Hazardous"])
worst    = alerts.iloc[0] if total > 0 else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Alert Days",      total)
c2.metric("Hazardous Days",         haz)
c3.metric("Cities Monitored",       len(selected_cities))
c4.metric("Threshold Set",          f"PM2.5 > {pm25_threshold}")

st.markdown("---")

if total == 0:
    st.success(f"No days exceeded PM2.5 > {pm25_threshold} for selected cities.")
else:
    # ── Alert feed ─────────────────────────────────────────
    col_feed, col_chart = st.columns([1, 1])

    with col_feed:
        st.subheader("Alert Feed (Latest 20)")
        for _, row in alerts.head(20).iterrows():
            color = "#ff4444" if row["Health_Risk"] == "Hazardous" else "#ff6600"
            icon  = "🚨" if row["Health_Risk"] == "Hazardous" else "⚠️"
            st.markdown(f"""
            <div style="background:#1a1d24; border-left:4px solid {color};
                        border-radius:8px; padding:0.7rem 1rem;
                        margin-bottom:0.4rem;">
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#fff; font-weight:600;">
                        {icon} {row['City']}
                    </span>
                    <span style="color:#555; font-size:0.8rem;">{row['Date']}</span>
                </div>
                <div style="margin-top:0.3rem;">
                    <span style="color:{color}; font-weight:700;">
                        PM2.5: {row['PM2.5']:.1f} μg/m³
                    </span>
                    <span style="color:#555; font-size:0.8rem; margin-left:10px;">
                        {row['Health_Risk']} · {row['Season']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_chart:
        st.subheader("Alert Days by City")
        alert_count = alerts.groupby("City").size().sort_values(
            ascending=True).reset_index()
        alert_count.columns = ["City", "Alert Days"]
        fig = px.bar(
            alert_count, x="Alert Days", y="City",
            orientation="h",
            color="Alert Days",
            color_continuous_scale="Reds",
            text="Alert Days"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
            font_color="white", height=400, showlegend=False,
            xaxis=dict(gridcolor="#2a2d35"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=40,t=10,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── By season ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("Which Season Has Most Alerts?")
    season_alerts = alerts.groupby("Season").size().reset_index()
    season_alerts.columns = ["Season","Alert Days"]
    season_alerts = season_alerts.sort_values("Alert Days", ascending=False)

    fig2 = px.bar(
        season_alerts, x="Season", y="Alert Days",
        color="Alert Days", color_continuous_scale="Reds",
        text="Alert Days"
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
        font_color="white", height=300, showlegend=False,
        yaxis=dict(gridcolor="#2a2d35"),
        margin=dict(l=0,r=0,t=10,b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.info("Winter produces the most alerts because cold air traps pollution near ground level — the same amount of smoke stays closer to where people breathe.")