import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard", layout="wide")

st.markdown("""
<style>
.metric-card {
    background: #1a1d24;
    border: 1px solid #2a2d35;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 700; color: #00d4aa; }
.metric-lbl { font-size: 0.8rem; color: #888; }
</style>
""", unsafe_allow_html=True)

st.title("City Dashboard")
st.markdown("Explore real pollution trends across Indian cities.")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/cleaned_data.csv")

df = load_data()

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.header("Filters")
selected_city = st.sidebar.selectbox("Select City", sorted(df["City"].unique()), index=list(sorted(df["City"].unique())).index("Delhi"))
selected_years = st.sidebar.multiselect("Select Years", sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))

df_city = df[(df["City"] == selected_city) & (df["Year"].isin(selected_years))]

# ── Metrics row ────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg PM2.5",       f"{df_city['PM2.5'].mean():.1f} μg/m³")
c2.metric("Avg PM10",        f"{df_city['PM10'].mean():.1f} μg/m³")
c3.metric("Avg NO2",         f"{df_city['NO2'].mean():.1f} μg/m³")
c4.metric("Hazardous Days",  int((df_city["Health_Risk"] == "Hazardous").sum()))
c5.metric("Safe Days",       int((df_city["Health_Risk"] == "Good").sum()))

st.markdown("---")

# ── PM2.5 trend ────────────────────────────────────────────
st.subheader(f"PM2.5 Trend — {selected_city}")
monthly = df_city.groupby(["Year","Month"])["PM2.5"].mean().reset_index()
monthly["Period"] = monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=monthly["Period"], y=monthly["PM2.5"],
    mode="lines+markers",
    line=dict(color="#00d4aa", width=2.5),
    marker=dict(size=5),
    fill="tozeroy",
    fillcolor="rgba(0,212,170,0.1)",
    name="PM2.5"
))
fig.add_hline(y=60, line_dash="dash", line_color="red",
              annotation_text="CPCB Safe Limit (60)", annotation_position="top left")
fig.update_layout(
    paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
    font_color="white", height=350,
    xaxis=dict(showgrid=False, tickangle=45),
    yaxis=dict(gridcolor="#2a2d35"),
    margin=dict(l=0, r=0, t=10, b=0)
)
st.plotly_chart(fig, use_container_width=True)

# ── Two charts side by side ────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Health Risk Breakdown")
    dist = df_city["Health_Risk"].value_counts().reset_index()
    dist.columns = ["Risk", "Count"]
    color_map = {"Good":"#00d4aa","Moderate":"#ffaa00","Poor":"#ff6600","Hazardous":"#ff4444"}
    fig2 = px.pie(dist, names="Risk", values="Count",
                  color="Risk", color_discrete_map=color_map,
                  hole=0.5)
    fig2.update_layout(paper_bgcolor="#0f1116", font_color="white",
                       legend=dict(orientation="h"), height=300,
                       margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.subheader("PM2.5 by Season")
    season_order = ["Winter","Spring","Summer","Monsoon","Post_Monsoon"]
    season_avg = df_city.groupby("Season")["PM2.5"].mean().reset_index()
    fig3 = px.bar(season_avg, x="Season", y="PM2.5",
                  color="PM2.5", color_continuous_scale="RdYlGn_r",
                  text=season_avg["PM2.5"].round(1))
    fig3.update_traces(textposition="outside")
    fig3.add_hline(y=60, line_dash="dash", line_color="red")
    fig3.update_layout(paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
                       font_color="white", height=300, showlegend=False,
                       yaxis=dict(gridcolor="#2a2d35"),
                       margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig3, use_container_width=True)

# ── All cities comparison ──────────────────────────────────
st.markdown("---")
st.subheader("All Cities — PM2.5 Comparison")
all_city_avg = df.groupby("City")["PM2.5"].mean().sort_values(ascending=True).reset_index()
fig4 = px.bar(all_city_avg, x="PM2.5", y="City",
              orientation="h",
              color="PM2.5", color_continuous_scale="RdYlGn_r",
              text=all_city_avg["PM2.5"].round(1))
fig4.add_vline(x=60, line_dash="dash", line_color="red",
               annotation_text="Safe Limit")
fig4.update_traces(textposition="outside")
fig4.update_layout(paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
                   font_color="white", height=600,
                   xaxis=dict(gridcolor="#2a2d35"),
                   yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                   margin=dict(l=0,r=0,t=10,b=0))
st.plotly_chart(fig4, use_container_width=True)