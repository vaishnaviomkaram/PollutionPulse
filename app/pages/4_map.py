import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Hotspot Map", layout="wide")
st.title("Pollution Hotspot Map")
st.markdown("Geographic view of pollution levels across Indian cities.")

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/cleaned_data.csv")
    clusters = pd.read_csv("data/processed/city_clusters.csv")
    return df, clusters

df, clusters = load_data()

city_coords = {
    "Ahmedabad":(23.02,72.57),"Aizawl":(23.72,92.71),"Amaravati":(16.50,80.64),
    "Amritsar":(31.63,74.87),"Bengaluru":(12.97,77.59),"Bhopal":(23.25,77.41),
    "Brajrajnagar":(21.82,83.92),"Chandigarh":(30.73,76.77),"Chennai":(13.08,80.27),
    "Coimbatore":(11.01,76.95),"Delhi":(28.61,77.20),"Ernakulam":(9.93,76.26),
    "Gurugram":(28.45,77.02),"Guwahati":(26.14,91.73),"Hyderabad":(17.38,78.48),
    "Jaipur":(26.91,75.78),"Jorapokhar":(23.70,86.42),"Kochi":(9.93,76.26),
    "Kolkata":(22.57,88.36),"Lucknow":(26.84,80.94),"Mumbai":(19.07,72.87),
    "Patna":(25.59,85.13),"Shillong":(25.57,91.89),"Talcher":(20.95,85.23),
    "Thiruvananthapuram":(8.52,76.93),"Visakhapatnam":(17.68,83.21)
}

city_avg = df.groupby("City")[["PM2.5","PM10","NO2"]].mean().reset_index()
city_avg["lat"] = city_avg["City"].map(lambda c: city_coords.get(c,(20,80))[0])
city_avg["lon"] = city_avg["City"].map(lambda c: city_coords.get(c,(20,80))[1])
city_avg = city_avg.merge(clusters[["City","Cluster"]], on="City", how="left")

label_map = {0:"Low Pollution", 1:"Medium Pollution", 2:"High Pollution"}
city_avg["Pollution Level"] = city_avg["Cluster"].map(label_map)

# ── Pollutant selector ─────────────────────────────────────
pollutant = st.selectbox("Select Pollutant to Map", ["PM2.5","PM10","NO2"])

fig = px.scatter_geo(
    city_avg, lat="lat", lon="lon",
    size=pollutant, color=pollutant,
    hover_name="City",
    hover_data={
        "PM2.5":":.1f","PM10":":.1f","NO2":":.1f",
        "Pollution Level":True,"lat":False,"lon":False
    },
    color_continuous_scale="RdYlGn_r",
    size_max=50, scope="asia",
    title=f"Average {pollutant} — Indian Cities",
    center={"lat":22,"lon":80}
)
fig.update_geos(
    bgcolor="#0f1116",
    lataxis_range=[6,38], lonaxis_range=[65,100],
    showcountries=True, countrycolor="#2a2d35",
    showsubunits=True, subunitcolor="#2a2d35",
    showland=True, landcolor="#1a1d24",
    showocean=True, oceancolor="#0f1116"
)
fig.update_layout(
    paper_bgcolor="#0f1116",
    font_color="white",
    height=600,
    margin=dict(l=0,r=0,t=40,b=0),
    coloraxis_colorbar=dict(
        title=dict(text=pollutant, font=dict(color="white")),
        tickfont=dict(color="white")
    )
)
st.plotly_chart(fig, use_container_width=True)

# ── Ranking table ──────────────────────────────────────────
st.markdown("---")
col_t, col_b = st.columns(2)

with col_t:
    st.subheader("Most Polluted Cities")
    top5 = city_avg.nlargest(5, "PM2.5")[["City","PM2.5","Pollution Level"]]
    for _, row in top5.iterrows():
        st.markdown(f"**{row['City']}** — PM2.5: `{row['PM2.5']:.1f}` μg/m³  🔴")

with col_b:
    st.subheader("Cleanest Cities")
    bot5 = city_avg.nsmallest(5, "PM2.5")[["City","PM2.5","Pollution Level"]]
    for _, row in bot5.iterrows():
        st.markdown(f"**{row['City']}** — PM2.5: `{row['PM2.5']:.1f}` μg/m³  🟢")