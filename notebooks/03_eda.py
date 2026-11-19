import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("reports/figures", exist_ok=True)

df = pd.read_csv("data/processed/cleaned_data.csv")
print("Loaded cleaned data:", df.shape)

# Chart 1: PM2.5 by City
plt.figure(figsize=(14, 6))
city_order = df.groupby("City")["PM2.5"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="City", y="PM2.5", order=city_order, palette="RdYlGn_r")
plt.xticks(rotation=45, ha="right")
plt.title("PM2.5 Distribution by City")
plt.axhline(60, color="red", linestyle="--", label="CPCB Safe Limit")
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/01_pm25_by_city.png", dpi=150)
plt.close()
print("Saved chart 1")

# Chart 2: Class Distribution
plt.figure(figsize=(8, 5))
colors = {"Good": "#1D9E75", "Moderate": "#EF9F27", "Poor": "#D85A30", "Hazardous": "#A32D2D"}
dist = df["Health_Risk"].value_counts()
bars = plt.bar(dist.index, dist.values,
               color=[colors[c] for c in dist.index], edgecolor="white", width=0.5)
for bar, val in zip(bars, dist.values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 50, str(val), ha="center", fontsize=10)
plt.title("Health Risk Class Distribution")
plt.ylabel("Number of Days")
plt.tight_layout()
plt.savefig("reports/figures/02_class_distribution.png", dpi=150)
plt.close()
print("Saved chart 2")

# Chart 3: Monthly PM2.5 Trend
plt.figure(figsize=(10, 5))
monthly = df.groupby("Month")["PM2.5"].mean()
plt.plot(monthly.index, monthly.values, marker="o", color="#185FA5", linewidth=2.5, markersize=8)
plt.xticks(range(1, 13), ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
plt.title("Monthly Average PM2.5 - Seasonal Pattern")
plt.ylabel("Avg PM2.5 (ug/m3)")
plt.axhline(60, color="red", linestyle="--", alpha=0.5, label="CPCB limit")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/03_monthly_trend.png", dpi=150)
plt.close()
print("Saved chart 3")

# Chart 4: Correlation Heatmap
plt.figure(figsize=(9, 7))
pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
corr = df[pollutants].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0, square=True, linewidths=0.5)
plt.title("Pollutant Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/figures/04_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved chart 4")

# Chart 5: Season vs Health Risk
plt.figure(figsize=(10, 5))
season_risk = pd.crosstab(df["Season"], df["Health_Risk"], normalize="index") * 100
season_risk[["Good","Moderate","Poor","Hazardous"]].plot(
    kind="bar", stacked=True,
    color=["#1D9E75","#EF9F27","#D85A30","#A32D2D"],
    edgecolor="white", ax=plt.gca()
)
plt.title("Health Risk by Season (%)")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("reports/figures/05_season_vs_risk.png", dpi=150)
plt.close()
print("Saved chart 5")

print("EDA complete! Check reports/figures/ folder")