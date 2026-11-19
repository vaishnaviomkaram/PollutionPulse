import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os

os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

df = pd.read_csv("data/processed/cleaned_data.csv")

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
city_profile = df.groupby("City")[FEATURES].median().reset_index()

city_profile = city_profile.dropna().reset_index(drop=True)
print(f"Cities after cleaning: {len(city_profile)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(city_profile[FEATURES])

# Elbow curve
inertias, sil_scores = [], []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(2,8), inertias, marker="o", color="#185FA5")
ax1.set_title("Elbow Curve")
ax1.set_xlabel("k")
ax1.set_ylabel("Inertia")
ax2.plot(range(2,8), sil_scores, marker="o", color="#1D9E75")
ax2.set_title("Silhouette Score")
ax2.set_xlabel("k")
plt.tight_layout()
plt.savefig("reports/figures/11_elbow_curve.png", dpi=150)
plt.close()
print("Saved elbow curve")

# Final KMeans k=3
km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
city_profile["Cluster"] = km_final.fit_predict(X_scaled)

for cid in range(3):
    cities = city_profile[city_profile["Cluster"]==cid]["City"].tolist()
    avg_pm = city_profile[city_profile["Cluster"]==cid]["PM2.5"].mean()
    print(f"Cluster {cid} (avg PM2.5={avg_pm:.1f}): {cities}")

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Variance explained: {sum(pca.explained_variance_ratio_):.2%}")

plt.figure(figsize=(10, 7))
colors = {0:"#1D9E75", 1:"#EF9F27", 2:"#D85A30"}
labels = {0:"Low Pollution", 1:"Medium Pollution", 2:"High Pollution"}
for cid in range(3):
    mask = city_profile["Cluster"] == cid
    plt.scatter(X_pca[mask,0], X_pca[mask,1],
                label=labels[cid], color=colors[cid], s=120, edgecolors="white")
for i, row in city_profile.iterrows():
    plt.annotate(row["City"], (X_pca[i,0]+0.05, X_pca[i,1]+0.05), fontsize=8)
plt.title("City Pollution Clusters (PCA 2D)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/figures/12_pca_clusters.png", dpi=150)
plt.close()
print("Saved PCA chart")

# DBSCAN
df_daily = df[["City","Date","PM2.5","PM10"]].dropna().copy()
scaler_db = StandardScaler()
X_db = scaler_db.fit_transform(df_daily[["PM2.5","PM10"]])
db = DBSCAN(eps=0.5, min_samples=10)
df_daily["DBSCAN_Cluster"] = db.fit_predict(X_db)
n_clusters = len(set(df_daily["DBSCAN_Cluster"])) - (1 if -1 in df_daily["DBSCAN_Cluster"].values else 0)
n_noise    = (df_daily["DBSCAN_Cluster"] == -1).sum()
print(f"DBSCAN: {n_clusters} clusters, {n_noise} outliers")

plt.figure(figsize=(10, 6))
plt.scatter(df_daily["PM2.5"], df_daily["PM10"],
            c=df_daily["DBSCAN_Cluster"], cmap="tab10", alpha=0.4, s=5)
plt.title(f"DBSCAN Clustering ({n_clusters} clusters, {n_noise} outliers)")
plt.xlabel("PM2.5")
plt.ylabel("PM10")
plt.tight_layout()
plt.savefig("reports/figures/13_dbscan.png", dpi=150)
plt.close()
print("Saved DBSCAN chart")

joblib.dump(km_final, "models/kmeans.pkl")
joblib.dump(scaler,   "models/kmeans_scaler.pkl")
joblib.dump(pca,      "models/pca.pkl")
city_profile.to_csv("data/processed/city_clusters.csv", index=False)
print("Clustering complete!")