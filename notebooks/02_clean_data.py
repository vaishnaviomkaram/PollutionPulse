import pandas as pd
import numpy as np

# ── Load raw data ──────────────────────────────────────────
df = pd.read_csv("data/raw/city_day.csv")
print("Original shape:", df.shape)

# ── Step 1: Check missing values ───────────────────────────
print("\nMissing values in each column:")
print(df.isnull().sum())

# ── Step 2: Keep only columns we need ──────────────────────
cols_we_need = ["City", "Date", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI", "AQI_Bucket"]
df = df[cols_we_need]
print("\nAfter keeping only useful columns:", df.shape)

# ── Step 3: Drop rows where ALL pollutants are missing ──────
pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
before = len(df)
df.dropna(subset=pollutants, how="all", inplace=True)
print(f"Removed {before - len(df)} rows where all pollutants were missing")

# ── Step 4: Fill missing values with city average ──────────
# Smarter than global average — Delhi's average ≠ Mumbai's average
for col in pollutants:
    df[col] = df.groupby("City")[col].transform(
        lambda x: x.fillna(x.median())
    )
print("Filled missing pollutant values with city median")

# ── Step 5: Fix AQI — fill missing using PM2.5 ─────────────
# If AQI is missing but PM2.5 exists, estimate AQI
# Simple rule based on Indian CPCB standards
def estimate_aqi_bucket(pm25):
    if pd.isna(pm25):
        return np.nan
    elif pm25 <= 30:
        return "Good"
    elif pm25 <= 60:
        return "Satisfactory"
    elif pm25 <= 90:
        return "Moderate"
    elif pm25 <= 120:
        return "Poor"
    elif pm25 <= 250:
        return "Very Poor"
    else:
        return "Severe"

df["AQI_Bucket"] = df["AQI_Bucket"].fillna(df["PM2.5"].apply(estimate_aqi_bucket))
print("Filled missing AQI_Bucket using PM2.5 values")

# ── Step 6: Remove rows still missing AQI_Bucket ───────────
before = len(df)
df.dropna(subset=["AQI_Bucket"], inplace=True)
print(f"Removed {before - len(df)} rows still missing AQI label")

# ── Step 7: Add useful time features ───────────────────────
df["Date"] = pd.to_datetime(df["Date"])
df["Year"]  = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Season"] = df["Month"].apply(lambda m:
    "Winter"      if m in [12, 1, 2] else
    "Summer"      if m in [3, 4, 5]  else
    "Monsoon"     if m in [6, 7, 8, 9] else
    "Post_Monsoon"
)

# ── Step 8: Simplify AQI into 4 classes ────────────────────
# Original has 6 classes — we merge into 4 for cleaner ML
def simplify_bucket(bucket):
    if bucket in ["Good", "Satisfactory"]:
        return "Good"
    elif bucket == "Moderate":
        return "Moderate"
    elif bucket in ["Poor", "Very Poor"]:
        return "Poor"
    else:
        return "Hazardous"

df["Health_Risk"] = df["AQI_Bucket"].apply(simplify_bucket)

# ── Step 9: Check final class distribution ─────────────────
print("\nHealth Risk class distribution:")
print(df["Health_Risk"].value_counts())

# ── Step 10: Save cleaned data ─────────────────────────────
import os
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/cleaned_data.csv", index=False)
print("\n Saved cleaned data to data/processed/cleaned_data.csv")
print("Final shape:", df.shape)