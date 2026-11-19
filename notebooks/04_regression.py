import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

df = pd.read_csv("data/processed/cleaned_data.csv")

FEATURES = ["PM10", "NO2", "SO2", "CO", "O3", "Month"]
TARGET = "PM2.5"

df_model = df[FEATURES + [TARGET]].dropna()
X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
print("--- Linear Regression ---")
print(f"R2: {r2_score(y_test, y_pred_lr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.4f}")

# Model 2: Polynomial Regression
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_s)
X_test_poly  = poly.transform(X_test_s)
pr = LinearRegression()
pr.fit(X_train_poly, y_train)
y_pred_pr = pr.predict(X_test_poly)
print("\n--- Polynomial Regression (degree 2) ---")
print(f"R2: {r2_score(y_test, y_pred_pr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_pr)):.4f}")

# Model 3: Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_s, y_train)
y_pred_ridge = ridge.predict(X_test_s)
print("\n--- Ridge Regression ---")
print(f"R2: {r2_score(y_test, y_pred_ridge):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}")

# Chart: Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Actual", color="#185FA5", linewidth=2)
plt.plot(y_pred_pr[:100], label="Predicted", color="#D85A30", linewidth=2, linestyle="--")
plt.title("Actual vs Predicted PM2.5")
plt.ylabel("PM2.5")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/figures/06_regression_prediction.png", dpi=150)
plt.close()

# Chart: Bias-Variance
degrees = [1, 2, 3]
train_scores, test_scores = [], []
for d in degrees:
    pf = PolynomialFeatures(degree=d, include_bias=False)
    Xtr = pf.fit_transform(X_train_s)
    Xte = pf.transform(X_test_s)
    m = LinearRegression().fit(Xtr, y_train)
    train_scores.append(r2_score(y_train, m.predict(Xtr)))
    test_scores.append(r2_score(y_test, m.predict(Xte)))

plt.figure(figsize=(7, 5))
plt.plot(degrees, train_scores, marker="o", label="Train R2", color="#1D9E75")
plt.plot(degrees, test_scores,  marker="o", label="Test R2",  color="#D85A30")
plt.title("Bias-Variance Tradeoff")
plt.xlabel("Polynomial Degree")
plt.ylabel("R2 Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/figures/07_bias_variance.png", dpi=150)
plt.close()

joblib.dump(pr,     "models/poly_regression.pkl")
joblib.dump(poly,   "models/poly_features.pkl")
joblib.dump(scaler, "models/regression_scaler.pkl")
print("\nRegression complete! Models saved.")