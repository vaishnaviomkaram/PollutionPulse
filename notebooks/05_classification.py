import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

df = pd.read_csv("data/processed/cleaned_data.csv")

FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Month"]
TARGET   = "Health_Risk"

df_model = df[FEATURES + [TARGET]].dropna()
X = df_model[FEATURES]
y = df_model[TARGET]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", list(le.classes_))
print("Original class distribution:", dict(zip(le.classes_, np.bincount(y_encoded))))

# ── CORRECT ORDER: Split FIRST then SMOTE ─────────────────
# If you do SMOTE before split → synthetic samples leak into
# test set → artificially inflated accuracy → data leakage
print("\nStep 1: Train-test split FIRST (before SMOTE)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded  # keeps class ratios same in both splits
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── SMOTE only on training data ────────────────────────────
print("\nStep 2: Apply SMOTE only on training data")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Train class distribution AFTER SMOTE:",
      dict(zip(le.classes_, np.bincount(y_train))))
print("Test class distribution (original, untouched):",
      dict(zip(le.classes_, np.bincount(y_test))))

# ── Scale features ─────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Train all 3 models + record metrics ───────────────────
results = []

# SVM
print("\nTraining SVM (RBF kernel)...")
t0  = time.time()
svm = SVC(kernel="rbf", probability=True, random_state=42, C=10, gamma="scale")
svm.fit(X_train_s, y_train)
svm_time = round(time.time() - t0, 2)
svm_acc  = svm.score(X_test_s, y_test)
svm_f1   = f1_score(y_test, svm.predict(X_test_s), average="weighted")
print(f"SVM — Accuracy: {svm_acc:.4f} | F1: {svm_f1:.4f} | Time: {svm_time}s")
print(classification_report(y_test, svm.predict(X_test_s), target_names=le.classes_))
results.append({"Model":"SVM (RBF Kernel)", "Accuracy":f"{svm_acc*100:.2f}%",
                "F1-Score":f"{svm_f1:.4f}", "Training Time":f"{svm_time}s",
                "Best For":"Non-linear boundaries between risk categories"})

# Random Forest
print("Training Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
rf_time = round(time.time() - t0, 2)
rf_acc  = rf.score(X_test_s, y_test)
rf_f1   = f1_score(y_test, rf.predict(X_test_s), average="weighted")
print(f"RF  — Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f} | Time: {rf_time}s")
results.append({"Model":"Random Forest", "Accuracy":f"{rf_acc*100:.2f}%",
                "F1-Score":f"{rf_f1:.4f}", "Training Time":f"{rf_time}s",
                "Best For":"Feature importance + handles correlated features"})

# Naive Bayes
print("Training Naive Bayes...")
t0 = time.time()
nb = GaussianNB()
nb.fit(X_train_s, y_train)
nb_time = round(time.time() - t0, 2)
nb_acc  = nb.score(X_test_s, y_test)
nb_f1   = f1_score(y_test, nb.predict(X_test_s), average="weighted")
print(f"NB  — Accuracy: {nb_acc:.4f} | F1: {nb_f1:.4f} | Time: {nb_time}s")
results.append({"Model":"Naive Bayes", "Accuracy":f"{nb_acc*100:.2f}%",
                "F1-Score":f"{nb_f1:.4f}", "Training Time":f"{nb_time}s",
                "Best For":"Baseline — fast but assumes feature independence"})

# ── Model comparison table ─────────────────────────────────
comparison_df = pd.DataFrame(results)
print("\n=== MODEL COMPARISON TABLE ===")
print(comparison_df.to_string(index=False))
comparison_df.to_csv("reports/model_comparison.csv", index=False)
print("Saved: reports/model_comparison.csv")

# ── Confusion Matrix ───────────────────────────────────────
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, svm.predict(X_test_s))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("SVM Confusion Matrix (test set — no SMOTE contamination)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("reports/figures/08_confusion_matrix.png", dpi=150)
plt.close()
print("Saved: 08_confusion_matrix.png")

# ── Feature Importance ─────────────────────────────────────
plt.figure(figsize=(8, 5))
importances = rf.feature_importances_
idx = np.argsort(importances)[::-1]
plt.bar(range(len(FEATURES)), importances[idx], color="#185FA5", edgecolor="white")
plt.xticks(range(len(FEATURES)), [FEATURES[i] for i in idx], rotation=30)
plt.title("Feature Importance — Random Forest\n(confirms PM2.5 dominance — validates our labelling approach)")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("reports/figures/09_feature_importance.png", dpi=150)
plt.close()
print("Saved: 09_feature_importance.png")

# ── Model accuracy comparison chart ───────────────────────
plt.figure(figsize=(8, 5))
models  = ["SVM (RBF)", "Random Forest", "Naive Bayes"]
scores  = [svm_acc, rf_acc, nb_acc]
f1s     = [svm_f1, rf_f1, nb_f1]
x       = np.arange(len(models))
width   = 0.35

bars1 = plt.bar(x - width/2, scores, width, label="Accuracy",
                color="#185FA5", edgecolor="white")
bars2 = plt.bar(x + width/2, f1s, width, label="F1-Score",
                color="#1D9E75", edgecolor="white")

for bar in bars1:
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f"{bar.get_height():.3f}", ha="center", fontsize=9)
for bar in bars2:
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f"{bar.get_height():.3f}", ha="center", fontsize=9)

plt.title("Model Comparison — Accuracy vs F1-Score")
plt.xticks(x, models)
plt.ylabel("Score")
plt.ylim(0, 1.15)
plt.legend()
plt.tight_layout()
plt.savefig("reports/figures/10_model_comparison.png", dpi=150)
plt.close()
print("Saved: 10_model_comparison.png")

# ── Save everything ────────────────────────────────────────
joblib.dump(svm,    "models/svm_model.pkl")
joblib.dump(rf,     "models/random_forest.pkl")
joblib.dump(nb,     "models/naive_bayes.pkl")
joblib.dump(scaler, "models/classifier_scaler.pkl")
joblib.dump(le,     "models/label_encoder.pkl")
print("\nAll models saved to models/")
print("Classification complete!")