import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

os.makedirs("reports/figures", exist_ok=True)

svm    = joblib.load("models/svm_model.pkl")
rf     = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/classifier_scaler.pkl")
le     = joblib.load("models/label_encoder.pkl")

df = pd.read_csv("data/processed/cleaned_data.csv")
FEATURES = ["PM2.5","PM10","NO2","SO2","CO","O3","Month"]
df_m = df[FEATURES + ["Health_Risk"]].dropna()
X = df_m[FEATURES]
y = le.transform(df_m["Health_Risk"])

smote = SMOTE(random_state=42)
X_r, y_r = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# ROC Curve
y_bin   = label_binarize(y_test, classes=range(len(le.classes_)))
y_score = svm.predict_proba(X_test_s)

plt.figure(figsize=(9, 7))
colors = ["#185FA5","#1D9E75","#D85A30","#A32D2D"]
for i, (cls, col) in enumerate(zip(le.classes_, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:,i], y_score[:,i])
    plt.plot(fpr, tpr, color=col, linewidth=2, label=f"{cls} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],"k--")
plt.title("ROC Curve — SVM (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/figures/14_roc_curve.png", dpi=150)
plt.close()
print("Saved ROC curve")

# Learning Curve
rf_lc = RandomForestClassifier(n_estimators=50, random_state=42)
train_sizes, train_sc, test_sc = learning_curve(
    rf_lc, X_train_s, y_train,
    train_sizes=np.linspace(0.1,1.0,8), cv=3, scoring="accuracy", n_jobs=-1)

plt.figure(figsize=(9, 6))
plt.plot(train_sizes, train_sc.mean(axis=1), marker="o", color="#185FA5", label="Train Score")
plt.plot(train_sizes, test_sc.mean(axis=1),  marker="o", color="#D85A30", label="Val Score")
plt.fill_between(train_sizes, train_sc.mean(axis=1)-train_sc.std(axis=1),
                              train_sc.mean(axis=1)+train_sc.std(axis=1), alpha=0.1, color="#185FA5")
plt.fill_between(train_sizes, test_sc.mean(axis=1)-test_sc.std(axis=1),
                              test_sc.mean(axis=1)+test_sc.std(axis=1),  alpha=0.1, color="#D85A30")
plt.title("Learning Curve — Random Forest")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reports/figures/15_learning_curve.png", dpi=150)
plt.close()
print("Saved learning curve")

# Failure Analysis
y_pred = svm.predict(X_test_s)
errors_df = pd.DataFrame({
    "Actual":    le.inverse_transform(y_test),
    "Predicted": le.inverse_transform(y_pred)
})
errors = errors_df[errors_df["Actual"] != errors_df["Predicted"]]
print(f"\nMisclassified: {len(errors)}/{len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)")
print("Common mistakes:")
print(pd.crosstab(errors["Actual"], errors["Predicted"]))

print("\nEvaluation complete!")