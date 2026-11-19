import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

st.set_page_config(page_title="Model Insights", layout="wide")
st.title("Model Insights")
st.markdown("The ML engine behind PollutionPulse — see how each model performs.")

@st.cache_resource
def load_everything():
    return (joblib.load("models/svm_model.pkl"),
            joblib.load("models/random_forest.pkl"),
            joblib.load("models/naive_bayes.pkl"),
            joblib.load("models/classifier_scaler.pkl"),
            joblib.load("models/label_encoder.pkl"),
            joblib.load("models/pca.pkl"),
            joblib.load("models/kmeans_scaler.pkl"))

svm, rf, nb, scaler, le, pca, km_sc = load_everything()

@st.cache_data
def get_test_data():
    df = pd.read_csv("data/processed/cleaned_data.csv")
    FEATURES = ["PM2.5","PM10","NO2","SO2","CO","O3","Month"]
    df_m = df[FEATURES + ["Health_Risk"]].dropna()
    X = df_m[FEATURES]
    y = le.transform(df_m["Health_Risk"])
    X_r, y_r = SMOTE(random_state=42).fit_resample(X, y)
    _, X_test, _, y_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    return scaler.transform(X_test), y_test

X_test_s, y_test = get_test_data()

svm_acc = (svm.predict(X_test_s) == y_test).mean()
rf_acc  = (rf.predict(X_test_s)  == y_test).mean()
nb_acc  = (nb.predict(X_test_s)  == y_test).mean()

# ── Score cards at top ─────────────────────────────────────
st.markdown("### Model Accuracy")
c1, c2, c3 = st.columns(3)
c1.metric("SVM (RBF Kernel)",    f"{svm_acc*100:.1f}%", "Best performer")
c2.metric("Random Forest",       f"{rf_acc*100:.1f}%")
c3.metric("Naive Bayes",         f"{nb_acc*100:.1f}%", "Baseline")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix","Feature Importance","PCA Clusters","ROC Curve"])

plt.rcParams.update({"figure.facecolor":"#0f1116","axes.facecolor":"#1a1d24",
                     "text.color":"white","axes.labelcolor":"white",
                     "xtick.color":"white","ytick.color":"white",
                     "axes.edgecolor":"#2a2d35"})

with tab1:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, svm.predict(X_test_s))
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt="d",
                cmap=sns.color_palette("light:#00d4aa", as_cmap=True),
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
                linewidths=0.5, linecolor="#0f1116")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("SVM Confusion Matrix", color="white", pad=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.info("Rows = actual class, Columns = what the model predicted. Diagonal = correct predictions.")

with tab2:
    FEATURES = ["PM2.5","PM10","NO2","SO2","CO","O3","Month"]
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig2, ax2 = plt.subplots(figsize=(8,5))
    bars = ax2.bar(range(len(FEATURES)), imp[idx],
                   color=["#00d4aa" if i==0 else "#185FA5" for i in range(len(FEATURES))],
                   edgecolor="#0f1116", width=0.6)
    ax2.set_xticks(range(len(FEATURES)))
    ax2.set_xticklabels([FEATURES[i] for i in idx], rotation=30)
    ax2.set_title("Feature Importance — Random Forest", color="white", pad=12)
    ax2.set_ylabel("Importance Score")
    ax2.grid(axis="y", alpha=0.3, color="#2a2d35")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    most_imp = FEATURES[idx[0]]
    st.success(f"**{most_imp}** is the strongest predictor of health risk. This matches real-world science — PM2.5 particles penetrate deepest into lungs.")

with tab3:
    city_profile = pd.read_csv("data/processed/city_clusters.csv")
    FEAT = ["PM2.5","PM10","NO2","SO2","CO","O3"]
    X_c   = km_sc.transform(city_profile[FEAT])
    X_pca = pca.transform(X_c)
    colors_c = {0:"#00d4aa", 1:"#ffaa00", 2:"#ff4444"}
    labels_c = {0:"Low Pollution", 1:"Medium Pollution", 2:"High Pollution"}
    ev = pca.explained_variance_ratio_
    fig3, ax3 = plt.subplots(figsize=(10,7))
    for cid in city_profile["Cluster"].unique():
        mask = city_profile["Cluster"]==cid
        ax3.scatter(X_pca[mask,0], X_pca[mask,1],
                    label=labels_c.get(cid,str(cid)),
                    color=colors_c.get(cid,"gray"),
                    s=150, edgecolors="#0f1116", linewidth=1.5, zorder=3)
    for i, row in city_profile.iterrows():
        ax3.annotate(row["City"],(X_pca[i,0]+0.08,X_pca[i,1]+0.08),
                     fontsize=8, color="white", alpha=0.8)
    ax3.set_xlabel(f"PC1 ({ev[0]:.1%} variance)", color="white")
    ax3.set_ylabel(f"PC2 ({ev[1]:.1%} variance)", color="white")
    ax3.set_title("City Pollution Clusters in PCA Space", color="white", pad=12)
    ax3.legend(facecolor="#1a1d24", edgecolor="#2a2d35", labelcolor="white")
    ax3.grid(alpha=0.2, color="#2a2d35")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    st.info(f"PCA reduces 6 pollution dimensions into 2. These 2 components explain {sum(ev):.1%} of total variance.")

with tab4:
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    try:
        classes_list = list(range(len(le.classes_)))
        y_bin   = label_binarize(y_test, classes=classes_list)
        y_score = svm.predict_proba(X_test_s)

        color_roc = {
            "Good":      "#00d4aa",
            "Moderate":  "#ffaa00",
            "Poor":      "#ff6600",
            "Hazardous": "#ff4444"
        }

        fig4, ax4 = plt.subplots(figsize=(9, 7))

        for i, cls in enumerate(le.classes_):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax4.plot(fpr, tpr,
                     color=color_roc.get(cls, "white"),
                     linewidth=2.5,
                     label=f"{cls} (AUC = {roc_auc:.2f})")

        ax4.plot([0,1], [0,1], "--", color="#444", linewidth=1, label="Random Classifier")
        ax4.set_facecolor("#1a1d24")
        fig4.patch.set_facecolor("#0f1116")
        ax4.set_title("ROC Curve — SVM", color="white", pad=12)
        ax4.set_xlabel("False Positive Rate", color="white")
        ax4.set_ylabel("True Positive Rate", color="white")
        ax4.tick_params(colors="white")
        ax4.spines["bottom"].set_color("#2a2d35")
        ax4.spines["left"].set_color("#2a2d35")
        ax4.spines["top"].set_color("#2a2d35")
        ax4.spines["right"].set_color("#2a2d35")
        ax4.legend(
            facecolor="#1a1d24",
            edgecolor="#2a2d35",
            labelcolor="white"
        )
        ax4.grid(alpha=0.2, color="#2a2d35")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
        st.info("AUC closer to 1.0 = better model. The dashed line = random guesser (AUC 0.5).")

    except Exception as e:
        st.error(f"ROC curve error: {e}")