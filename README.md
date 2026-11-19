# 🌿 PollutionPulse

An AI-powered air quality intelligence dashboard for India. Real monitoring-station data from 26 cities is analysed with a full machine-learning pipeline — regression, classification, clustering and dimensionality reduction — and surfaced as an interactive Streamlit app that predicts health risk, maps pollution hotspots and simulates "what-if" policy scenarios.

---

## What it does

| Page | Description |
| --- | --- |
| **Home** | Project overview, live dataset statistics, the health-risk scale and the models in use. |
| **City Dashboard** | PM2.5 trends, seasonal patterns and health-risk breakdowns for any of the 26 cities across 2015–2020. |
| **Health Risk Predictor** | Enter pollutant readings and get a risk class from two trained classifiers (SVM and Random Forest) with confidence scores. |
| **What-If Scenario Simulator** | Move sliders to change pollution levels and watch the risk gauge update live — e.g. factory shutdowns or Diwali night. |
| **Hotspot Map** | Geographic bubble map of India showing which cities have the worst air, coloured by cluster. |
| **Model Insights** | Confusion matrix, ROC curves, feature importance, learning curves and PCA city clusters — the ML engine behind the predictions. |
| **Personal Health Risk Advisor** | Adjusts the risk assessment for age and health profile: the same PM2.5 level is not equally dangerous for everyone. |
| **Pollution Alert System** | Simulates an automated alert feed that triggers when a city crosses a PM2.5 threshold you choose. |

---

## Dataset

- **Source:** India's Central Pollution Control Board (CPCB) monitoring stations, distributed as `city_day.csv` on Kaggle.
- **Coverage:** 26 Indian cities, 2015–2020, **27,951 cleaned daily records**.
- **Raw columns:** `City`, `Date`, `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`, `AQI`, `AQI_Bucket`.
- **Engineered columns:** `Year`, `Month`, `Season`, `Health_Risk`.

Nothing in this project is synthetic — every row is a real station reading.

### Health risk bands

| Class | PM2.5 (µg/m³) | Meaning |
| --- | --- | --- |
| **Good** | < 30 | Safe for everyone, including children and the elderly |
| **Moderate** | 30 – 90 | Sensitive groups should limit outdoor exposure |
| **Poor** | 90 – 120 | General population may feel health effects |
| **Hazardous** | > 120 | Health emergency — stay indoors, use an N95 mask |

Class distribution in the cleaned dataset: Good 11,922 · Moderate 9,192 · Poor 5,477 · Hazardous 1,360.

---

## Machine learning

**Classification** (health-risk class from `PM10, NO2, SO2, CO, O3, Month`):

| Model | Accuracy | F1-Score | Train time | Notes |
| --- | --- | --- | --- | --- |
| SVM (RBF kernel) | 82.77 % | 0.828 | 138.6 s | Best on non-linear class boundaries |
| Random Forest (SMOTE) | **85.62 %** | **0.857** | 6.4 s | Best overall; drives feature importance |
| Naive Bayes | 77.41 % | 0.772 | 0.02 s | Fast baseline, assumes feature independence |

**Regression:** Linear, Ridge and degree-2 Polynomial Regression for PM2.5 prediction, compared with R², RMSE and MAE plus bias–variance and learning curves.

**Clustering:** `K-Means` (elbow method selects the number of clusters) and `DBSCAN` for hotspot detection, with `PCA` used to visualise city clusters in two dimensions.

---

## Repository layout

```
PollutionPulse/
├── app/
│   ├── main.py                  # Landing page
│   └── pages/                   # One Streamlit page per feature
│       ├── 1_dashboard.py
│       ├── 2_predict.py
│       ├── 3_whatif.py
│       ├── 4_map.py
│       ├── 5_insights.py
│       ├── 6_personal_risk.py
│       └── 7_alerts.py
├── data/
│   ├── raw/city_day.csv         # Original CPCB export
│   └── processed/
│       ├── cleaned_data.csv     # Cleaned + feature-engineered
│       └── city_clusters.csv    # Per-city averages and cluster labels
├── models/                      # 11 serialised scikit-learn artefacts (.pkl)
├── notebooks/                   # Pipeline, run in numbered order
│   ├── 01_load_data.py
│   ├── 02_clean_data.py
│   ├── 03_eda.py
│   ├── 04_regression.py
│   ├── 05_classification.py
│   ├── 06_clustering.py
│   └── 07_evaluation.py
├── reports/
│   ├── figures/                 # 15 exported charts
│   └── model_comparison.csv
├── requirements.txt
└── README.md
```

---

## Getting started

Requires Python 3.10+.

```bash
git clone https://github.com/vaishnaviomkaram/PollutionPulse.git
cd PollutionPulse

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Run the dashboard **from the project root** — the pages load data and models with relative paths:

```bash
streamlit run app/main.py
```

Then open <http://localhost:8501>.

---

## Reproducing the pipeline

The trained models and processed CSVs are committed, so the app runs straight after `pip install`. To rebuild everything from the raw data, run the notebooks in order:

```bash
python notebooks/01_load_data.py
python notebooks/02_clean_data.py
python notebooks/03_eda.py
python notebooks/04_regression.py
python notebooks/05_classification.py
python notebooks/06_clustering.py
python notebooks/07_evaluation.py
```

Running them regenerates `data/processed/*.csv`, `models/*.pkl` and `reports/figures/*.png`.

> **Note:** the `.pkl` files were pickled with the scikit-learn version available at build time. If you see an `InconsistentVersionWarning` when loading a model, re-run notebooks `02`, `04`, `05` and `06` to regenerate them against your installed version — predictions are unaffected beyond that warning.

---

## Results at a glance

| Chart | File |
| --- | --- |
| Feature importance (Random Forest) | `reports/figures/09_feature_importance.png` |
| Confusion matrix | `reports/figures/08_confusion_matrix.png` |
| ROC curve | `reports/figures/14_roc_curve.png` |
| PCA city clusters | `reports/figures/12_pca_clusters.png` |
| Model comparison | `reports/figures/10_model_comparison.png` |

---

## Limitations

- The dataset ends in 2020 — the dashboard is an analysis of historical CPCB data, not a live feed.
- Health-risk labels are derived from CPCB PM2.5 thresholds, not from clinical outcomes.
- The personal-risk advisor is an educational heuristic built on the classifier output, not medical advice.

---

## Credits

- **Data:** Central Pollution Control Board (CPCB), India — via the `city_day.csv` dataset on Kaggle.
- **Stack:** Python, scikit-learn, imbalanced-learn, pandas, Plotly, Streamlit.
- Academic AI/ML project — 2nd Year CSE.
