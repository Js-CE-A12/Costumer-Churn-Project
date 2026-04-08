# Customer Churn Prediction — Telco Dataset

Predict which telecom customers are likely to cancel their subscription so the business can act before they leave.

---

## Problem Statement

Customer churn is one of the most costly problems in subscription businesses. Acquiring a new customer costs several times more than retaining an existing one. This project builds a binary classification model that outputs, for each customer, the **probability of churning** — giving the retention team a ranked list of at-risk customers to contact.

---

## Dataset

**Source:** IBM Telco Customer Churn (publicly available on Kaggle / IBM Sample Data Sets)

| Property | Value |
|---|---|
| Rows | 7 043 customers |
| Columns | 21 features |
| Target | `Churn` (Yes / No → 1 / 0) |
| Class balance | ~73 % No churn / ~27 % Churn |

Key features:
- **tenure** — months with the company (strong negative correlation with churn)
- **MonthlyCharges** — monthly bill
- **Contract** — Month-to-month / One year / Two year (month-to-month → 43 % churn rate)
- Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Services: `PhoneService`, `InternetService`, `OnlineSecurity`, streaming, etc.

---

## Project Structure

```
customer-churn-project/
├── data/
│   └── Telco-Customer-Churn.csv   # Raw dataset
├── notebooks/
│   ├── 01_EDA.ipynb               # Full analysis + modelling notebook
│   └── churn_predictions.csv      # Model output (Actual vs Predicted_Prob)
└── Readme.md
```

---

## Notebook Walkthrough (`01_EDA.ipynb`)

| Step | What we do |
|---|---|
| 1 | Load data, inspect structure (`df.head`) |
| 2 | Check dimensions (`df.shape` → 7 043 × 21) |
| 3 | Inspect dtypes and nulls (`df.info`) — discover `TotalCharges` stored as string |
| 4 | Descriptive statistics (`df.describe`) |
| 5 | Null count — 11 hidden missing values surface after type conversion |
| 6 | Churn distribution — **27 % churn rate** (class imbalance) |
| 7 | Histograms — bi-modal `tenure` distribution reveals early-churn risk |
| 8 | Crosstab Contract × Churn — month-to-month customers churn 4× more |
| 9 | Drop `customerID` (no predictive value) |
| 10 | Convert `TotalCharges` to `float64` with `pd.to_numeric(errors='coerce')` |
| 11 | Drop 11 rows with missing `TotalCharges` |
| 12 | Box-plot Tenure vs Churn — churned customers have median tenure ~10 months vs ~38 months |
| 13 | Encode target: `Yes → 1`, `No → 0` |
| 14 | Correlation heatmap — `tenure` (−0.35) and `MonthlyCharges` (+0.19) most correlated with churn |
| 15 | One-hot encode categoricals, train/test split (80/20) |
| 16 | Train three baseline models (Logistic Regression wins at 78.8 % accuracy) |
| 17 | Scale features with `StandardScaler` |
| 18 | Tune `C` with `GridSearchCV` — best C = 10, CV score 80.7 % |
| 19 | Handle class imbalance with oversampling (`resample`) |
| 20 | Confusion matrix + classification report |
| 21 | ROC curve — **AUC = 0.83** |
| 22 | Custom threshold (0.3) — prioritises recall over precision to catch more churners |

---

## Key Results

| Metric | Value |
|---|---|
| Best model | Logistic Regression (C=10, scaled features) |
| Accuracy | ~78.9 % |
| AUC-ROC | **0.83** |
| Threshold used | 0.3 (optimised for recall) |

At threshold = 0.3:
- Recall on churn class: **90 %** (catches 9 out of 10 actual churners)
- Precision on churn class: 42 % (some false alarms)

This trade-off is intentional — in a churn context it is cheaper to contact a customer who was *not* going to churn than to miss one who was.

---

## How to Run

```bash
# 1. Clone / download the repo
# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab

# 3. Launch the notebook
jupyter lab notebooks/01_EDA.ipynb
```

> The notebook was developed with Python 3.13 and scikit-learn 1.x. All cells are pre-executed and outputs are saved.

---

## Next Steps / Ideas

- Try gradient boosting (XGBoost / LightGBM) for potentially higher AUC
- Feature engineering: ratio features (e.g. `TotalCharges / tenure`), interaction terms
- SHAP values for model explainability
- Deploy as a REST API (Flask / FastAPI) for real-time scoring
