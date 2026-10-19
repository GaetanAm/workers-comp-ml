# 🏥 Workers' Compensation Claims Valuation & Risk Triage

> **Data Mining & Machine Learning (DMML) Course Project**  
> **Authors:** Marius Occhiminuti · Gaëtan Amiens  
> **Dataset:** 100,000 Workers' Compensation Claims Records  

---

## 📌 Project Overview
In Workers' Compensation insurance, setting accurate financial reserves early in the claim lifecycle is critical for solvency and effective case management. Initial case estimates (`InitialCaseEstimate`) made by claims adjusters are often subjective and subject to major revisions.

**Objective:** Build a leak-proof, end-to-end Machine Learning pipeline to predict the final settlement cost (`UltimateIncurredClaimCost`) and identify severe high-cost claims (top 10% highest costs) early in the process.

---

## 🏗️ Pipeline & Leak-Proof Architecture
- **Dataset Size:** 100,000 records (80,000 Train / 20,000 Test).
- **Domain Feature Engineering:** Engineered `ReportDelayDays` calculated strictly as `(DateReported - DateTimeOfAccident) / 86400`.
- **Strict Leak-Proof Split:** Data is split into Train/Test sets *before* fitting any preprocessing transformers.
- **Target Transformation:** Models train on $\log(1 + y)$ with **upper-bound prediction clipping** at $\max(\log(1 + y_{\text{train}}))$ to stabilize heavy-tailed cost distributions and prevent exponential prediction explosion during $\exp(y) - 1$.

---

## 📊 Benchmark Results (Weeks 1–6 Models)

Evaluated on the held-out **20% test set** (20,000 claims):

| Model Family | RMSE ($) | MAE ($) | MAPE (%) | $R^2$ Score | Fit Time (s) | Status / Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`random_forest`** | **6,891.03** | 3,307.70 | **80.38%** | **+0.2512** | 0.22s | 🌲 Best $R^2$ & Overall RMSE |
| **`lightgbm`** | 6,935.99 | **3,301.41** | 81.00% | +0.2414 | 1.45s | ⚡ Best Absolute Error (MAE) |
| **`gradient_boosting`** | 7,087.54 | 3,331.90 | 80.83% | +0.2079 | 0.68s | 🚀 Strong Ensemble Performance |
| `lasso_regression` | 8,357.92 | 4,128.06 | 133.37% | -0.1016 | 0.08s | 📉 Baseline Regularized |
| `dummy_mean_baseline` | 50,442.06 | 11,189.54 | 313.25% | -0.0460 | 0.07s | 🎯 Naive Target Mean Baseline |
| `ridge_regression` | 8,937.65 | 3,793.88 | 90.81% | -0.2597 | 0.08s | Linear bias under heavy-tail |
| `linear_regression` | 8,992.77 | 3,802.13 | 90.83% | -0.2753 | 0.09s | Linear bias under heavy-tail |

### 💡 Key Empirical Takeaways
1. **20% Error Reduction:** Tree ensembles (Random Forest, LightGBM) reduce MAE from **$4,128.06** (Lasso) down to **$3,301.41**.
2. **Variance Explained ($R^2$):** Non-linear models boost $R^2$ from negative linear baseline scores to **+0.2512**.

---

## 🚀 Repository Structure

```text
workers-comp-ml/
├── .gitignore                 # Ignores data files, .venv, LaTeX build artifacts
├── README.md                  # Project overview and benchmark results
├── requirements.txt           # Python dependencies
├── src/
│   └── script.py              # Main benchmark pipeline (W1-W6 models)
└── proposal/
    ├── proposal.tex           # LaTeX source for Week 9 Proposal
    └── proposal.pdf           # Compiled 1-page proposal document