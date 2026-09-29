# Financial Transaction Failure Prediction

Predict which payment transactions will fail settlement, so a payments team can
intervene early (step-up authentication, manual review, issuer routing) instead of
losing the sale.

## What is inside

- **Synthetic transaction data** for all of 2025 (~180k rows) with realistic failure
  drivers: prior failed attempts, slow authorizations, night-time traffic, velocity
  bursts, new accounts, high-risk corridors, channel and issuer effects.
- **EDA** showing exactly where failures concentrate.
- **Two models** - Logistic Regression vs Random Forest - trained on Jan-Sep 2025 and
  evaluated on a strict time-based hold-out (Oct-Dec 2025, no leakage). Decision
  thresholds tuned on Aug-Sep 2025 to maximise F1.
- **Streamlit dashboard** with failure-pattern explorer and a live single-transaction
  risk scorer.
- Full results in `reports/model_results.md`.

## Project structure

```
financial-transaction-failure-prediction/
├── data/
│   ├── raw/transactions.csv            # generated dataset
│   └── processed/clean_transactions.csv
├── notebooks/
│   ├── 01_data_generation.ipynb        # build + inspect the dataset (executed)
│   ├── 02_eda.ipynb                   # failure-pattern analysis (executed)
│   └── 03_model_training.ipynb        # train, compare, interpret (executed)
├── src/
│   ├── data_cleaning.py               # dedupe, impute, fix invalid values
│   ├── feature_engineering.py         # features, time split, preprocessor
│   ├── train_model.py                 # train LR + RF, tune threshold, save bundle
│   └── evaluate_model.py              # reproduce metrics -> reports/model_results.md
├── models/transaction_failure_model.pkl
├── dashboard/app.py                   # Streamlit app
├── reports/model_results.md
├── requirements.txt
├── README.md
└── .gitignore
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. generate the data (or open notebooks/01_data_generation.ipynb)
# 2. run the pipeline end to end:
python src/data_cleaning.py
python src/train_model.py
python src/evaluate_model.py

# 3. launch the dashboard
streamlit run dashboard/app.py
```

Run order mirrors `notebooks/`: generation -> EDA -> cleaning -> features ->
training -> evaluation -> dashboard -> docs.

## Key findings

- Failures concentrate at night, on slow authorizations (>3s), after prior failed
  attempts, on velocity bursts and new accounts - behavioural signals beat demographics.
- Logistic Regression slightly beats Random Forest on the hold-out
  (see `reports/model_results.md`): the failure drivers combine additively, which
  suits a linear model - a good reminder to benchmark the simple model first.
- The F1-tuned threshold favours recall: better to review a good transaction than
  miss a failing one.

## Notes

- `failure_reason` is excluded from modelling - it is only known after the outcome
  (target leakage).
- Data is synthetic (seed 42, reproducible) and built for portfolio demonstration.
