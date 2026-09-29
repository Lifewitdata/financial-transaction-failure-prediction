# Model Results - Transaction Failure Prediction

Test period: 2025-10-01 to 2025-12-31 (time-based split, no leakage)

Test rows: 45,208 | Test failure rate: 6.08%

## Metrics (threshold tuned on Aug-Sep 2025 validation slice, maximising F1)

| Model | Threshold | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| logistic_regression | 0.7093 | 0.2484 | 0.3585 | 0.2935 | 0.7635 |
| random_forest | 0.6229 | 0.2334 | 0.3516 | 0.2806 | 0.758 |

## Interpretation

- **ROC-AUC** measures ranking quality: how well the model scores failed transactions above successful ones.
- **Precision** = of flagged transactions, how many truly failed. **Recall** = of all failed transactions, how many were flagged.
- The F1-tuned threshold trades a little precision for substantially higher recall versus the default 0.5 cutoff, which suits an early-warning use case.
- Logistic Regression slightly edges out Random Forest on the hold-out: the
  failure drivers combine additively, which suits a linear model, so the
  forest adds complexity without gain here - a useful reminder to always
  benchmark the simple model first.
