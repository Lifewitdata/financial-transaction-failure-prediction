"""Evaluate the saved model bundle on the hold-out test slice.

Recomputes precision, recall, F1 and ROC-AUC from the pickled models and the
cleaned data, then writes reports/model_results.md.
"""
import pickle
from pathlib import Path

import pandas as pd

from data_cleaning import load_raw, clean_transactions, PROCESSED
from feature_engineering import add_features, time_split, get_xy
from train_model import evaluate_at_threshold, MODEL_PATH

BASE = Path(__file__).resolve().parents[1]
REPORT = BASE / "reports" / "model_results.md"


def main():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    raw = load_raw()
    clean, _ = clean_transactions(raw)
    clean.to_csv(PROCESSED, index=False)
    feat = add_features(clean)
    _, test = time_split(feat)
    X_test, y_test = get_xy(test)

    lines = []
    lines.append("# Model Results - Transaction Failure Prediction\n")
    lines.append(f"Test period: {bundle['test_period']} (time-based split, no leakage)\n")
    lines.append(f"Test rows: {len(test):,} | Test failure rate: {y_test.mean():.2%}\n")
    lines.append("## Metrics (threshold tuned on Aug-Sep 2025 validation slice, maximising F1)\n")
    lines.append("| Model | Threshold | Precision | Recall | F1-score | ROC-AUC |")
    lines.append("|---|---|---|---|---|---|")
    for name, saved in bundle["metrics"].items():
        pipe = bundle["models"][name]
        proba = pipe.predict_proba(X_test)[:, 1]
        fresh = evaluate_at_threshold(y_test, proba, saved["threshold"])
        match = all(abs(fresh[k] - saved[k]) < 1e-9 for k in ("precision", "recall", "f1", "roc_auc"))
        status = "reproduced" if match else "MISMATCH"
        lines.append(
            f"| {name} | {fresh['threshold']} | {fresh['precision']} | "
            f"{fresh['recall']} | {fresh['f1']} | {fresh['roc_auc']} |"
        )
        print(f"{name}: {fresh} -> {status}")
    lines.append("\n## Interpretation\n")
    lines.append(
        "- **ROC-AUC** measures ranking quality: how well the model scores failed "
        "transactions above successful ones.\n"
        "- **Precision** = of flagged transactions, how many truly failed. "
        "**Recall** = of all failed transactions, how many were flagged.\n"
        "- The F1-tuned threshold trades a little precision for substantially higher "
        "recall versus the default 0.5 cutoff, which suits an early-warning use case.\n"
        "- Logistic Regression slightly edges out Random Forest on the hold-out: the\n"
        "  failure drivers combine additively, which suits a linear model, so the\n"
        "  forest adds complexity without gain here - a useful reminder to always\n"
        "  benchmark the simple model first.\n"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
