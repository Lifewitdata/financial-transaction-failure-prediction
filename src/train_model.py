"""Train Logistic Regression and Random Forest failure-prediction models.

Pipeline:
  1. Load cleaned data, engineer features.
  2. Time split: train on Jan-Sep 2025, hold out Oct-Dec 2025 as test.
  3. Inside train: fit on Jan-Jul, tune the decision threshold on Aug-Sep
     (maximising F1), then refit on the full Jan-Sep train slice.
  4. Evaluate on the Oct-Dec test slice; save the bundle to
     models/transaction_failure_model.pkl.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from data_cleaning import load_raw, clean_transactions, PROCESSED
from feature_engineering import (
    add_features, build_preprocessor, time_split, get_xy, TARGET, TIMESTAMP,
)

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "transaction_failure_model.pkl"
VALIDATION_CUTOFF = "2025-08-01"  # fit: Jan-Jul, validate: Aug-Sep


def build_models():
    return {
        "logistic_regression": Pipeline(
            [
                ("pre", build_preprocessor()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("pre", build_preprocessor()),
                ("clf", RandomForestClassifier(
                    n_estimators=100, max_depth=14, min_samples_leaf=10,
                    class_weight="balanced", n_jobs=-1, random_state=42,
                )),
            ]
        ),
    }


def tune_threshold(y_true, proba):
    """Pick the threshold maximising F1 on the validation slice."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best = int(np.argmax(f1s))
    # precision_recall_curve returns len(thresholds) == len(f1s) - 1
    return float(thresholds[best]) if best < len(thresholds) else 0.5


def evaluate_at_threshold(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(threshold, 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "n_test": int(len(y_true)),
        "failure_rate_test": round(float(y_true.mean()), 4),
    }


def main():
    raw = load_raw()
    clean, _ = clean_transactions(raw)
    clean.to_csv(PROCESSED, index=False)
    feat = add_features(clean)

    train, test = time_split(feat)
    fit = train[train[TIMESTAMP] < VALIDATION_CUTOFF]
    valid = train[train[TIMESTAMP] >= VALIDATION_CUTOFF]
    X_fit, y_fit = get_xy(fit)
    X_valid, y_valid = get_xy(valid)
    X_train, y_train = get_xy(train)
    X_test, y_test = get_xy(test)
    print(f"fit: {len(fit):,} | valid: {len(valid):,} | test: {len(test):,}")

    models = build_models()
    results, fitted = {}, {}
    for name, pipe in models.items():
        pipe.fit(X_fit, y_fit)
        thr = tune_threshold(y_valid, pipe.predict_proba(X_valid)[:, 1])
        pipe.fit(X_train, y_train)  # refit on full train slice
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = evaluate_at_threshold(y_test, proba, thr)
        fitted[name] = pipe
        print(f"{name}: threshold={thr:.3f} " +
              " ".join(f"{k}={v}" for k, v in results[name].items() if k != "threshold"))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "models": fitted,
        "metrics": results,
        "validation_cutoff": VALIDATION_CUTOFF,
        "test_period": "2025-10-01 to 2025-12-31",
        "feature_lists": {"numeric": __import__("feature_engineering").NUMERIC,
                          "categorical": __import__("feature_engineering").CATEGORICAL},
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved bundle -> {MODEL_PATH}")
    return bundle, (X_test, y_test)


if __name__ == "__main__":
    main()
