"""Feature engineering for transaction failure prediction.

Builds model-ready variables from the cleaned transaction table, provides a
time-based train/test split (no leakage: the model is trained on Jan-Sep 2025
and evaluated on Oct-Dec 2025) and the shared sklearn preprocessor.

NOTE: `failure_reason` is intentionally excluded - it is only known after the
outcome, so using it would leak the target.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "is_failed"
TIMESTAMP = "timestamp"
CUTOFF = "2025-10-01"  # train: Jan-Sep 2025, test: Oct-Dec 2025

CATEGORICAL = ["payment_method", "merchant_category", "device_type", "currency", "issuer_bank"]
NUMERIC = [
    "transaction_amount",
    "log_amount",
    "hour",
    "hour_sin",
    "hour_cos",
    "is_night",
    "is_weekend",
    "authorization_response_time_ms",
    "account_age_days",
    "prior_failed_attempts_24h",
    "txn_velocity_24h",
    "is_high_risk_country",
    "is_new_account",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features and the binary target. Returns a new DataFrame."""
    df = df.copy()
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP])
    df[TARGET] = (df["status"] == "failed").astype(int)

    df["hour"] = df[TIMESTAMP].dt.hour
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["is_weekend"] = (df[TIMESTAMP].dt.dayofweek >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["log_amount"] = np.log1p(df["transaction_amount"])
    df["is_new_account"] = (df["account_age_days"] < 30).astype(int)
    df["is_high_risk_country"] = df["is_high_risk_country"].astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Scale numerics, one-hot encode categoricals."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def time_split(df: pd.DataFrame, cutoff: str = CUTOFF):
    """Split into train (< cutoff) and test (>= cutoff) on the timestamp."""
    train = df[df[TIMESTAMP] < cutoff].reset_index(drop=True)
    test = df[df[TIMESTAMP] >= cutoff].reset_index(drop=True)
    return train, test


def get_xy(df: pd.DataFrame):
    """Return (X, y) using the engineered feature lists."""
    return df[NUMERIC + CATEGORICAL], df[TARGET]
