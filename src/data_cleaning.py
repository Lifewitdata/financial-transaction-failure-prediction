"""Data cleaning for the transaction failure prediction project.

Reads data/raw/transactions.csv and writes data/processed/clean_transactions.csv
with duplicates removed, missing values imputed and invalid values corrected.
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "transactions.csv"
PROCESSED = BASE / "data" / "processed" / "clean_transactions.csv"

NUMERIC_IMPUTE_MEDIAN = [
    "transaction_amount",
    "authorization_response_time_ms",
    "account_age_days",
]
CATEGORICAL_IMPUTE_MODE = ["device_type", "issuer_bank"]


def load_raw(path=RAW) -> pd.DataFrame:
    """Load the raw transaction file."""
    return pd.read_csv(path, parse_dates=["timestamp"])


def clean_transactions(df: pd.DataFrame):
    """Return (cleaned_df, report_dict)."""
    df = df.copy()
    report = {"rows_in": len(df)}

    # 1. duplicates
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    # 2. missing values
    report["missing_values_before"] = int(df.isna().sum().sum())
    for col in NUMERIC_IMPUTE_MEDIAN:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_IMPUTE_MODE:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    report["missing_values_after"] = int(df.isna().sum().sum())

    # 3. invalid values
    df["transaction_amount"] = df["transaction_amount"].clip(lower=1.0)
    cap = df["transaction_amount"].quantile(0.999)
    report["amount_outliers_capped_at_99_9pct"] = int((df["transaction_amount"] > cap).sum())
    df["transaction_amount"] = df["transaction_amount"].clip(upper=cap)
    df["authorization_response_time_ms"] = df["authorization_response_time_ms"].clip(lower=50)
    df["prior_failed_attempts_24h"] = df["prior_failed_attempts_24h"].clip(lower=0)
    df["txn_velocity_24h"] = df["txn_velocity_24h"].clip(lower=1)

    df = df.sort_values("timestamp").reset_index(drop=True)
    report["rows_out"] = len(df)
    return df, report


def main():
    df = load_raw()
    clean, report = clean_transactions(df)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED, index=False)
    print("Cleaning report:")
    for k, v in report.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    print(f"Wrote {PROCESSED} ({len(clean):,} rows)")


if __name__ == "__main__":
    main()
