"""Streamlit dashboard: transaction failure risk and model insights."""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "transaction_failure_model.pkl"
DATA_PATH = BASE / "data" / "processed" / "clean_transactions.csv"

st.set_page_config(page_title="Transaction Failure Risk", layout="wide")


@st.cache_resource
def load_bundle():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df["is_failed"] = (df["status"] == "failed").astype(int)
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    return df


bundle = load_bundle()
rf = bundle["models"]["random_forest"]
thr = bundle["metrics"]["random_forest"]["threshold"]
NUM = bundle["feature_lists"]["numeric"]
CAT = bundle["feature_lists"]["categorical"]

st.title("Transaction Failure Prediction")
page = st.sidebar.radio("Navigate", ["Overview", "Failure patterns", "Risk scorer", "Model insights"])

if page == "Overview":
    df = load_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Failure rate", f"{df['is_failed'].mean():.2%}")
    c3.metric("Failed transactions", f"{int(df['is_failed'].sum()):,}")
    c4.metric("Value at risk", f"${df.loc[df['is_failed'] == 1, 'transaction_amount'].sum():,.0f}")
    st.subheader("Monthly failure rate")
    monthly = df.groupby("month")["is_failed"].mean() * 100
    st.line_chart(monthly)
    st.subheader("Failures by hour of day")
    hourly = df.groupby("hour")["is_failed"].mean() * 100
    st.bar_chart(hourly)

elif page == "Failure patterns":
    df = load_data()
    st.subheader("Failure rate by payment method")
    st.bar_chart(df.groupby("payment_method")["is_failed"].mean().sort_values() * 100)
    st.subheader("Failure rate by merchant category")
    st.bar_chart(df.groupby("merchant_category")["is_failed"].mean().sort_values() * 100)
    st.subheader("Failure rate by issuer bank")
    st.bar_chart(df.groupby("issuer_bank")["is_failed"].mean().sort_values() * 100)
    st.subheader("Failure rate by corridor")
    corr = df.groupby("is_high_risk_country")["is_failed"].mean() * 100
    corr.index = ["standard", "high-risk"]
    st.bar_chart(corr)

elif page == "Risk scorer":
    st.subheader("Score a single transaction")
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction amount ($)", 1.0, 100000.0, 120.0)
        hour = st.slider("Hour of day", 0, 23, 14)
        is_weekend = st.checkbox("Weekend")
        payment_method = st.selectbox("Payment method",
            ["credit_card", "debit_card", "upi", "net_banking", "wallet"])
        merchant_category = st.selectbox("Merchant category",
            ["grocery", "fashion", "food_delivery", "travel", "electronics", "fuel", "health", "gaming"])
        device_type = st.selectbox("Device", ["mobile", "desktop", "tablet", "pos"])
    with col2:
        currency = st.selectbox("Currency", ["USD", "INR", "EUR", "GBP", "AED", "SGD"])
        issuer_bank = st.selectbox("Issuer bank",
            ["BankA", "BankB", "BankC", "BankD", "BankE", "BankF"])
        auth_ms = st.number_input("Authorization latency (ms)", 50, 30000, 800)
        account_age = st.number_input("Account age (days)", 1, 3000, 400)
        prior_fails = st.number_input("Prior failed attempts (24h)", 0, 20, 0)
        velocity = st.number_input("Transactions in last 24h", 1, 100, 3)
        high_risk = st.checkbox("High-risk corridor")

    if st.button("Score transaction", type="primary"):
        row = {
            "transaction_amount": amount, "log_amount": np.log1p(amount),
            "hour": hour, "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "is_night": int(hour <= 5), "is_weekend": int(is_weekend),
            "authorization_response_time_ms": auth_ms, "account_age_days": account_age,
            "prior_failed_attempts_24h": prior_fails, "txn_velocity_24h": velocity,
            "is_high_risk_country": int(high_risk), "is_new_account": int(account_age < 30),
            "payment_method": payment_method, "merchant_category": merchant_category,
            "device_type": device_type, "currency": currency, "issuer_bank": issuer_bank,
        }
        X = pd.DataFrame([row])[NUM + CAT]
        proba = float(rf.predict_proba(X)[0, 1])
        st.metric("Failure probability", f"{proba:.1%}")
        st.progress(min(proba, 1.0))
        if proba >= thr:
            st.error(f"High risk - above the alert threshold ({thr:.0%}). Recommend step-up authentication or manual review.")
        else:
            st.success(f"Low risk - below the alert threshold ({thr:.0%}).")
        flags = []
        if prior_fails > 0: flags.append(f"{prior_fails} prior failed attempt(s) in 24h")
        if auth_ms > 3000: flags.append("slow authorization (>3s)")
        if velocity > 12: flags.append("velocity burst (13+ txns/24h)")
        if account_age < 30: flags.append("new account (<30 days)")
        if high_risk: flags.append("high-risk corridor")
        if hour <= 5: flags.append("night-time transaction")
        if flags:
            st.warning("Risk flags: " + "; ".join(flags))

else:  # Model insights
    st.subheader("Hold-out test metrics (Oct-Dec 2025)")
    st.dataframe(pd.DataFrame(bundle["metrics"]).T.style.format("{:.4f}"))
    st.subheader("Top risk drivers (Random Forest)")
    names = rf.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names)
    top = imp.sort_values(ascending=False).head(12)
    st.bar_chart(top)
    st.caption(
        "Thresholds were tuned on Aug-Sep 2025 to maximise F1. "
        "Logistic Regression slightly edges out Random Forest here: the failure "
        "drivers combine additively, which suits a linear model."
    )
