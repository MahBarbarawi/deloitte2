"""Metric calculations kept separate from display code."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import load_dashboard_data


def fraud_summary(data: pd.DataFrame) -> dict[str, float]:
    transactions = len(data)
    fraud_count = int(data["fraud"].sum()) if transactions else 0
    return {
        "transactions": transactions,
        "fraud_count": fraud_count,
        "legitimate_count": transactions - fraud_count,
        "fraud_rate": fraud_count / transactions if transactions else 0.0,
        "fraud_amount": float(data.loc[data.fraud.eq(1), "amount"].sum()) if transactions else 0.0,
    }


def grouped_fraud(data: pd.DataFrame, group: str) -> pd.DataFrame:
    grouped = data.groupby(group, observed=True).agg(
        transaction_count=("fraud", "size"), fraud_count=("fraud", "sum"), total_amount=("amount", "sum"),
    )
    fraud_amount = data.loc[data["fraud"].eq(1)].groupby(group, observed=True)["amount"].sum()
    grouped["fraud_amount"] = fraud_amount.reindex(grouped.index, fill_value=0)
    grouped["fraud_rate"] = grouped["fraud_count"] / grouped["transaction_count"]
    return grouped.reset_index()


@st.cache_data(show_spinner=False, max_entries=24)
def build_fraud_analysis(
    step_range: tuple[int, int], hour, ages: tuple[str, ...], genders: tuple[str, ...],
    categories: tuple[str, ...], status: str, amount_range: tuple[float, float],
) -> dict[str, object]:
    """Filter once and return only compact, chart-ready aggregates."""
    data = load_dashboard_data()
    mask = data["step"].between(*step_range) & data["amount"].between(*amount_range)
    if hour != "All":
        mask &= (data["step"] % 24).eq(hour)
    if ages:
        mask &= data["age"].astype(str).isin(ages)
    if genders:
        mask &= data["gender"].astype(str).isin(genders)
    if categories:
        mask &= data["category"].isin(categories)
    if status != "All":
        mask &= data["fraud"].eq(1 if status == "Fraud" else 0)
    filtered = data.loc[mask]
    summary = fraud_summary(filtered)
    if filtered.empty:
        return {"summary": summary, "histogram": pd.DataFrame(), "time": pd.DataFrame(),
                "category": pd.DataFrame(), "customer": pd.DataFrame(), "merchant": pd.DataFrame()}

    amounts = filtered["amount"].to_numpy()
    edges = np.histogram_bin_edges(amounts, bins=60)
    if len(edges) < 2 or edges[0] == edges[-1]:
        edges = np.array([amounts[0] - .5, amounts[0] + .5])
    hist_parts = []
    for fraud_value, label in ((0, "Non-fraud"), (1, "Fraud")):
        counts, _ = np.histogram(filtered.loc[filtered["fraud"].eq(fraud_value), "amount"], bins=edges)
        hist_parts.append(pd.DataFrame({
            "bin_start": edges[:-1], "bin_end": edges[1:], "bin_center": (edges[:-1] + edges[1:]) / 2,
            "count": counts, "status": label,
        }))
    time = filtered.groupby("step", observed=True).agg(
        transaction_count=("fraud", "size"), fraud_count=("fraud", "sum")
    ).reset_index()
    time["fraud_rate"] = time["fraud_count"] / time["transaction_count"]
    return {
        "summary": summary,
        "histogram": pd.concat(hist_parts, ignore_index=True),
        "time": time,
        "category": grouped_fraud(filtered, "category"),
        "customer": grouped_fraud(filtered, "customer"),
        "merchant": grouped_fraud(filtered, "merchant"),
    }
