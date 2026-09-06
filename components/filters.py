"""Global page filter controls."""
from __future__ import annotations

import streamlit as st

from utils.data_loader import load_filter_options


def fraud_analysis_filters() -> tuple[dict, int]:
    options = load_filter_options()
    with st.sidebar:
        st.subheader("Page filters")
        step_min, step_max = options["step_min"], options["step_max"]
        step_range = st.slider("Step / day range", step_min, step_max, (step_min, step_max))
        hour_options = ["All"] + list(range(24))
        hour = st.selectbox("Hour", hour_options)
        ages = list(options["ages"])
        selected_ages = st.multiselect("Age group", ages, default=[], placeholder="All age groups")
        genders = list(options["genders"])
        selected_genders = st.multiselect("Gender", genders, default=[], placeholder="All genders")
        categories = list(options["categories"])
        selected_categories = st.multiselect("Category", categories, default=[], placeholder="All categories")
        status = st.radio("Fraud status", ["All", "Fraud", "Non-fraud"], horizontal=True)
        amount_min, amount_max = options["amount_min"], options["amount_max"]
        amount_range = st.slider("Amount range", amount_min, amount_max, (amount_min, amount_max), format="%.2f")
        top_n = st.slider("Top customers / merchants", 5, 30, 12)

    return {
        "step_range": tuple(step_range), "hour": hour,
        "ages": tuple(selected_ages), "genders": tuple(selected_genders),
        "categories": tuple(selected_categories), "status": status,
        "amount_range": tuple(amount_range),
    }, top_n
