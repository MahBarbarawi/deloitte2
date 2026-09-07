"""Shared presentation for one Random Forest assessment."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.cards import kpi_card, output_card, section_rule
from utils.formatting import money, percent
from utils.inference import FraudAssessment


def _ratio(value: float | None) -> str:
    return "Not available" if value is None else f"{value:.2f}×"


def render_decision_evidence(assessment: FraudAssessment) -> None:
    evidence = assessment.model
    st.subheader("Model Decision Evidence")
    columns = st.columns(3)
    with columns[0]:
        output_card("Prediction", "FRAUD" if evidence.prediction else "NON-FRAUD", "Frozen Random Forest", fraud=bool(evidence.prediction))
    with columns[1]:
        output_card("Fraud Probability", percent(evidence.fraud_probability), "Random Forest positive-class probability", fraud=bool(evidence.prediction))
    with columns[2]:
        output_card("Decision Threshold", percent(evidence.decision_threshold), "Frozen operating point")

    margin_points = evidence.decision_margin * 100
    columns = st.columns(4)
    with columns[0]:
        kpi_card("Decision Margin", f"{margin_points:+.2f} pp", "Fraud probability minus threshold")
    with columns[1]:
        kpi_card(
            "Tree Agreement",
            f"{evidence.tree_agreement_count} of {evidence.tree_count}",
            f"{percent(evidence.tree_agreement)} vote for the final class at the same threshold",
        )
    with columns[2]:
        kpi_card("Tree Probability Spread", f"{evidence.tree_probabilities.std:.4f}", "Standard deviation across fitted trees")
    with columns[3]:
        kpi_card(
            "Typical Leaf Training Support",
            f"{evidence.leaf_training_support.median:,.0f}",
            "Median unique in-bag training observations",
        )

    probability = evidence.tree_probabilities
    support = evidence.leaf_training_support
    details = pd.DataFrame([
        {
            "Evidence": "Per-tree fraud probability",
            "Mean": f"{probability.mean:.4f}",
            "Std": f"{probability.std:.4f}",
            "Min": f"{probability.minimum:.4f}",
            "25%": f"{probability.q25:.4f}",
            "Median": f"{probability.median:.4f}",
            "75%": f"{probability.q75:.4f}",
            "Max": f"{probability.maximum:.4f}",
        },
        {
            "Evidence": "Leaf training support",
            "Mean": f"{support.mean:.1f}",
            "Std": f"{support.std:.1f}",
            "Min": f"{support.minimum:.0f}",
            "25%": f"{support.q25:.0f}",
            "Median": f"{support.median:.0f}",
            "75%": f"{support.q75:.0f}",
            "Max": f"{support.maximum:.0f}",
        },
    ])
    st.dataframe(details, hide_index=True, use_container_width=True)
    st.caption(
        "Tree agreement thresholds every tree's fraud probability at the same frozen operating threshold. "
        "Leaf support uses tree_.n_node_samples: the number of unique in-bag training observations reaching "
        "the selected leaf. Weighted support is not used because this model combines bootstrap multiplicity "
        "with class weights, so it is not a literal transaction count."
    )


def render_historical_context(assessment: FraudAssessment) -> None:
    history = assessment.history
    st.subheader("Historical Context")
    st.caption(
        f"Counts and amount summaries come only from the chronological training partition: "
        f"{history.training_row_count:,} transactions on simulated days "
        f"{history.training_step_min}-{history.training_step_max}. Fraud labels are not used."
    )

    st.markdown("**Entity Familiarity**")
    columns = st.columns(3)
    entities = [
        ("Customer Training Transactions", history.entity.customer),
        ("Merchant Training Transactions", history.entity.merchant),
        ("Category Training Transactions", history.entity.category),
    ]
    for column, (label, entity) in zip(columns, entities):
        with column:
            note = "New in post-training data" if entity.is_new else f"Training mean amount {money(entity.amount_mean or 0)}"
            kpi_card(label, f"{entity.transaction_count:,}", note)

    columns = st.columns(2)
    with columns[0]:
        note = "New relationship" if history.entity.customer_merchant_count == 0 else "Training transactions for this pair"
        kpi_card("Customer-Merchant History", f"{history.entity.customer_merchant_count:,}", note)
    with columns[1]:
        note = "New relationship" if history.entity.customer_category_count == 0 else "Training transactions for this pair"
        kpi_card("Customer-Category History", f"{history.entity.customer_category_count:,}", note)

    behavior = history.behavior
    st.markdown("**Behavior Familiarity**")
    if behavior.customer_category_frequency is None:
        st.caption("Category frequency is unavailable because this customer was not observed in training.")
    else:
        st.caption(
            f"{history.category_label} appeared in {history.entity.customer_category_count:,} of "
            f"{history.entity.customer.transaction_count:,} customer training transactions "
            f"({percent(behavior.customer_category_frequency)})."
        )
    columns = st.columns(3)
    with columns[0]:
        kpi_card("Current / Customer Mean", _ratio(behavior.customer_amount_ratio), "Amount comparison")
    with columns[1]:
        kpi_card("Current / Merchant Mean", _ratio(behavior.merchant_amount_ratio), "Amount comparison")
    with columns[2]:
        kpi_card("Current / Category Mean", _ratio(behavior.category_amount_ratio), "Amount comparison")

    st.info(history.similar.reason)


def render_assessment(assessment: FraudAssessment) -> None:
    render_decision_evidence(assessment)
    section_rule()
    render_historical_context(assessment)
