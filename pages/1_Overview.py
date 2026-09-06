import streamlit as st

from components.cards import BLUE, FRAUD, concept_card, configure_page, flow_diagram, kpi_card, page_header, section_rule
from utils.artifact_loader import load_model_config
from utils.data_loader import load_dashboard_summary, load_result_json
from utils.formatting import percent

configure_page("Overview")
page_header("Fraud Detection & Risk Intelligence", "Transaction-level fraud analysis, model evaluation, and confidence-aware decision support.", "EXECUTIVE OVERVIEW")

summary = load_dashboard_summary()
config = load_model_config()
test = load_result_json("final_metrics.json")["test"]

cols = st.columns(5)
values = [
    ("Transactions", f"{summary['transactions']:,}", "Analyzed records", None),
    ("Fraud Transactions", f"{summary['fraud_count']:,}", "Known fraud labels", FRAUD),
    ("Fraud Rate", percent(summary["fraud_rate"]), "Highly imbalanced target", FRAUD),
    ("Final Model", config["model_name"], "Frozen classifier", BLUE),
    ("Selected Features", str(config["selected_feature_count"]), "From 113 processed features", BLUE),
]
for col, item in zip(cols, values):
    with col: kpi_card(*item)

section_rule()
st.subheader("Final test performance")
cols = st.columns(4)
for col, (label, key, note) in zip(cols, [
    ("Precision", "precision", "Reliability of fraud flags"), ("Recall", "recall", "Share of fraud detected"),
    ("F1", "f1", "Precision–recall balance"), ("PR-AUC", "pr_auc", "Ranking quality under imbalance"),
]):
    with col: kpi_card(label, percent(test[key]), note)

section_rule()
st.subheader("Decision pipeline")
flow_diagram(["Transaction", "Behavioral Feature Engineering", "Preprocessing", "Feature Selection", "Random Forest", "Fraud Score", "Decision Threshold", "Fraud / Non-Fraud", "Confidence Layer"])

st.subheader("How to read the metrics")
cols = st.columns(2)
with cols[0]: concept_card("Precision", "When the system flags fraud, how often is it actually fraud?")
with cols[1]: concept_card("Recall", "Of all actual fraud transactions, how many did the system catch?", FRAUD)
st.caption("Accuracy is intentionally de-emphasized because non-fraud transactions dominate the dataset.")
