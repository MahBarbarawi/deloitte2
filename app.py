"""Entry point for the frozen-model fraud intelligence dashboard."""
import streamlit as st

from components.cards import BLUE, concept_card, configure_page, flow_diagram, page_header
from utils.artifact_loader import load_model_config

configure_page("Home")
page_header(
    "Fraud Detection & Risk Intelligence",
    "Transaction-level fraud analysis, model evaluation, and interpretable decision evidence.",
)
st.info("Use the sidebar to explore fraud patterns, review model development and performance, inspect decision evidence, or score a validation transaction.")

cols = st.columns(3)
with cols[0]:
    concept_card("Analyze", "Explore fraud volume, rates, amounts, customers, merchants, categories, and time patterns.", BLUE)
with cols[1]:
    concept_card("Evaluate", "Review the frozen Random Forest, chronological evaluation, operating threshold, and error profile.", "#2f7d68")
with cols[2]:
    concept_card("Decide", "Inspect threshold distance, tree behavior, leaf support, and training-only historical context.", "#b7791f")

st.subheader("Frozen inference architecture")
flow_diagram(["Raw transaction", "Saved preprocessor", "57 selected features", "Random Forest", "Frozen threshold", "Decision evidence"])
config = load_model_config()
st.caption(f"{config['model_name']} · {config['selected_feature_count']} selected features · threshold {config['decision_threshold']:.6f} · no model training occurs in this app")
