import numpy as np
import streamlit as st

from components.assessment import render_assessment
from components.cards import configure_page, kpi_card, output_card, page_header, section_rule
from utils.data_loader import load_validation_filter_options, load_validation_index, load_validation_transaction
from utils.formatting import money
from utils.inference import assess_transaction


configure_page("Transaction Demo")
page_header(
    "Transaction Demo",
    "Inspect one held-out validation transaction using the frozen Random Forest and training-only historical context.",
    "LIVE FROZEN INFERENCE",
)

index_data = load_validation_index()
filter_options = load_validation_filter_options()
with st.sidebar:
    st.subheader("Transaction finder")
    prediction_filter = st.radio("Saved prediction", ["All", "Fraud", "Non-fraud"], horizontal=True)
    label_filter = st.radio("Actual label", ["All", "Fraud", "Non-fraud"], horizontal=True)
    category_filter = st.multiselect("Category", list(filter_options["categories"]), default=[], placeholder="All categories")
    amount_limits = (filter_options["amount_min"], filter_options["amount_max"])
    amount_filter = st.slider("Amount", *amount_limits, amount_limits, format="%.2f")

mask = index_data.amount.between(*amount_filter)
if prediction_filter != "All":
    mask &= index_data.prediction.eq(1 if prediction_filter == "Fraud" else 0)
if label_filter != "All":
    mask &= index_data.actual_fraud.eq(1 if label_filter == "Fraud" else 0)
if category_filter:
    mask &= index_data.category.isin(category_filter)
candidate_ids = index_data.loc[mask, "source_index"]
if candidate_ids.empty:
    st.warning("No validation transactions match these filters.")
    st.stop()

if "demo_source_index" not in st.session_state or not candidate_ids.eq(st.session_state.demo_source_index).any():
    st.session_state.demo_source_index = int(candidate_ids.iloc[0])

lookup_col, button_col = st.columns([4, 1])
with lookup_col:
    exact_id = st.text_input("Open exact transaction identifier", placeholder="Enter a source index")
with button_col:
    st.write("")
    if st.button("Open ID", use_container_width=True):
        try:
            requested = int(exact_id)
        except ValueError:
            st.error("Enter a numeric source index.")
        else:
            if index_data.source_index.eq(requested).any():
                st.session_state.demo_source_index = requested
                st.rerun()
            else:
                st.error("That source index is not in the validation demo.")

selector_col, random_col = st.columns([5, 1])
with random_col:
    st.write("")
    if st.button("Random", use_container_width=True):
        st.session_state.demo_source_index = int(candidate_ids.sample(1).iloc[0])
        st.rerun()
with selector_col:
    options = candidate_ids.head(100).astype(int).tolist()
    if st.session_state.demo_source_index not in options:
        options.insert(0, st.session_state.demo_source_index)
    selected = st.selectbox("Browse matching transactions (first 100)", options, index=options.index(st.session_state.demo_source_index))
    if selected != st.session_state.demo_source_index:
        st.session_state.demo_source_index = selected
        st.rerun()
st.caption(f"{len(candidate_ids):,} transactions match the filters. The browser is capped at 100 IDs; use exact lookup or Random for the rest.")

row = load_validation_transaction(int(st.session_state.demo_source_index))
if row.empty:
    st.error("The selected transaction could not be loaded.")
    st.stop()

with st.spinner("Scoring one row with the frozen Random Forest..."):
    assessment = assess_transaction(row)

st.subheader("Transaction Details")
columns = st.columns([1, 1.55, 1, 1, 1])
details = [
    ("Amount", money(float(row.amount.iloc[0]))),
    ("Category", str(row.category.iloc[0]).replace("es_", "").replace("_", " ").title()),
    ("Age", str(row.age.iloc[0])),
    ("Gender", str(row.gender.iloc[0])),
    ("Simulation Day", str(int(row.step.iloc[0]))),
]
for column, (label, value) in zip(columns, details):
    with column:
        kpi_card(label, value, compact=label == "Category")

section_rule()
render_assessment(assessment)

section_rule()
st.subheader("Ground Truth")
actual = int(row.actual_fraud.iloc[0])
columns = st.columns([1, 2])
with columns[0]:
    output_card("Actual Label", "FRAUD" if actual else "NON-FRAUD", "Held-out validation label", fraud=bool(actual))
with columns[1]:
    if actual == assessment.model.prediction:
        st.success("The frozen model prediction matches the ground-truth label.")
    else:
        st.error("The frozen model prediction does not match the ground-truth label.")
    st.info("The actual label is displayed only after inference and is not used to generate the prediction or historical context.")

saved_prediction = int(row.prediction.iloc[0])
saved_probability = float(row.fraud_probability.iloc[0])
if assessment.model.prediction != saved_prediction or not np.isclose(
    assessment.model.fraud_probability, saved_probability, atol=1e-10
):
    st.warning("Recomputed Random Forest output differs from the exported validation result. Check artifact/library compatibility.")
