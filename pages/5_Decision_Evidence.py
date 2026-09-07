import streamlit as st

from components.assessment import render_assessment
from components.cards import BLUE, SAFE, concept_card, configure_page, page_header, section_rule
from utils.data_loader import load_validation_index, load_validation_transaction
from utils.inference import assess_transaction


configure_page("Decision Evidence")
page_header(
    "Decision Evidence",
    "Transparent evidence from the frozen Random Forest, kept separate from training-data familiarity.",
    "INTERPRETABLE ASSESSMENT",
)

columns = st.columns(2)
with columns[0]:
    concept_card(
        "Model Decision Evidence",
        "Probability, threshold distance, tree votes, tree dispersion, and leaf support come directly from the fitted Random Forest.",
        BLUE,
    )
with columns[1]:
    concept_card(
        "Historical Context",
        "Raw entity and relationship counts come only from the chronological training partition. They do not change the prediction.",
        SAFE,
    )
st.info(
    "No single probability of prediction correctness is estimated. Each evidence value below retains its literal meaning."
)

section_rule()
st.subheader("Choose a held-out validation transaction")
index_data = load_validation_index()
default_id = int(index_data.source_index.iloc[0])
source_id = st.number_input(
    "Source index",
    min_value=int(index_data.source_index.min()),
    max_value=int(index_data.source_index.max()),
    value=default_id,
    step=1,
)
row = load_validation_transaction(int(source_id))
if row.empty:
    st.warning("That source index is not part of the validation export. Choose another validation transaction.")
    st.stop()

with st.spinner("Calculating evidence from the frozen Random Forest..."):
    assessment = assess_transaction(row)
st.caption(
    f"Transaction {int(source_id)} · simulated day {int(row.step.iloc[0])} · "
    f"{str(row.category.iloc[0]).replace('es_', '').replace('_', ' ').title()}"
)

section_rule()
render_assessment(assessment)
