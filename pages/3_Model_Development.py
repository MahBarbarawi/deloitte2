import plotly.express as px
import streamlit as st

from components.cards import BLUE, SAFE, concept_card, configure_page, flow_diagram, kpi_card, page_header, section_rule
from components.charts import feature_importance_chart, style_figure
from utils.artifact_loader import load_model_config
from utils.data_loader import load_result_csv

configure_page("Model Development")
page_header("Model Development", "A leakage-aware, chronological workflow produced the frozen model used by this application.", "METHODOLOGY")

st.subheader("Development workflow")
flow_diagram(["Data Cleaning", "EDA", "Historical Feature Engineering", "Leakage Validation", "Missing-History Handling", "Chronological Split", "Preprocessing", "Feature Selection", "Model Comparison", "RF Tuning", "Threshold Optimization", "Final Test", "Decision Evidence"])
st.info("Each step is one simulated day. Historical features use only transactions from earlier simulated days; transactions on the same day never see one another.")

section_rule(); st.subheader("Feature engineering groups")
groups = {
    "Transaction": ["amount and log amount", "simulation chronology for leakage-safe splitting"],
    "Customer": ["historical transaction count", "total, average, standard deviation, min/max spend", "rolling activity", "velocity and recency"],
    "Merchant": ["historical volume and received amount", "average, standard deviation, min/max amount", "rolling activity and velocity"],
    "Category": ["historical volume and amount", "average, standard deviation, min/max amount", "rolling activity and velocity"],
    "Customer ↔ Merchant": ["previous relationship transactions", "average, min/max and standard deviation", "recency", "amount ratio", "transaction and spend share", "new relationship flag"],
    "Customer ↔ Category": ["previous category relationship", "amount profile and recency", "amount ratio", "transaction and spend share", "new category flag"],
}
for title, items in groups.items():
    with st.expander(title):
        st.markdown("\n".join(f"- {item}" for item in items))

section_rule(); st.subheader("Feature selection")
results = load_result_csv("feature_selection_results.csv")
config = load_model_config()
cols = st.columns(4)
for col, values in zip(cols, [
    ("Processed Features", "113", "Before selection"), ("Logistic Selected", "57", "L1 coefficient-based"),
    ("RF Selected", "57", "Tree importance-based"), ("Selected by Both", "27", "Common subset"),
]):
    with col: kpi_card(*values)

left,right=st.columns(2)
with left: concept_card("Logistic Regression selector", "L1 coefficient-based feature selection identifies variables with non-zero linear contribution.")
with right: concept_card("Random Forest selector", "Tree feature importance-based selection preserves the strongest nonlinear signals.", SAFE)

metric = st.selectbox("Compare validation metric", ["val_f1", "val_pr_auc", "val_precision", "val_recall"], format_func=lambda value: value.replace("val_", "Validation ").replace("_", " ").title())
plot = results.copy(); plot["configuration"] = plot["feature_set"].str.replace("_", " ") + " · " + plot["model"]
fig=px.bar(plot,x="configuration",y=metric,color="model",barmode="group",title="Saved feature-set comparison",color_discrete_map={"Random Forest":BLUE,"Logistic Regression":"#8791a3"})
fig.update_yaxes(tickformat=".1%",range=[max(0,plot[metric].min()-.08),min(1,plot[metric].max()+.03)])
st.plotly_chart(style_figure(fig,430),use_container_width=True)
st.success(f"Selected configuration: Random Forest + RF-selected {config['selected_feature_count']} features.")
st.caption("The frozen preprocessing export contains legacy day/hour-derived columns from an incorrect interpretation of step. Neither legacy temporal column was selected by the final Random Forest feature mask; the frozen artifacts remain unchanged pending a reviewed comparison experiment.")

section_rule()
importance = load_result_csv("rf_feature_importance.csv")
st.plotly_chart(feature_importance_chart(importance), use_container_width=True)
st.caption("Global impurity-based importance describes the fitted fraud model. It is not a transaction-level certainty measure and does not imply causality.")
