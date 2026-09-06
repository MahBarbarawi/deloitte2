import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components.cards import BLUE, FRAUD, SAFE, configure_page, kpi_card, output_card, page_header, section_rule
from components.charts import style_figure
from utils.artifact_loader import load_model_config
from utils.data_loader import load_validation_filter_options, load_validation_index, load_validation_transaction
from utils.formatting import money, percent
from utils.inference import predict_transactions

configure_page("Transaction Demo")
page_header("Transaction Demo", "Run one exported validation transaction through the saved preprocessing, Random Forest, threshold, and confidence model.", "LIVE FROZEN INFERENCE")

index_data=load_validation_index(); filter_options=load_validation_filter_options(); config=load_model_config()
with st.sidebar:
    st.subheader("Transaction finder")
    prediction_filter=st.radio("Saved prediction",["All","Fraud","Non-fraud"],horizontal=True)
    label_filter=st.radio("Actual label",["All","Fraud","Non-fraud"],horizontal=True)
    categories=list(filter_options["categories"])
    category_filter=st.multiselect("Category",categories,default=[] ,placeholder="All categories")
    amount_limits=(filter_options["amount_min"],filter_options["amount_max"])
    amount_filter=st.slider("Amount",*amount_limits,amount_limits,format="%.2f")

mask=index_data.amount.between(*amount_filter)
if prediction_filter!="All":mask &= index_data.prediction.eq(1 if prediction_filter=="Fraud" else 0)
if label_filter!="All":mask &= index_data.actual_fraud.eq(1 if label_filter=="Fraud" else 0)
if category_filter:mask &= index_data.category.isin(category_filter)
candidate_ids=index_data.loc[mask,"source_index"]
if candidate_ids.empty:
    st.warning("No validation transactions match these filters.");st.stop()

if "demo_source_index" not in st.session_state or not candidate_ids.eq(st.session_state.demo_source_index).any():
    st.session_state.demo_source_index=int(candidate_ids.iloc[0])

lookup_col,button_col=st.columns([4,1])
with lookup_col:
    exact_id=st.text_input("Open exact transaction identifier",placeholder="Enter a source index")
with button_col:
    st.write("")
    if st.button("Open ID",use_container_width=True):
        try: requested=int(exact_id)
        except ValueError: st.error("Enter a numeric source index.")
        else:
            if index_data.source_index.eq(requested).any(): st.session_state.demo_source_index=requested;st.rerun()
            else: st.error("That source index is not in the validation demo.")

selector_col,random_col=st.columns([5,1])
with random_col:
    st.write("")
    if st.button("Random",use_container_width=True):
        st.session_state.demo_source_index=int(candidate_ids.sample(1).iloc[0]);st.rerun()
with selector_col:
    options=candidate_ids.head(100).astype(int).tolist()
    if st.session_state.demo_source_index not in options: options.insert(0,st.session_state.demo_source_index)
    selected=st.selectbox("Browse matching transactions (first 100)",options,index=options.index(st.session_state.demo_source_index))
    if selected != st.session_state.demo_source_index: st.session_state.demo_source_index=selected;st.rerun()
st.caption(f"{len(candidate_ids):,} transactions match the filters. The browser is capped at 100 IDs; use exact lookup or Random for the rest.")

row=load_validation_transaction(int(st.session_state.demo_source_index))
if row.empty: st.error("The selected transaction could not be loaded.");st.stop()

run_col,note_col=st.columns([1,3])
with run_col:
    run_live=st.button("Run live inference",type="primary",use_container_width=True)
with note_col:
    st.caption("Browsing uses exported scores. Live inference loads fitted artifacts and scores only this selected row.")
if run_live:
    with st.spinner("Scoring one row with frozen artifacts..."):
        st.session_state.live_inference=predict_transactions(row)
        st.session_state.live_inference_source=int(st.session_state.demo_source_index)

is_live=st.session_state.get("live_inference_source")==int(st.session_state.demo_source_index)
inference=(st.session_state.live_inference.iloc[0] if is_live else row.iloc[0])
prediction=int(inference.prediction); probability=float(inference.fraud_probability); confidence=float(inference.prediction_confidence)

st.subheader("Transaction details")
cols=st.columns([1,1.55,1,1,1,1])
details=[("Amount",money(float(row.amount.iloc[0]))),("Category",str(row.category.iloc[0]).replace("es_","").replace("_"," ").title()),("Age",str(row.age.iloc[0])),("Gender",str(row.gender.iloc[0])),("Day",str(int(row.day.iloc[0]))),("Hour",f"{int(row.hour_of_day.iloc[0]):02d}:00")]
for col,(label,value) in zip(cols,details):
    with col:kpi_card(label,value,compact=label == "Category")

section_rule();st.subheader("Behavioral context")
cols=st.columns(5)
history=[("Customer previous",row.customer_previous_transaction_count.iloc[0]),("Merchant previous",row.merchant_previous_transaction_count.iloc[0]),("Category previous",row.category_previous_transaction_count.iloc[0]),("Customer ↔ Merchant",row.customer_merchant_previous_transaction_count.iloc[0]),("Customer ↔ Category",row.customer_category_previous_transaction_count.iloc[0])]
for col,(label,value) in zip(cols,history):
    with col:kpi_card(label,f"{int(value):,}","Prior transactions only")
cols=st.columns(3)
for col,(label,field) in zip(cols,[("Customer amount ratio","customer_amount_ratio"),("Merchant amount ratio","merchant_amount_ratio"),("Category amount ratio","category_amount_ratio")]):
    with col:kpi_card(label,f"{float(row[field].iloc[0]):.2f}×","Current / historical average")

section_rule();st.subheader("Model outputs")
cols=st.columns(3)
source_note="Live frozen inference" if is_live else "Exported validation result"
with cols[0]:output_card("Prediction","FRAUD" if prediction else "NON-FRAUD",source_note,fraud=bool(prediction))
with cols[1]:output_card("Fraud Score",percent(probability),f"Operating threshold {percent(config['decision_threshold'])}",fraud=probability>=config['decision_threshold'])
with cols[2]:output_card("Prediction Confidence",percent(confidence),str(inference.confidence_level),fraud=None)
st.caption("Fraud Score measures fraud likelihood from the Random Forest. Prediction Confidence estimates the probability that the final class decision is correct. " + ("These values were recomputed live for one row." if is_live else "Click Run live inference to recompute this row."))

section_rule();st.subheader("Confidence evidence")
evidence={
    "Decision margin":float(inference.decision_margin),"Entropy certainty":float(inference.entropy_certainty),
    "Tree agreement":float(inference.tree_agreement),"Tree probability std":float(inference.tree_probability_std),
}
left,right=st.columns([1.4,.6])
with left:
    names=list(evidence);values=[evidence[name] for name in names]
    fig=go.Figure(go.Bar(x=values[:3],y=names[:3],orientation="h",marker_color=[BLUE,SAFE,"#8b6cae"],text=[percent(v) for v in values[:3]],textposition="inside"))
    fig.update_xaxes(range=[0,1],tickformat=".0%",title="Normalized evidence")
    fig.update_layout(title="Certainty and agreement signals")
    st.plotly_chart(style_figure(fig,300,False),use_container_width=True)
with right:
    kpi_card("Tree probability dispersion",f"{evidence['Tree probability std']:.4f}","Lower values mean less tree-to-tree variation")
    supports=[float(inference[f"support__{field}"]) for field in ["customer_previous_transaction_count","merchant_previous_transaction_count","category_previous_transaction_count","customer_merchant_previous_transaction_count","customer_category_previous_transaction_count"]]
    st.metric("Average historical support",percent(float(np.mean(supports))))

with st.expander("Historical support detail"):
    labels=["Customer","Merchant","Category","Customer ↔ Merchant","Customer ↔ Category"]
    support_cols=st.columns(5)
    for col,label,value in zip(support_cols,labels,supports):
        with col:st.metric(label,percent(value))
    st.caption("Each count is log1p-normalized against its exported 95th-percentile reference and clipped to [0, 1], exactly as in the training notebook.")

section_rule();st.subheader("Ground Truth")
actual=int(row.actual_fraud.iloc[0])
cols=st.columns([1,2])
with cols[0]:output_card("Actual Label","FRAUD" if actual else "NON-FRAUD","Held-out validation label",fraud=bool(actual))
with cols[1]:
    correct=actual==prediction
    if correct:
        st.success("The frozen model prediction matches the ground-truth label.")
    else:
        st.error("The frozen model prediction does not match the ground-truth label.")
    st.info("The actual label is displayed only after inference. It is excluded from the 89 raw model inputs and was not used to generate this prediction.")

saved_probability=float(row.fraud_probability.iloc[0]);saved_confidence=float(row.prediction_confidence.iloc[0])
if not (np.isclose(probability,saved_probability,atol=1e-10) and np.isclose(confidence,saved_confidence,atol=1e-10)):
    st.warning("Recomputed outputs differ slightly from the exported demo values. Check artifact/library compatibility.")
