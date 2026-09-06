import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.cards import BLUE, FRAUD, SAFE, configure_page, concept_card, kpi_card, page_header, section_rule
from components.charts import confusion_matrix, style_figure
from utils.artifact_loader import load_model_config
from utils.data_loader import load_result_csv, load_result_json, load_validation_index
from utils.formatting import percent

configure_page("Model Performance")
page_header("Model Performance", "Chronological train, validation, and final-test evidence for the frozen Random Forest.", "EVALUATION")
config=load_model_config(); metrics=load_result_json("final_metrics.json"); test=metrics["test"]

cols=st.columns(3)
for col, values in zip(cols, [("Final Model",config["model_name"],"Frozen fitted artifact"),("Selected Features",str(config["selected_feature_count"]),"RF-selected"),("Operating Threshold",f"{config['decision_threshold']:.5f}","Frozen decision point")]):
    with col:kpi_card(*values)
cols=st.columns(6)
for col,(label,key) in zip(cols,[("Precision","precision"),("Recall","recall"),("F1","f1"),("F2","f2"),("PR-AUC","pr_auc"),("ROC-AUC","roc_auc")]):
    with col:kpi_card(label,percent(test[key]),"Final test")
st.caption("Accuracy is omitted from the headline metrics because the target is highly imbalanced.")

section_rule(); left,right=st.columns([1.15,.85])
with left: st.plotly_chart(confusion_matrix(test),use_container_width=True)
with right:
    st.subheader("Business interpretation")
    concept_card("Fraud detected",percent(test["recall"]),SAFE)
    st.write("")
    concept_card("Fraud missed",percent(1-test["recall"]),FRAUD)
    false_alarm_rate=test["fp"]/(test["fp"]+test["tn"])
    st.metric("False alarm rate among legitimate transactions",percent(false_alarm_rate,3))

section_rule(); st.subheader("Train vs validation vs test")
rows=[]
for split in ["train","validation","test"]:
    for key,label in [("precision","Precision"),("recall","Recall"),("f1","F1"),("pr_auc","PR-AUC"),("roc_auc","ROC-AUC")]:
        rows.append({"Split":split.title(),"Metric":label,"Value":metrics[split][key]})
frame=pd.DataFrame(rows)
fig=px.bar(frame,x="Metric",y="Value",color="Split",barmode="group",title="Performance remains close from validation to final test",color_discrete_map={"Train":"#8791a3","Validation":BLUE,"Test":SAFE})
fig.update_yaxes(tickformat=".0%",range=[.75,1.01]);st.plotly_chart(style_figure(fig),use_container_width=True)
comparison=load_result_csv("validation_vs_test.csv")
st.dataframe(comparison.style.format({"Validation":"{:.2%}","Test":"{:.2%}","Difference":"{:+.2%}"}),hide_index=True,use_container_width=True)

section_rule(); st.subheader("Threshold analysis")
thresholds=load_result_csv("threshold_results.csv"); rf=thresholds[thresholds.model.eq("Random Forest")].sort_values("threshold")
fig=go.Figure()
for metric,color in [("precision",BLUE),("recall",FRAUD),("f1",SAFE)]: fig.add_scatter(x=rf.threshold,y=rf[metric],mode="lines+markers",name=metric.title(),line=dict(color=color))
fig.add_vline(x=config["decision_threshold"],line_dash="dash",line_color="#172338",annotation_text=f"Frozen threshold {config['decision_threshold']:.4f}")
fig.update_layout(title="Exported Random Forest threshold candidates");fig.update_yaxes(tickformat=".0%")
st.plotly_chart(style_figure(fig),use_container_width=True)
cols=st.columns(2)
with cols[0]:concept_card("Lower threshold","More fraud caught, with more false positives.",FRAUD)
with cols[1]:concept_card("Higher threshold","Fewer false positives, with more fraud missed.",BLUE)
st.caption("The dashed operating point is loaded from model_config.json. The exported threshold table contains comparison candidates rather than a dense curve at every probability.")

section_rule(); st.subheader("Error analysis")
cols=st.columns(2)
with cols[0]:
    kpi_card("False Positives",f"{test['fp']:,}","Legitimate transactions incorrectly flagged as fraud",BLUE)
with cols[1]:
    kpi_card("False Negatives",f"{test['fn']:,}","Fraud transactions missed by the model",FRAUD)
with st.expander("Validation error details"):
    st.caption("Examples are loaded only on request to keep the default page lightweight.")
    if st.button("Load validation error examples"):
      demo=load_validation_index()
      fp=demo[(demo.actual_fraud.eq(0)) & (demo.prediction.eq(1))]
      fn=demo[(demo.actual_fraud.eq(1)) & (demo.prediction.eq(0))]
      tab1,tab2=st.tabs([f"False positives ({len(fp):,})",f"False negatives ({len(fn):,})"])
      show=["source_index","step","customer","merchant","category","amount","fraud_probability","prediction_confidence"]
      with tab1:st.dataframe(fp[show].head(100),hide_index=True,use_container_width=True)
      with tab2:st.dataframe(fn[show].head(100),hide_index=True,use_container_width=True)
      st.caption("False negatives deserve particular attention because each represents missed fraud. Tables show up to 100 exported validation examples.")
