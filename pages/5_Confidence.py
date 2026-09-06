import plotly.express as px
import streamlit as st

from components.cards import AMBER, BLUE, SAFE, concept_card, confidence_scale, configure_page, kpi_card, page_header, section_rule
from components.charts import coefficient_chart, confidence_comparison, style_figure
from utils.data_loader import load_confidence_quality, load_result_csv
from utils.formatting import percent

configure_page("Confidence")
page_header("Prediction Confidence", "A separate fitted model estimates whether the Random Forest's final decision is likely to be correct.", "CONFIDENCE LAYER")

cols=st.columns(3)
with cols[0]:concept_card("Fraud Score","How fraudulent does the transaction look to the classifier?",BLUE)
with cols[1]:concept_card("Prediction Confidence","How likely is the model's final fraud / non-fraud decision to be correct?",SAFE)
with cols[2]:concept_card("Explanation","What factors influenced the fraud model?",AMBER)
st.warning("Fraud probability and prediction confidence are separate outputs. A fraud score must never be labeled as confidence.")

section_rule(); st.subheader("Confidence model quality")
quality=load_confidence_quality()
cols=st.columns(2)
with cols[0]:kpi_card("Brier Score",f"{quality['brier']:.4f}","Lower is better; measures probability calibration")
with cols[1]:kpi_card("Correctness ROC-AUC",f"{quality['correctness_auc']:.4f}","Separates likely correct from likely incorrect decisions")
st.caption("These metrics evaluate the separate Logistic Regression correctness model on exported validation predictions.")

section_rule(); st.subheader("Confidence validation")
confidence_scale()
summary=load_result_csv("confidence_summary.csv")
st.plotly_chart(confidence_comparison(summary),use_container_width=True)
st.dataframe(summary.style.format({"predicted_confidence":"{:.2%}","actual_accuracy":"{:.2%}","average_fraud_probability":"{:.2%}"}),hide_index=True,use_container_width=True)

section_rule(); st.subheader("Confidence by prediction type")
by_prediction=load_result_csv("confidence_by_prediction.csv")
prediction=st.radio("Prediction type",["Fraud","Non-Fraud"],horizontal=True)
frame=by_prediction[by_prediction.prediction_name.eq(prediction)].copy()
order=["Low","Medium","High","Very High"]
frame["confidence_level"] = frame.confidence_level.astype("category").cat.set_categories(order,ordered=True)
frame=frame.sort_values("confidence_level")
fig=px.bar(frame,x="confidence_level",y=["predicted_confidence","actual_accuracy"],barmode="group",title=f"{prediction} decisions: confidence vs observed accuracy",color_discrete_map={"predicted_confidence":BLUE,"actual_accuracy":SAFE})
fig.update_yaxes(tickformat=".0%",range=[max(0,float(frame[["predicted_confidence","actual_accuracy"]].min().min())-.1),1.01])
fig.update_layout(xaxis_title="Confidence level",yaxis_title="Rate",legend_title="")
st.plotly_chart(style_figure(fig),use_container_width=True)
if prediction=="Fraud": st.info("Fraud decisions show a clear accuracy gradient across confidence levels. There are no Very High fraud predictions in this exported validation summary.")
else: st.caption("Very High confidence is dominated by easy non-fraud decisions, so aggregate confidence should always be interpreted alongside prediction type.")

section_rule(); st.subheader("Learned confidence factors")
coefficients=load_result_csv("confidence_coefficients.csv")
st.plotly_chart(coefficient_chart(coefficients),use_container_width=True)
cols=st.columns(2)
with cols[0]:concept_card("Positive coefficient","Associated with increased estimated prediction correctness.",SAFE)
with cols[1]:concept_card("Negative coefficient","Associated with decreased estimated prediction correctness.",BLUE)
st.caption("Coefficients are learned associations in the correctness model; they are not causal effects and are not global fraud-feature importance.")
