import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.cards import BLUE, FRAUD, SAFE, configure_page, kpi_card, page_header, section_rule
from components.charts import empty_chart, fraud_donut, style_figure
from components.filters import fraud_analysis_filters
from components.metrics import build_fraud_analysis
from utils.formatting import compact_money, percent

configure_page("Fraud Analysis")
page_header("Fraud Analysis", "Explore volume, fraud incidence, and monetary exposure. All sidebar filters apply to every view on this page.")
filters, top_n = fraud_analysis_filters()
analysis = build_fraud_analysis(**filters)
summary = analysis["summary"]

cols = st.columns(4)
for col, values in zip(cols, [
    ("Transactions", f"{summary['transactions']:,}", "Filtered population", None),
    ("Fraud Transactions", f"{summary['fraud_count']:,}", "Count, not rate", FRAUD),
    ("Fraud Rate", percent(summary["fraud_rate"]), "Fraud / transactions", FRAUD),
    ("Fraud Amount", compact_money(summary["fraud_amount"]), "Known fraudulent value", BLUE),
]):
    with col: kpi_card(*values)

if summary["transactions"] == 0:
    st.plotly_chart(empty_chart("No transactions match the selected filters."), use_container_width=True)
    st.stop()

section_rule(); st.subheader("Fraud overview")
left, right = st.columns([.8, 1.2])
with left: st.plotly_chart(fraud_donut(summary["fraud_count"], summary["legitimate_count"]), use_container_width=True)
with right:
    histogram = analysis["histogram"]
    fig = px.bar(histogram, x="bin_center", y="count", color="status", barmode="overlay", opacity=.78,
                       color_discrete_map={"Fraud":FRAUD,"Non-fraud":SAFE}, title="Transaction amount distribution",
                       custom_data=["bin_start", "bin_end"])
    fig.update_traces(hovertemplate="Amount: %{customdata[0]:.2f}–%{customdata[1]:.2f}<br>Transactions: %{y:,}<extra></extra>")
    fig.update_layout(xaxis_title="Amount", yaxis_title="Transactions", legend_title="Status")
    st.plotly_chart(style_figure(fig, 360), use_container_width=True)

section_rule(); st.subheader("Daily activity")
time = analysis["time"]
metric = st.radio(
    "Time series metric",
    ["Transaction Volume", "Fraud Rate", "Fraud Amount"],
    index=1,
    horizontal=True,
)
metric_col = {"Transaction Volume":"transaction_count","Fraud Rate":"fraud_rate","Fraud Amount":"fraud_amount"}[metric]
fig = px.line(
    time,
    x="step",
    y=metric_col,
    title=f"{metric} by Simulated Day",
    labels={"step":"Simulated Day", metric_col:metric},
    color_discrete_sequence=[BLUE if metric == "Transaction Volume" else FRAUD],
)
if metric == "Fraud Rate":
    fig.update_traces(hovertemplate="Simulated Day: %{x}<br>Fraud Rate: %{y:.2%}<extra></extra>")
    fig.update_yaxes(tickformat=".2%")
elif metric == "Fraud Amount":
    fig.update_traces(hovertemplate="Simulated Day: %{x}<br>Fraud Amount: %{y:,.2f}<extra></extra>")
    fig.update_yaxes(tickformat=",")
else:
    fig.update_traces(hovertemplate="Simulated Day: %{x}<br>Transactions: %{y:,}<extra></extra>")
    fig.update_yaxes(tickformat=",")
st.plotly_chart(style_figure(fig), use_container_width=True)
st.caption("BankSim contains exactly 40 fraudulent transactions per simulated day, so fraud volume is constant by construction.")

section_rule(); st.subheader("Category")
category = analysis["category"]
cols = st.columns(3)
specs=[("fraud_count","Fraud count by category",FRAUD,","),("fraud_rate","Fraud rate by category",BLUE,".1%"),("transaction_count","Transaction count by category",SAFE,",")]
for col,(field,title,color,tick) in zip(cols,specs):
    with col:
        frame=category.sort_values(field,ascending=True)
        fig=px.bar(frame,x=field,y="category",orientation="h",title=title,color_discrete_sequence=[color])
        fig.update_xaxes(tickformat=tick); st.plotly_chart(style_figure(fig,430,False),use_container_width=True)

section_rule(); st.subheader("Customer")
customer = analysis["customer"]
c1,c2=st.columns(2)
with c1:
    frame=customer.nlargest(top_n,"fraud_count").sort_values("fraud_count")
    fig=px.bar(frame,x="fraud_count",y="customer",orientation="h",title=f"Top {top_n} customers by fraud count",color_discrete_sequence=[FRAUD])
    st.plotly_chart(style_figure(fig,440,False),use_container_width=True)
with c2:
    frame=customer.nlargest(top_n,"fraud_amount").sort_values("fraud_amount")
    fig=px.bar(frame,x="fraud_amount",y="customer",orientation="h",title=f"Top {top_n} customers by fraud amount",color_discrete_sequence=[BLUE])
    st.plotly_chart(style_figure(fig,440,False),use_container_width=True)

section_rule(); st.subheader("Merchant")
merchant = analysis["merchant"]
merchant_metric=st.selectbox("Rank merchants by",["Fraud count","Fraud rate","Transaction volume"])
field={"Fraud count":"fraud_count","Fraud rate":"fraud_rate","Transaction volume":"transaction_count"}[merchant_metric]
frame=merchant.nlargest(top_n,field).sort_values(field)
custom=frame[["transaction_count","fraud_count","fraud_rate"]].to_numpy()
fig=go.Figure(go.Bar(x=frame[field],y=frame.merchant,orientation="h",marker_color=FRAUD if field!="transaction_count" else BLUE,
 customdata=custom,hovertemplate="Merchant: %{y}<br>Transactions: %{customdata[0]:,}<br>Fraud count: %{customdata[1]:,}<br>Fraud rate: %{customdata[2]:.2%}<extra></extra>"))
fig.update_layout(title=f"Top {top_n} merchants by {merchant_metric.lower()}")
fig.update_xaxes(tickformat=".1%" if field=="fraud_rate" else ",")
st.plotly_chart(style_figure(fig,470,False),use_container_width=True)
