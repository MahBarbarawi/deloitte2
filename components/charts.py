"""Plotly chart builders with a restrained visual language."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.cards import BLUE, FRAUD, MUTED, SAFE
from components.styles import BORDER, CARD_BG, PAGE_BG, PURPLE, TEXT_PRIMARY, TEXT_SECONDARY, WARNING

COLORWAY = [BLUE, FRAUD, SAFE, PURPLE, WARNING, "#70B7C7"]


def style_figure(fig, height: int = 390, legend: bool = True):
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=28, r=22, t=88, b=30),
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(family="Inter, Arial, sans-serif", color=TEXT_PRIMARY),
        title_font=dict(color=TEXT_PRIMARY, size=17), title=dict(y=.98, x=.02, xanchor="left"),
        colorway=COLORWAY, legend=dict(orientation="h", yanchor="bottom", y=1.10, x=0,
            bgcolor=CARD_BG, bordercolor=BORDER, borderwidth=1, font=dict(color=TEXT_PRIMARY)),
        showlegend=legend, hoverlabel=dict(bgcolor="#F5F8FC", bordercolor="#24344D", font=dict(color="#0B1220")),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont=dict(color=TEXT_SECONDARY), title_font=dict(color=TEXT_PRIMARY))
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER, zeroline=False, tickfont=dict(color=TEXT_SECONDARY), title_font=dict(color=TEXT_PRIMARY))
    return fig


def fraud_donut(fraud_count: int, legitimate_count: int):
    fig = go.Figure(go.Pie(labels=["Fraud", "Non-fraud"], values=[fraud_count, legitimate_count], hole=.68,
                           marker_colors=[FRAUD, SAFE], textinfo="percent", hovertemplate="%{label}<br>%{value:,}<br>%{percent}<extra></extra>"))
    fig.update_layout(title="Fraud composition", annotations=[dict(text=f"{fraud_count:,}<br><span style='font-size:11px'>fraud</span>", x=.5, y=.5, showarrow=False)])
    return style_figure(fig, 360)


def confusion_matrix(metrics: dict):
    z = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    labels = [["Legitimate correctly classified", "False alarm"], ["Fraud missed", "Fraud correctly detected"]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted non-fraud", "Predicted fraud"], y=["Actual non-fraud", "Actual fraud"],
        colorscale=[[0, "#24344D"], [.5, "#35689F"], [1, BLUE]], showscale=False,
        hovertemplate="%{y}<br>%{x}<br>Transactions: %{z:,}<extra></extra>"))
    annotations=[]
    for row in range(2):
        for col in range(2):
            annotations.append(dict(x=col,y=row,text=f"<b>{z[row][col]:,}</b><br>{labels[row][col]}",showarrow=False,font=dict(color=TEXT_PRIMARY,size=12)))
    fig.update_layout(title="Final test confusion matrix", annotations=annotations)
    return style_figure(fig, 420, False)


def coefficient_chart(data: pd.DataFrame):
    frame = data.sort_values("coefficient")
    fig = px.bar(frame, x="coefficient", y="feature", orientation="h", color="coefficient",
                 color_continuous_scale=[[0, FRAUD], [.5, "#52637A"], [1, SAFE]],
                 title="Learned confidence factors")
    fig.update_layout(coloraxis_showscale=False)
    fig.add_vline(x=0, line_color=TEXT_SECONDARY)
    return style_figure(fig, 470, False)


def feature_importance_chart(data: pd.DataFrame, top_n: int = 20):
    frame = data.nlargest(top_n, "importance").sort_values("importance")
    frame = frame.assign(feature=frame["feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False))
    fig = px.bar(frame, x="importance", y="feature", orientation="h", title="Global Random Forest Feature Importance", color_discrete_sequence=[BLUE])
    fig.update_xaxes(tickformat=".1%", title="Mean decrease in impurity")
    return style_figure(fig, 580, False)


def confidence_comparison(data: pd.DataFrame):
    order = ["Low", "Medium", "High", "Very High"]
    frame = data.copy(); frame["confidence_level"] = pd.Categorical(frame["confidence_level"], order, ordered=True)
    frame = frame.sort_values("confidence_level")
    fig = go.Figure()
    fig.add_bar(x=frame["confidence_level"], y=frame["predicted_confidence"], name="Average predicted confidence", marker_color=BLUE)
    fig.add_bar(x=frame["confidence_level"], y=frame["actual_accuracy"], name="Observed accuracy", marker_color=SAFE)
    fig.update_layout(title="Predicted confidence vs observed accuracy", barmode="group")
    fig.update_yaxes(tickformat=".0%", range=[0.6, 1.01])
    return style_figure(fig)


def empty_chart(message: str):
    fig = go.Figure(); fig.add_annotation(text=message, x=.5, y=.5, showarrow=False, font=dict(color=MUTED))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return style_figure(fig, 300, False)
