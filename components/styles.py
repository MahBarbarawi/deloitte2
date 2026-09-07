"""Single source of truth for the dashboard's high-contrast dark theme."""
from __future__ import annotations

import streamlit as st

PAGE_BG = "#0B1220"
CARD_BG = "#131F33"
CARD_BG_SECONDARY = "#18263C"
TEXT_PRIMARY = "#F3F7FC"
TEXT_SECONDARY = "#C5D0DF"
TEXT_MUTED = "#91A0B5"
BORDER = "#2B3B53"
ACCENT = "#69A7FF"
POSITIVE = "#45C49A"
WARNING = "#F4B860"
FRAUD = "#FF6678"
NON_FRAUD = POSITIVE
PURPLE = "#B397F2"

def apply_dashboard_style() -> None:
    """Apply explicit component styling on top of the Streamlit theme."""
    st.markdown(
        f"""
        <style>
        :root {{
          --page-bg: {PAGE_BG}; --card-bg: {CARD_BG}; --card-bg-2: {CARD_BG_SECONDARY};
          --text-1: {TEXT_PRIMARY}; --text-2: {TEXT_SECONDARY}; --text-muted: {TEXT_MUTED};
          --border: {BORDER}; --accent: {ACCENT}; --positive: {POSITIVE};
          --warning: {WARNING}; --fraud: {FRAUD};
        }}
        .stApp, [data-testid="stAppViewContainer"] {{ background: var(--page-bg); color: var(--text-1); overflow-x: hidden; }}
        .block-container {{ max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }}
        h1, h2, h3, h4, h5, h6, p, label {{ color: var(--text-1) !important; }}
        h1, h2, h3 {{ letter-spacing: -0.02em; }} h1 {{ font-size: 2.35rem !important; }}
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
        .stCaption, small {{ color: var(--text-muted) !important; opacity: 1 !important; }}
        [data-testid="stSidebar"] {{ background: #0D1728; border-right: 1px solid var(--border); }}
        [data-testid="stSidebar"] * {{ color: var(--text-1) !important; opacity: 1 !important; }}
        [data-testid="stSidebarNav"] a[aria-current="page"] {{ background: var(--card-bg-2); border-left: 3px solid var(--accent); }}
        [data-baseweb="select"] > div, [data-baseweb="input"] > div,
        [data-testid="stNumberInputContainer"] input, [data-testid="stTextInput"] input {{
          background: var(--card-bg-2) !important; color: var(--text-1) !important;
          border-color: var(--border) !important;
        }}
        [data-baseweb="popover"], [role="listbox"], [role="option"] {{
          background: var(--card-bg-2) !important; color: var(--text-1) !important;
        }}
        [data-baseweb="tag"] {{ background: #24446C !important; color: var(--text-1) !important; }}
        [data-testid="stSlider"] [role="slider"] {{ background: var(--accent) !important; }}
        [data-testid="stMetric"] {{ background: var(--card-bg); border: 1px solid var(--border);
          border-radius: 12px; padding: .9rem 1rem; }}
        [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * {{ color: var(--text-1) !important; opacity: 1 !important; }}
        [data-testid="stExpander"] {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; }}
        [data-testid="stExpander"] summary * {{ color: var(--text-1) !important; }}
        [data-testid="stTabs"] button {{ color: var(--text-secondary) !important; }}
        [data-testid="stTabs"] button[aria-selected="true"] {{ color: var(--text-1) !important; border-bottom-color: var(--accent) !important; }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{ border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
        [data-testid="stDataFrame"] iframe {{ color-scheme: dark; }}
        .eyebrow {{ color: var(--accent); text-transform: uppercase; letter-spacing: .12em;
          font-weight: 750; font-size: .75rem; margin-bottom: .35rem; }}
        .subtitle {{ color: var(--text-secondary); font-size: 1.05rem; max-width: 800px; margin: -.4rem 0 1.6rem; }}
        .kpi-card, .concept-card, .output-card {{ background: var(--card-bg); border: 1px solid var(--border);
          border-radius: 14px; padding: 1rem 1.05rem; box-shadow: 0 4px 12px rgba(0,0,0,.16); height: 100%; }}
        .kpi-label {{ color: var(--text-secondary); font-size: .82rem; font-weight: 650; margin-bottom: .4rem; }}
        .kpi-value {{ color: var(--text-1); font-size: 1.4rem; font-weight: 760; line-height: 1.16; }}
        .kpi-value.compact {{ font-size: .95rem; word-break: normal; overflow-wrap: normal; hyphens: none; }}
        .kpi-note {{ color: var(--text-muted); font-size: .76rem; margin-top: .42rem; }}
        .concept-title {{ color: var(--text-1); font-weight: 750; margin-bottom: .45rem; }}
        .concept-text {{ color: var(--text-secondary); font-size: .9rem; line-height: 1.45; }}
        .section-rule {{ border: 0; border-top: 1px solid var(--border); margin: 1.7rem 0; }}
        .flow {{ display:flex; align-items:center; gap:.45rem; flex-wrap:wrap; margin: 1rem 0; }}
        .flow-node {{ background:var(--card-bg); border:1px solid var(--border); border-radius:10px;
          padding:.65rem .85rem; font-size:.82rem; font-weight:650; color:var(--text-1); }}
        .flow-arrow {{ color:var(--text-muted); font-size:1rem; }}
        div[data-testid="stPlotlyChart"] {{ background:var(--card-bg); border:1px solid var(--border); border-radius:14px; padding:.3rem; }}
        [data-testid="stAlert"] {{ color: var(--text-1); border: 1px solid var(--border); }}
        button[kind="primary"], button[kind="secondary"] {{ border-color: var(--border) !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
