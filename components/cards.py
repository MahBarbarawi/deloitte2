"""Reusable dashboard styling and cards."""
from __future__ import annotations

import html
import streamlit as st

from components.styles import (
    ACCENT as BLUE,
    FRAUD,
    NON_FRAUD as SAFE,
    TEXT_MUTED as MUTED,
    TEXT_PRIMARY as INK,
    WARNING as AMBER,
    apply_dashboard_style,
)


def configure_page(title: str) -> None:
    st.set_page_config(page_title=f"{title} | Fraud Intelligence", page_icon="◈", layout="wide")
    apply_dashboard_style()


def page_header(title: str, subtitle: str, eyebrow: str = "FRAUD INTELLIGENCE") -> None:
    st.markdown(f'<div class="eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, note: str = "", accent: str | None = None, compact: bool = False) -> None:
    color = accent or INK
    value_class = "kpi-value compact" if compact else "kpi-value"
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="{value_class}" style="color:{color}">{html.escape(value)}</div>'
        f'<div class="kpi-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def concept_card(title: str, text: str, accent: str = BLUE) -> None:
    st.markdown(
        f'<div class="concept-card" style="border-top:3px solid {accent}">'
        f'<div class="concept-title">{html.escape(title)}</div>'
        f'<div class="concept-text">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def output_card(label: str, value: str, note: str = "", fraud: bool | None = None) -> None:
    color = FRAUD if fraud is True else SAFE if fraud is False else BLUE
    st.markdown(
        f'<div class="output-card" style="border-top:4px solid {color}">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value" style="color:{color};font-size:2rem">{html.escape(value)}</div>'
        f'<div class="kpi-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def flow_diagram(steps: list[str]) -> None:
    items = []
    for index, step in enumerate(steps):
        items.append(f'<div class="flow-node">{html.escape(step)}</div>')
        if index < len(steps) - 1:
            items.append('<div class="flow-arrow">→</div>')
    st.markdown(f'<div class="flow">{"".join(items)}</div>', unsafe_allow_html=True)


def section_rule() -> None:
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
