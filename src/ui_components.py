"""Reusable Streamlit presentation helpers for the research dashboard."""

from __future__ import annotations

from html import escape
from typing import Any, Iterable, Mapping, Optional

import pandas as pd
import streamlit as st


def inject_premium_css() -> None:
    """Inject the shared dark-mode-friendly command-center styling."""
    st.markdown(
        """
        <style>
        :root {
            --ui-bg: #101214;
            --ui-panel: #191c1f;
            --ui-panel-2: #202428;
            --ui-border: #343a40;
            --ui-text: #f1f3f5;
            --ui-muted: #a9b0b7;
            --ui-accent: #55b8d0;
            --ui-positive: #57bd86;
            --ui-warning: #d9ad55;
            --ui-critical: #df6b74;
        }

        .stApp { background: var(--ui-bg); color: var(--ui-text); }
        .block-container { max-width: 1480px; padding-top: 1.6rem; padding-bottom: 3rem; }
        section[data-testid="stSidebar"] { background: #141617; border-right: 1px solid var(--ui-border); }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--ui-muted); }
        section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.2rem; }
        section[data-testid="stSidebar"] label { border-radius: 6px; padding: 0.34rem 0.42rem; transition: background 120ms ease; }
        section[data-testid="stSidebar"] label:hover { background: #202326; }
        section[data-testid="stSidebar"] label:has(input:checked) { background: #252a2e; border: 1px solid #3d464d; }

        .main-header { color: var(--ui-text); font-size: 2rem; font-weight: 720; letter-spacing: 0; margin: 0; }
        .sub-header { color: var(--ui-muted); font-size: 0.98rem; letter-spacing: 0; margin: 0.25rem 0 1rem; }
        .ui-section { margin: 1.65rem 0 0.75rem; border-bottom: 1px solid var(--ui-border); padding-bottom: 0.55rem; }
        .ui-section h2 { color: var(--ui-text); font-size: 1.15rem; margin: 0; letter-spacing: 0; }
        .ui-section p { color: var(--ui-muted); margin: 0.3rem 0 0; font-size: 0.88rem; }

        .ui-status-card { min-height: 116px; background: var(--ui-panel); border: 1px solid var(--ui-border); border-radius: 8px; padding: 0.9rem 1rem; margin-bottom: 0.75rem; }
        .ui-status-card .title { color: var(--ui-muted); font-size: 0.76rem; font-weight: 650; text-transform: uppercase; letter-spacing: 0; }
        .ui-status-card .value { color: var(--ui-text); font-size: 1.35rem; font-weight: 720; line-height: 1.2; margin-top: 0.45rem; overflow-wrap: anywhere; }
        .ui-status-card .subtitle { color: var(--ui-muted); font-size: 0.78rem; margin-top: 0.45rem; line-height: 1.35; }
        .ui-status-card.positive { border-top: 3px solid var(--ui-positive); }
        .ui-status-card.warning { border-top: 3px solid var(--ui-warning); }
        .ui-status-card.critical { border-top: 3px solid var(--ui-critical); }
        .ui-status-card.info { border-top: 3px solid var(--ui-accent); }
        .ui-status-card.neutral { border-top: 3px solid #718096; }

        .capital-banner { border: 1px solid #7d343b; border-left: 4px solid var(--ui-critical); background: #24171b; border-radius: 6px; padding: 0.85rem 1rem; margin: 0.75rem 0 1rem; }
        .capital-banner strong { color: #ff9aa2; }
        .capital-banner span { color: #d8b9bd; font-size: 0.86rem; margin-left: 0.45rem; }

        .pipeline-map { display: flex; align-items: center; gap: 0.42rem; flex-wrap: wrap; padding: 0.9rem 0 0.35rem; }
        .pipeline-step { background: var(--ui-panel); border: 1px solid var(--ui-border); border-radius: 6px; color: var(--ui-muted); padding: 0.52rem 0.72rem; font-size: 0.78rem; white-space: nowrap; }
        .pipeline-step.complete { color: #b8e6cd; border-color: #397659; background: #17231d; }
        .pipeline-step.active { color: #d9f5fb; border-color: #3e8292; background: #172226; font-weight: 700; }
        .pipeline-arrow { color: var(--ui-accent); font-weight: 700; }

        div[data-testid="stMetric"] { background: var(--ui-panel); border: 1px solid var(--ui-border); border-radius: 8px; padding: 0.8rem 0.9rem; }
        div[data-testid="stMetricValue"] { color: var(--ui-text); font-size: 1.45rem; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--ui-border); border-radius: 6px; overflow: hidden; }
        .stButton > button, .stDownloadButton > button { border-radius: 6px; border-color: #3b4a5c; min-height: 2.35rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.35rem; border-bottom: 1px solid var(--ui-border); }
        .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; padding: 0.55rem 0.75rem; }
        div[data-testid="stExpander"] { background: var(--ui-panel); border-color: var(--ui-border); border-radius: 8px; }
        div[data-testid="stAlert"] { border-radius: 8px; }

        @media (max-width: 720px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; }
            .main-header { font-size: 1.55rem; }
            .ui-status-card { min-height: 104px; }
            .pipeline-map { gap: 0.3rem; }
            .pipeline-step { white-space: normal; flex: 1 1 42%; text-align: center; }
            .pipeline-arrow { display: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(title: str, value: Any, subtitle: str = "", status: str = "neutral") -> None:
    status_name = status if status in {"neutral", "positive", "warning", "critical", "info"} else "neutral"
    st.markdown(
        f"""
        <div class="ui-status-card {status_name}">
            <div class="title">{escape(str(title))}</div>
            <div class="value">{escape(str(value))}</div>
            <div class="subtitle">{escape(str(subtitle))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(cards: Iterable[Any]) -> None:
    card_list = list(cards)
    if not card_list:
        return
    for row_start in range(0, len(card_list), 4):
        row_cards = card_list[row_start:row_start + 4]
        columns = st.columns(len(row_cards))
        for column, card in zip(columns, row_cards):
            if isinstance(card, Mapping):
                title = card.get("title", "")
                value = card.get("value", "")
                subtitle = card.get("subtitle", "")
                status = card.get("status", "neutral")
            else:
                values = list(card) if isinstance(card, (tuple, list)) else [str(card)]
                title = values[0] if values else ""
                value = values[1] if len(values) > 1 else ""
                subtitle = values[2] if len(values) > 2 else ""
                status = values[3] if len(values) > 3 else "neutral"
            with column:
                render_status_card(str(title), value, str(subtitle), str(status))


def render_section_header(title: str, subtitle: str = "") -> None:
    subtitle_html = f"<p>{escape(str(subtitle))}</p>" if subtitle else ""
    st.markdown(
        f'<div class="ui-section"><h2>{escape(str(title))}</h2>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def render_research_disclaimer() -> None:
    st.info("Research-only decision support. Outputs are evidence diagnostics, not execution instructions or financial advice.")


def render_blocked_capital_banner() -> None:
    st.markdown(
        '<div class="capital-banner"><strong>Real capital status: Blocked</strong><span>Only paper tracking and research review are permitted.</span></div>',
        unsafe_allow_html=True,
    )


def render_pipeline_stepper(steps: Iterable[Any], active_step: Any = None) -> None:
    """Render a compact pipeline with optional completed/active states."""
    labels = [str(step) for step in steps]
    if not labels:
        return
    active_index = None
    if active_step is not None:
        if isinstance(active_step, int) and 0 <= active_step < len(labels):
            active_index = active_step
        elif str(active_step) in labels:
            active_index = labels.index(str(active_step))

    parts = []
    for index, label in enumerate(labels):
        state = ""
        if active_index is not None:
            if index < active_index:
                state = " complete"
            elif index == active_index:
                state = " active"
        parts.append(f'<span class="pipeline-step{state}">{escape(label)}</span>')
    html = '<div class="pipeline-map">' + '<span class="pipeline-arrow">&rarr;</span>'.join(parts) + '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_glossary_expander(terms: Any) -> None:
    """Render glossary entries supplied as a mapping or iterable of labels."""
    if isinstance(terms, Mapping):
        entries = [(str(term), str(explanation)) for term, explanation in terms.items()]
    else:
        entries = [(str(term), "See the Guided Research Workflow glossary for the full definition.") for term in (terms or [])]
    if not entries:
        return
    with st.expander("Key terms", expanded=False):
        for term, explanation in entries:
            st.markdown(f"**{escape(term)}:** {escape(explanation)}")


def render_safe_table(df: Optional[pd.DataFrame], title: str, empty_message: str) -> None:
    render_section_header(title)
    table = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if table.empty:
        st.info(empty_message)
        return
    st.dataframe(table, width="stretch")


def render_download_buttons(table_map: Mapping[str, Any]) -> None:
    entries = list(table_map.items())
    if not entries:
        return
    for row_start in range(0, len(entries), 3):
        row_entries = entries[row_start:row_start + 3]
        columns = st.columns(len(row_entries))
        for index, (column, (label, value)) in enumerate(zip(columns, row_entries), start=row_start):
            if isinstance(value, tuple):
                table, filename = value
            else:
                table = value
                filename = f"{str(label).lower().replace(' ', '_')}.csv"
            frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame()
            with column:
                st.download_button(
                    str(label),
                    data=frame.to_csv(index=False).encode("utf-8"),
                    file_name=str(filename),
                    mime="text/csv",
                    key=f"ui_download_{filename}_{index}",
                    disabled=frame.empty,
                    width="stretch",
                )


__all__ = [
    "inject_premium_css",
    "render_status_card",
    "render_metric_grid",
    "render_section_header",
    "render_research_disclaimer",
    "render_blocked_capital_banner",
    "render_pipeline_stepper",
    "render_glossary_expander",
    "render_safe_table",
    "render_download_buttons",
]
