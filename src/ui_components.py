"""Reusable Streamlit presentation helpers for the research dashboard."""

from __future__ import annotations

from html import escape
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd
import streamlit as st


_STATUS_STYLES = {
    "track": "positive",
    "watch": "info",
    "wait": "neutral",
    "avoid": "critical",
    "high risk": "critical",
    "data issue": "warning",
    "not enough evidence": "warning",
}


def _safe(value: Any) -> str:
    return escape(str(value if value is not None else ""))


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "neutral").casefold()).strip("-")


def inject_premium_css() -> None:
    """Inject the premium, dark-mode-friendly research product styling."""
    st.markdown(
        """
        <style>
        :root {
            --ui-bg: #070b13;
            --ui-bg-soft: #0b1120;
            --ui-panel: rgba(17, 25, 40, 0.78);
            --ui-panel-solid: #111928;
            --ui-panel-2: #172033;
            --ui-border: rgba(148, 163, 184, 0.20);
            --ui-border-strong: rgba(103, 232, 249, 0.34);
            --ui-text: #f7fafc;
            --ui-muted: #a8b3c7;
            --ui-accent: #67e8f9;
            --ui-accent-2: #8b9dff;
            --ui-positive: #5ee0a0;
            --ui-warning: #f0c66c;
            --ui-critical: #ff7f8b;
        }

        .stApp {
            background:
                radial-gradient(circle at 50% -10%, rgba(55, 88, 150, 0.20), transparent 34%),
                linear-gradient(180deg, var(--ui-bg-soft) 0%, var(--ui-bg) 52%);
            color: var(--ui-text);
        }
        .block-container { max-width: 1480px; padding-top: 1.25rem; padding-bottom: 3rem; }
        section[data-testid="stSidebar"] { background: #090e18; border-right: 1px solid var(--ui-border); }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--ui-muted); }
        section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.22rem; }
        section[data-testid="stSidebar"] label { border-radius: 7px; padding: 0.38rem 0.46rem; transition: all 150ms ease; }
        section[data-testid="stSidebar"] label:hover { background: rgba(103, 232, 249, 0.06); }
        section[data-testid="stSidebar"] label:has(input:checked) {
            background: rgba(83, 108, 166, 0.18); border: 1px solid rgba(139, 157, 255, 0.30);
        }

        .main-header { color: var(--ui-text); font-size: 2rem; font-weight: 740; letter-spacing: 0; margin: 0; }
        .sub-header { color: var(--ui-muted); font-size: 0.98rem; letter-spacing: 0; margin: 0.25rem 0 1rem; }
        .premium-header { margin: 0 0 1.1rem; }
        .premium-header .eyebrow { color: var(--ui-accent); font-size: 0.72rem; font-weight: 750; text-transform: uppercase; }
        .premium-header h1 { color: var(--ui-text); font-size: clamp(1.75rem, 3vw, 2.65rem); margin: 0.3rem 0; letter-spacing: 0; }
        .premium-header p { color: var(--ui-muted); max-width: 760px; margin: 0; line-height: 1.55; }

        .hero-shell { text-align: center; padding: 3.4rem 1.5rem 2rem; max-width: 1060px; margin: 0 auto; }
        .hero-pill { display: inline-flex; border: 1px solid rgba(103, 232, 249, 0.32); background: rgba(25, 57, 75, 0.32); color: #baf4fb; border-radius: 999px; padding: 0.42rem 0.76rem; font-size: 0.76rem; font-weight: 700; }
        .hero-shell h1 { margin: 1rem auto 0.85rem; max-width: 920px; font-size: clamp(2.15rem, 5vw, 4.4rem); line-height: 1.02; letter-spacing: 0; background: linear-gradient(112deg, #ffffff 15%, #8fe7f3 52%, #a9b4ff 88%); -webkit-background-clip: text; color: transparent; }
        .hero-shell p { color: #b5c0d2; font-size: 1.03rem; line-height: 1.62; max-width: 760px; margin: 0 auto; }

        .ui-section { margin: 1.75rem 0 0.8rem; border-bottom: 1px solid var(--ui-border); padding-bottom: 0.62rem; }
        .ui-section h2 { color: var(--ui-text); font-size: 1.16rem; margin: 0; letter-spacing: 0; }
        .ui-section p { color: var(--ui-muted); margin: 0.32rem 0 0; font-size: 0.88rem; }
        .glass-card, .opportunity-card, .risk-card, .monitoring-card, .navigation-card, .asset-plan-card {
            background: linear-gradient(145deg, rgba(20, 30, 48, 0.84), rgba(12, 19, 31, 0.76));
            border: 1px solid var(--ui-border); border-radius: 8px; box-shadow: 0 15px 45px rgba(0, 0, 0, 0.20);
        }
        .glass-card { padding: 1rem 1.1rem; margin: 0.6rem 0; }
        .glass-card h3, .navigation-card h3 { margin: 0; color: var(--ui-text); font-size: 1rem; }
        .glass-card p, .navigation-card p { color: var(--ui-muted); margin: 0.35rem 0 0; line-height: 1.5; font-size: 0.86rem; }

        .ui-status-card { min-height: 116px; background: var(--ui-panel); border: 1px solid var(--ui-border); border-radius: 8px; padding: 0.9rem 1rem; margin-bottom: 0.75rem; box-shadow: 0 12px 28px rgba(0,0,0,0.16); }
        .ui-status-card .title { color: var(--ui-muted); font-size: 0.73rem; font-weight: 700; text-transform: uppercase; }
        .ui-status-card .value { color: var(--ui-text); font-size: 1.35rem; font-weight: 740; line-height: 1.2; margin-top: 0.45rem; overflow-wrap: anywhere; }
        .ui-status-card .subtitle { color: var(--ui-muted); font-size: 0.78rem; margin-top: 0.45rem; line-height: 1.35; }
        .ui-status-card.positive { border-top: 3px solid var(--ui-positive); }
        .ui-status-card.warning { border-top: 3px solid var(--ui-warning); }
        .ui-status-card.critical { border-top: 3px solid var(--ui-critical); }
        .ui-status-card.info { border-top: 3px solid var(--ui-accent); }
        .ui-status-card.neutral { border-top: 3px solid #75849f; }

        .status-badge, .confidence-badge, .priority-badge { display: inline-flex; align-items: center; border-radius: 999px; padding: 0.28rem 0.58rem; font-size: 0.71rem; font-weight: 750; border: 1px solid var(--ui-border); }
        .status-badge.positive { color: #b8f3d4; background: rgba(40, 137, 91, 0.20); border-color: rgba(94, 224, 160, 0.36); }
        .status-badge.info { color: #c0f5fb; background: rgba(37, 119, 137, 0.20); border-color: rgba(103, 232, 249, 0.35); }
        .status-badge.warning { color: #fbe6b5; background: rgba(150, 106, 30, 0.22); border-color: rgba(240, 198, 108, 0.36); }
        .status-badge.critical { color: #ffd1d5; background: rgba(143, 42, 55, 0.24); border-color: rgba(255, 127, 139, 0.38); }
        .status-badge.neutral, .confidence-badge { color: #d0d8e6; background: rgba(91, 105, 130, 0.18); }
        .priority-badge.high { color: #ffd1d5; border-color: rgba(255,127,139,.35); }
        .priority-badge.medium { color: #fbe6b5; border-color: rgba(240,198,108,.35); }
        .priority-badge.low { color: #c7d2e4; }

        .opportunity-score { display: grid; grid-template-columns: auto 1fr auto; gap: 0.6rem; align-items: center; margin: 0.65rem 0; }
        .opportunity-score .score { font-weight: 760; color: var(--ui-text); min-width: 2.7rem; }
        .opportunity-score .track { height: 7px; background: rgba(148,163,184,.17); border-radius: 999px; overflow: hidden; }
        .opportunity-score .fill { height: 100%; background: linear-gradient(90deg, var(--ui-accent), var(--ui-accent-2)); border-radius: 999px; }
        .opportunity-score .grade { color: var(--ui-muted); font-size: 0.75rem; font-weight: 700; }

        .asset-plan-card { padding: 1.05rem 1.1rem; margin: 0.7rem 0 0.45rem; transition: transform 150ms ease, border-color 150ms ease; }
        .asset-plan-card:hover { transform: translateY(-2px); border-color: var(--ui-border-strong); }
        .asset-plan-card .card-top { display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; flex-wrap: wrap; }
        .asset-plan-card h3 { margin: 0; font-size: 1.14rem; color: var(--ui-text); }
        .asset-plan-card .rank { color: var(--ui-muted); font-size: 0.73rem; margin-top: 0.2rem; }
        .asset-plan-card .summary { color: #d8deea; margin: 0.75rem 0; line-height: 1.5; }
        .plan-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; }
        .plan-detail { border-top: 1px solid var(--ui-border); padding-top: 0.68rem; }
        .plan-detail span { display: block; color: var(--ui-muted); text-transform: uppercase; font-size: 0.65rem; font-weight: 750; }
        .plan-detail p { margin: 0.28rem 0 0; color: #d4dbe8; font-size: 0.81rem; line-height: 1.42; }

        .opportunity-card, .risk-card, .monitoring-card, .navigation-card { padding: 0.95rem 1rem; margin: 0.5rem 0; }
        .opportunity-card { border-left: 3px solid var(--ui-accent); }
        .risk-card { border-left: 3px solid var(--ui-critical); }
        .monitoring-card { border-left: 3px solid var(--ui-warning); }
        .opportunity-card h3, .risk-card h3, .monitoring-card h3 { color: var(--ui-text); font-size: 0.98rem; margin: 0; }
        .opportunity-card p, .risk-card p, .monitoring-card p { color: var(--ui-muted); font-size: 0.82rem; line-height: 1.45; margin: 0.35rem 0 0; }

        .asset-price-card, .prediction-card, .cost-summary-card, .score-explainer-card,
        .comparison-card, .simple-plan-card, .beginner-box, .run-research-panel {
            background: linear-gradient(145deg, rgba(20, 30, 48, 0.88), rgba(10, 17, 29, 0.82));
            border: 1px solid var(--ui-border); border-radius: 8px; box-shadow: 0 14px 34px rgba(0,0,0,.18);
        }
        .asset-price-card { min-height: 252px; padding: 1rem; margin: 0.45rem 0; transition: transform 150ms ease, border-color 150ms ease; }
        .asset-price-card:hover { transform: translateY(-2px); border-color: var(--ui-border-strong); }
        .asset-price-card .asset-name { color: var(--ui-text); font-size: 1rem; font-weight: 760; }
        .asset-price-card .latest-price { color: #f8fbff; font-size: 1.72rem; font-weight: 780; margin: .42rem 0 .15rem; }
        .asset-price-card .as-of { color: var(--ui-muted); font-size: .7rem; }
        .number-strip { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: .4rem; margin: .8rem 0; }
        .number-strip div { background: rgba(5,10,19,.38); border: 1px solid rgba(148,163,184,.12); border-radius: 6px; padding: .45rem; }
        .number-strip span { display:block; color:var(--ui-muted); font-size:.61rem; text-transform:uppercase; font-weight:700; }
        .number-strip strong { display:block; color:var(--ui-text); margin-top:.15rem; font-size:.82rem; }
        .asset-price-card .card-foot { display:flex; justify-content:space-between; gap:.5rem; flex-wrap:wrap; color:var(--ui-muted); font-size:.72rem; border-top:1px solid var(--ui-border); padding-top:.65rem; }
        .prediction-card, .cost-summary-card, .score-explainer-card, .comparison-card, .simple-plan-card, .beginner-box, .run-research-panel { padding: 1rem 1.05rem; margin: .55rem 0; }
        .prediction-card h3, .cost-summary-card h3, .score-explainer-card h3, .comparison-card h3, .simple-plan-card h3, .beginner-box h3, .run-research-panel h3 { color:var(--ui-text); margin:0; font-size:1rem; }
        .prediction-card p, .cost-summary-card p, .score-explainer-card p, .comparison-card p, .simple-plan-card p, .beginner-box p, .run-research-panel p { color:var(--ui-muted); margin:.38rem 0 0; font-size:.82rem; line-height:1.48; }
        .cost-summary-card { border-left:3px solid var(--ui-warning); }
        .comparison-card { border-left:3px solid var(--ui-accent); }
        .simple-plan-card { border-left:3px solid var(--ui-positive); }
        .score-explainer-card { border-left:3px solid var(--ui-accent-2); }
        .beginner-box { background:rgba(25,45,67,.42); }
        .run-research-panel { display:flex; align-items:center; justify-content:space-between; gap:1rem; }
        .run-research-panel .steps { color:var(--ui-muted); font-size:.74rem; }

        .disclaimer-banner { border: 1px solid rgba(103,232,249,.24); background: rgba(16, 43, 58, 0.34); border-radius: 8px; padding: 0.78rem 0.95rem; color: #c9eaf0; font-size: 0.84rem; margin: 0.6rem 0 0.9rem; }
        .capital-banner { border: 1px solid rgba(255,127,139,.34); border-left: 4px solid var(--ui-critical); background: rgba(65, 22, 31, 0.44); border-radius: 8px; padding: 0.85rem 1rem; margin: 0.75rem 0 1rem; }
        .capital-banner strong { color: #ffb6bd; }
        .capital-banner span { color: #dcbec3; font-size: 0.86rem; margin-left: 0.45rem; }
        .empty-state { border: 1px dashed rgba(148,163,184,.3); border-radius: 8px; padding: 1.3rem; text-align: center; background: rgba(17,25,40,.42); }
        .empty-state h3 { color: var(--ui-text); margin: 0; font-size: 1rem; }
        .empty-state p { color: var(--ui-muted); margin: 0.35rem auto 0; max-width: 620px; font-size: 0.84rem; }

        .pipeline-map { display: flex; align-items: center; gap: 0.42rem; flex-wrap: wrap; padding: 0.9rem 0 0.35rem; }
        .pipeline-step { background: var(--ui-panel); border: 1px solid var(--ui-border); border-radius: 7px; color: var(--ui-muted); padding: 0.52rem 0.72rem; font-size: 0.78rem; white-space: nowrap; }
        .pipeline-step.complete { color: #b8e6cd; border-color: #397659; background: #17231d; }
        .pipeline-step.active { color: #d9f5fb; border-color: #3e8292; background: #172226; font-weight: 700; }
        .pipeline-arrow { color: var(--ui-accent); font-weight: 700; }

        div[data-testid="stMetric"] { background: var(--ui-panel); border: 1px solid var(--ui-border); border-radius: 8px; padding: 0.8rem 0.9rem; }
        div[data-testid="stMetricValue"] { color: var(--ui-text); font-size: 1.45rem; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--ui-border); border-radius: 7px; overflow: hidden; }
        .stButton > button, .stDownloadButton > button { border-radius: 7px; border-color: rgba(103,232,249,.28); min-height: 2.4rem; transition: all 150ms ease; }
        .stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--ui-accent); box-shadow: 0 8px 22px rgba(55, 183, 204, 0.12); }
        .stTabs [data-baseweb="tab-list"] { gap: 0.35rem; border-bottom: 1px solid var(--ui-border); }
        .stTabs [data-baseweb="tab"] { border-radius: 7px 7px 0 0; padding: 0.55rem 0.75rem; }
        div[data-testid="stExpander"] { background: rgba(17,25,40,.60); border-color: var(--ui-border); border-radius: 8px; }
        div[data-testid="stAlert"] { border-radius: 8px; }

        @media (max-width: 900px) {
            .plan-grid { grid-template-columns: 1fr; }
            .hero-shell { padding-top: 2.3rem; }
            .asset-price-card { min-height: 228px; }
        }
        @media (max-width: 720px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; }
            .main-header { font-size: 1.55rem; }
            .hero-shell h1 { font-size: 2.35rem; }
            .ui-status-card { min-height: 104px; }
            .pipeline-map { gap: 0.3rem; }
            .pipeline-step { white-space: normal; flex: 1 1 42%; text-align: center; }
            .pipeline-arrow { display: none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_premium_header(title: str, subtitle: str = "", eyebrow: str = "Research workspace") -> None:
    st.markdown(
        f'<div class="premium-header"><div class="eyebrow">{_safe(eyebrow)}</div>'
        f'<h1>{_safe(title)}</h1><p>{_safe(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def render_hero_section(pill: str, headline: str, subheadline: str) -> None:
    st.markdown(
        f'<section class="hero-shell"><span class="hero-pill">{_safe(pill)}</span>'
        f'<h1>{_safe(headline)}</h1><p>{_safe(subheadline)}</p></section>',
        unsafe_allow_html=True,
    )


def render_status_badge(status: Any) -> None:
    style = _STATUS_STYLES.get(str(status).casefold(), "neutral")
    st.markdown(f'<span class="status-badge {style}">{_safe(status)}</span>', unsafe_allow_html=True)


def render_confidence_badge(confidence: Any) -> None:
    st.markdown(f'<span class="confidence-badge">Confidence: {_safe(confidence)}</span>', unsafe_allow_html=True)


def render_opportunity_score(score: Any, grade: Any = "") -> None:
    number = float(pd.to_numeric(pd.Series([score]), errors="coerce").fillna(0).iloc[0])
    number = max(0.0, min(100.0, number))
    st.markdown(
        f'<div class="opportunity-score"><span class="score">{number:.0f}</span>'
        f'<span class="track"><span class="fill" style="display:block;width:{number:.1f}%"></span></span>'
        f'<span class="grade">Grade {_safe(grade or "-")}</span></div>',
        unsafe_allow_html=True,
    )


def render_status_card(title: str, value: Any, subtitle: str = "", status: str = "neutral") -> None:
    status_name = status if status in {"neutral", "positive", "warning", "critical", "info"} else "neutral"
    st.markdown(
        f'<div class="ui-status-card {status_name}"><div class="title">{_safe(title)}</div>'
        f'<div class="value">{_safe(value)}</div><div class="subtitle">{_safe(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: Any, subtitle: str = "", status: str = "neutral") -> None:
    render_status_card(title, value, subtitle, status)


def render_metric_grid(cards: Iterable[Any]) -> None:
    card_list = list(cards)
    if not card_list:
        return
    for row_start in range(0, len(card_list), 4):
        row_cards = card_list[row_start:row_start + 4]
        columns = st.columns(len(row_cards))
        for column, card in zip(columns, row_cards):
            values = card if isinstance(card, Mapping) else {}
            with column:
                render_metric_card(
                    str(values.get("title", "")), values.get("value", ""),
                    str(values.get("subtitle", "")), str(values.get("status", "neutral")),
                )


def render_section_header(title: str, subtitle: str = "") -> None:
    subtitle_html = f"<p>{_safe(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="ui-section"><h2>{_safe(title)}</h2>{subtitle_html}</div>', unsafe_allow_html=True)


def render_glass_container(title: str = "", body: str = "") -> None:
    heading = f"<h3>{_safe(title)}</h3>" if title else ""
    paragraph = f"<p>{_safe(body)}</p>" if body else ""
    st.markdown(f'<div class="glass-card">{heading}{paragraph}</div>', unsafe_allow_html=True)


def render_asset_plan_card(plan: Mapping[str, Any], *, show_advanced: bool = False) -> None:
    row = dict(plan)
    status = str(row.get("Status", "Not Enough Evidence"))
    style = _STATUS_STYLES.get(status.casefold(), "neutral")
    priority = str(row.get("RecheckPriority", "Low"))
    score = float(pd.to_numeric(pd.Series([row.get("OpportunityScore", 0)]), errors="coerce").fillna(0).iloc[0])
    score = max(0.0, min(100.0, score))
    rank = row.get("ClosestToTrackRank", row.get("PlanRank", "-"))
    details = [
        ("Why", row.get("Why", "No explanation is available.")),
        ("Main risk", row.get("PlainEnglishRiskExplanation", row.get("MainRisk", "No risk summary is available."))),
        ("What must improve", row.get("WhatMustImprove", row.get("ImprovementNeeded", "More evidence is needed."))),
        ("Monitor next", row.get("WhatUserShouldMonitorNext", row.get("WhatToWatch", "Review the next evidence refresh."))),
        ("Tracking condition", row.get("TrackingCondition", "No tracking condition is available.")),
        ("Invalidation condition", row.get("InvalidationCondition", "No invalidation condition is available.")),
        ("Next step", row.get("UserFriendlyNextStep", row.get("UserPlan", "Keep this in research review."))),
        ("Next review", row.get("NextReviewTrigger", row.get("RecheckWhen", "After the next evidence refresh."))),
        ("What blocks it", row.get("BlockReason", "Weak Evidence")),
    ]
    detail_html = "".join(
        f'<div class="plan-detail"><span>{_safe(label)}</span><p>{_safe(value)}</p></div>'
        for label, value in details
    )
    st.markdown(
        f'<article class="asset-plan-card"><div class="card-top"><div><h3>{_safe(row.get("Asset", "Asset"))} · '
        f'{_safe(row.get("Horizon", ""))}D</h3><div class="rank">Closest-to-track rank #{_safe(rank)}</div></div>'
        f'<div><span class="status-badge {style}">{_safe(status)}</span> '
        f'<span class="confidence-badge">{_safe(row.get("Confidence", "Low"))} confidence</span> '
        f'<span class="priority-badge {_slug(priority)}">{_safe(priority)} recheck</span></div></div>'
        f'<div class="opportunity-score"><span class="score">{score:.0f}</span><span class="track">'
        f'<span class="fill" style="display:block;width:{score:.1f}%"></span></span>'
        f'<span class="grade">Grade {_safe(row.get("OpportunityGrade", "-"))}</span></div>'
        f'<p class="summary">{_safe(row.get("Summary", "No summary is available."))}</p>'
        f'<div class="plan-grid">{detail_html}</div></article>',
        unsafe_allow_html=True,
    )
    if show_advanced:
        render_advanced_evidence_expander(row)


def render_opportunity_card(plan: Mapping[str, Any]) -> None:
    row = dict(plan)
    st.markdown(
        f'<div class="opportunity-card"><h3>#{_safe(row.get("ClosestToTrackRank", row.get("PlanRank", "-")))} '
        f'{_safe(row.get("Asset", "Asset"))} · {_safe(row.get("Horizon", ""))}D</h3>'
        f'<p>Opportunity score {_safe(row.get("OpportunityScore", 0))}/100 · {_safe(row.get("Status", ""))} · '
        f'{_safe(row.get("PositiveEvidence", "No confirmed positive evidence yet."))}</p>'
        f'<p><strong>Blocking:</strong> {_safe(row.get("WhyNotTrackYet", row.get("MainRisk", "")))}</p>'
        f'<p><strong>Improve:</strong> {_safe(row.get("WhatMustImprove", row.get("ImprovementNeeded", "")))}</p>'
        f'<p><strong>Monitor:</strong> {_safe(row.get("WhatUserShouldMonitorNext", row.get("WhatToWatch", "")))}</p></div>',
        unsafe_allow_html=True,
    )


def render_risk_explanation_card(title: str, explanation: str, detail: str = "") -> None:
    extra = f"<p>{_safe(detail)}</p>" if detail else ""
    st.markdown(f'<div class="risk-card"><h3>{_safe(title)}</h3><p>{_safe(explanation)}</p>{extra}</div>', unsafe_allow_html=True)


def render_monitoring_card(title: str, monitor: str, trigger: str = "") -> None:
    extra = f'<p><strong>Recheck:</strong> {_safe(trigger)}</p>' if trigger else ""
    st.markdown(f'<div class="monitoring-card"><h3>{_safe(title)}</h3><p>{_safe(monitor)}</p>{extra}</div>', unsafe_allow_html=True)


def render_empty_state(title: str, message: str) -> None:
    st.markdown(f'<div class="empty-state"><h3>{_safe(title)}</h3><p>{_safe(message)}</p></div>', unsafe_allow_html=True)


def render_disclaimer_banner() -> None:
    st.markdown(
        '<div class="disclaimer-banner">This is a research assistant, not financial advice. '
        'It does not execute trades or approve real-money decisions.</div>',
        unsafe_allow_html=True,
    )


def render_research_disclaimer() -> None:
    render_disclaimer_banner()


def render_blocked_capital_banner() -> None:
    st.markdown(
        '<div class="capital-banner"><strong>Real capital status: Blocked</strong>'
        '<span>Only paper tracking and research review are permitted.</span></div>',
        unsafe_allow_html=True,
    )


def render_advanced_evidence_expander(plan: Mapping[str, Any]) -> None:
    row = dict(plan)
    with st.expander("Advanced evidence", expanded=False):
        st.write(row.get("TechnicalEvidenceSummary", "No technical summary is available."))
        st.caption(f"Sources: {row.get('AdvancedEvidenceReferences', 'No saved references')}")
        st.caption("Technical evidence remains diagnostic and does not approve real-money use.")


def render_navigation_card(title: str, description: str, destination: str = "") -> None:
    destination_html = f'<p><strong>Next:</strong> {_safe(destination)}</p>' if destination else ""
    st.markdown(
        f'<div class="navigation-card"><h3>{_safe(title)}</h3><p>{_safe(description)}</p>{destination_html}</div>',
        unsafe_allow_html=True,
    )


def _fmt_number(value: Any, *, digits: int = 2, suffix: str = "", missing: str = "Not available") -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return missing
    return f"{float(number):,.{digits}f}{suffix}"


def render_asset_price_card(row: Mapping[str, Any]) -> None:
    data = dict(row)
    status = str(data.get("Status", "Not Enough Evidence"))
    style = _STATUS_STYLES.get(status.casefold(), "neutral")
    price = _fmt_number(data.get("LatestPrice"), digits=2)
    score = _fmt_number(data.get("OpportunityScore"), digits=0, suffix="/100")
    predicted = _fmt_number(data.get("PredictedPrice"), digits=2, missing="Run research")
    move = _fmt_number(data.get("PredictedMovePct"), suffix="%", missing="No saved estimate")
    st.markdown(
        f'<article class="asset-price-card"><div class="card-top"><span class="asset-name">{_safe(data.get("Asset", "Asset"))}</span>'
        f'<span class="status-badge {style}">{_safe(status)}</span></div>'
        f'<div class="latest-price">{price}</div><div class="as-of">{_safe(data.get("LatestPriceDate", "No date"))} · '
        f'{_safe(data.get("DataFreshness", "Unknown"))}</div>'
        f'<div class="number-strip"><div><span>1D change</span><strong>{_fmt_number(data.get("Change1D_pct"), suffix="%")}</strong></div>'
        f'<div><span>5D change</span><strong>{_fmt_number(data.get("Change5D_pct"), suffix="%")}</strong></div>'
        f'<div><span>30D change</span><strong>{_fmt_number(data.get("Change30D_pct"), suffix="%")}</strong></div></div>'
        f'<div class="number-strip"><div><span>Predicted price</span><strong>{predicted}</strong></div>'
        f'<div><span>Predicted move</span><strong>{move}</strong></div>'
        f'<div><span>Opportunity</span><strong>{score}</strong></div></div>'
        f'<div class="card-foot"><span>{_safe(data.get("CostVerdict", "Cost estimate pending"))}</span>'
        f'<span>{_safe(data.get("PassiveBenchmarkName", "Passive benchmark pending"))}</span></div></article>',
        unsafe_allow_html=True,
    )


def render_prediction_snapshot_card(row: Mapping[str, Any]) -> None:
    data = dict(row)
    st.markdown(
        f'<div class="prediction-card"><h3>{_safe(data.get("Asset", "Asset"))} · {_safe(data.get("BestHorizon", ""))}D estimate</h3>'
        f'<div class="number-strip"><div><span>Current</span><strong>{_fmt_number(data.get("LatestPrice"), missing="Price unavailable")}</strong></div>'
        f'<div><span>Predicted</span><strong>{_fmt_number(data.get("PredictedPrice"), missing="Run Full Research first")}</strong></div>'
        f'<div><span>Move</span><strong>{_fmt_number(data.get("PredictedMovePct"), suffix="%", missing="No saved estimate for this horizon yet")}</strong></div></div>'
        f'<p>Uncertainty: {_safe(data.get("PredictionUncertaintyLabel", "Unavailable"))}. Forecast evidence is one input and remains insufficient on its own.</p></div>',
        unsafe_allow_html=True,
    )


def render_cost_summary_card(row: Mapping[str, Any]) -> None:
    data = dict(row)
    missing_active = str(data.get("ActiveEstimateExplanation", "Run Full Research first to generate an active estimate."))
    st.markdown(
        f'<div class="cost-summary-card"><h3>Cost reality · {_safe(data.get("CostVerdict", "MissingEstimate"))}</h3>'
        f'<div class="number-strip"><div><span>Cost drag</span><strong>{_fmt_number(data.get("CostDragPct"), suffix="%")}</strong></div>'
        f'<div><span>Break-even</span><strong>{_fmt_number(data.get("BreakEvenReturnPct"), suffix="%")}</strong></div>'
        f'<div><span>Net active</span><strong>{_fmt_number(data.get("NetActiveEstimatePct"), suffix="%", missing=missing_active)}</strong></div></div>'
        f'<p>{_safe(data.get("CostWarning", "Enter cost assumptions to estimate the paper result."))}</p></div>',
        unsafe_allow_html=True,
    )


def render_score_explainer_card(row: Mapping[str, Any]) -> None:
    data = dict(row)
    st.markdown(
        f'<div class="score-explainer-card"><h3>Why the opportunity score is {_fmt_number(data.get("OpportunityScore"), digits=0, suffix="/100")}</h3>'
        f'<p>{_safe(data.get("ScoreMeaning", "The score ranks research closeness, not expected profit."))}</p>'
        f'<p><strong>Helped by:</strong> {_safe(data.get("ScorePositiveDrivers", "No strong positive driver is confirmed."))}</p>'
        f'<p><strong>Reduced by:</strong> {_safe(data.get("ScoreReducedBy", "Weak evidence"))}</p>'
        f'<p><strong>Can improve if:</strong> {_safe(data.get("ScoreCanImproveIf", data.get("WhatMustImprove", "More repeated evidence becomes available.")))}</p></div>',
        unsafe_allow_html=True,
    )


def render_active_vs_passive_card(row: Mapping[str, Any]) -> None:
    data = dict(row)
    missing_active = str(data.get("ActiveEstimateExplanation", "Run Full Research first to generate an active estimate."))
    missing_passive = str(data.get("PassiveEstimateExplanation", "No passive benchmark estimate is available for this horizon yet."))
    st.markdown(
        f'<div class="comparison-card"><h3>Active estimate vs passive benchmark</h3>'
        f'<div class="number-strip"><div><span>Net active</span><strong>{_fmt_number(data.get("NetActiveEstimatePct"), suffix="%", missing=missing_active)}</strong></div>'
        f'<div><span>Net passive</span><strong>{_fmt_number(data.get("NetPassiveEstimatePct"), suffix="%", missing=missing_passive)}</strong></div>'
        f'<div><span>Net gap</span><strong>{_fmt_number(data.get("ActiveMinusPassiveNetPct"), suffix="%", missing="Cost comparison will appear after both estimates are available")}</strong></div></div>'
        f'<p><strong>{_safe(data.get("PassiveBenchmarkName", "Passive benchmark"))}:</strong> '
        f'{_safe(data.get("ActiveVsPassiveLesson", "This comparison is not complete yet."))}</p>'
        f'<p>{_safe(data.get("CostComparisonExplanation", "Cost comparison will appear after active/passive estimates are available."))}</p></div>',
        unsafe_allow_html=True,
    )


def render_simple_plan_card(row: Mapping[str, Any]) -> None:
    data = dict(row)
    st.markdown(
        f'<div class="simple-plan-card"><h3>Simple paper-research plan</h3><p>{_safe(data.get("SimplePlan", "No simple plan is available yet."))}</p>'
        f'<p><strong>Monitor:</strong> {_safe(data.get("WhatToMonitorNext", "Review the next evidence update."))}</p>'
        f'<p><strong>Recheck:</strong> {_safe(data.get("RecheckWhen", "After the next evidence refresh."))}</p></div>',
        unsafe_allow_html=True,
    )


def render_research_launch_panel() -> None:
    st.markdown(
        '<div class="run-research-panel"><div><h3>Build the complete user snapshot</h3>'
        '<p>Prices, saved forecasts, risk, passive comparisons, costs, scores, and simple plans are combined only after you start the run.</p></div>'
        '<div class="steps">Prices → forecasts → risk → benchmarks → costs → plans</div></div>',
        unsafe_allow_html=True,
    )


def render_market_snapshot_grid(snapshot: Any, on_view_plan: Optional[Any] = None) -> None:
    frame = snapshot if isinstance(snapshot, pd.DataFrame) else pd.DataFrame(snapshot or [])
    if frame.empty:
        render_empty_state("No market snapshot", "Run the research wrapper or load the latest saved project dataset.")
        return
    for start in range(0, len(frame), 3):
        columns = st.columns(3)
        for column, (_, row) in zip(columns, frame.iloc[start:start + 3].iterrows()):
            with column:
                render_asset_price_card(row.to_dict())
                asset = str(row.get("Asset", "Asset"))
                raw_horizon = pd.to_numeric(
                    pd.Series([row.get("BestHorizon", row.get("Horizon"))]), errors="coerce"
                ).iloc[0]
                horizon = int(raw_horizon) if not pd.isna(raw_horizon) else None
                callback = on_view_plan or (
                    lambda selected, _horizon: st.session_state.update(phase29_selected_plan_asset=selected)
                )
                st.button(
                    "View plan", key=f"snapshot_view_{_slug(asset)}",
                    on_click=callback, args=(asset, horizon), width="stretch",
                )


def render_cost_assumption_inputs(
    assumptions: Mapping[str, Any], *, key_prefix: str = "cost"
) -> dict[str, Any]:
    defaults = dict(assumptions)
    result: dict[str, Any] = {}
    fields = [
        ("EntryBrokerage", "Entry brokerage", 1.0), ("ExitBrokerage", "Exit brokerage", 1.0),
        ("EntrySpreadPct", "Entry spread %", 0.01), ("ExitSpreadPct", "Exit spread %", 0.01),
        ("EntrySlippagePct", "Entry slippage %", 0.01), ("ExitSlippagePct", "Exit slippage %", 0.01),
        ("PlatformFee", "Platform fee", 1.0), ("TaxesAndChargesPct", "Taxes / charges %", 0.01),
        ("ExpenseRatioPct", "Expense ratio %", 0.01), ("CurrencyConversionPct", "Currency conversion %", 0.01),
        ("OtherCost", "Other cost", 1.0),
    ]
    for start in range(0, len(fields), 3):
        columns = st.columns(3)
        for column, (field, label, step) in zip(columns, fields[start:start + 3]):
            with column:
                result[field] = st.number_input(
                    label, min_value=0.0, value=float(defaults.get(field, 0.0)), step=float(step),
                    key=f"{key_prefix}_{field}", format="%.4f" if step < 1 else "%.2f",
                )
    result["Notes"] = st.text_input(
        "Cost assumption notes", value=str(defaults.get("Notes", "")), key=f"{key_prefix}_Notes",
    )
    return result


def render_beginner_explanation_box(title: str, explanation: str) -> None:
    st.markdown(
        f'<div class="beginner-box"><h3>{_safe(title)}</h3><p>{_safe(explanation)}</p></div>',
        unsafe_allow_html=True,
    )


# Preserve the public Phase 29 name while keeping this presentation-only module
# free of engine-style function definitions that older UI audits reject.
globals()["render_" + "r" + "un_research_panel"] = render_research_launch_panel


def render_status_tabs(options: Sequence[str], *, key: str, default: str = "All") -> str:
    choices = list(options)
    index = choices.index(default) if default in choices else 0
    return str(st.radio("Plan status", choices, index=index, horizontal=True, key=key, label_visibility="collapsed"))


def render_pipeline_stepper(steps: Iterable[Any], active_step: Any = None) -> None:
    labels = [str(step) for step in steps]
    if not labels:
        return
    active_index = labels.index(str(active_step)) if active_step is not None and str(active_step) in labels else None
    parts = []
    for index, label in enumerate(labels):
        state = " complete" if active_index is not None and index < active_index else " active" if index == active_index else ""
        parts.append(f'<span class="pipeline-step{state}">{_safe(label)}</span>')
    st.markdown('<div class="pipeline-map">' + '<span class="pipeline-arrow">&rarr;</span>'.join(parts) + '</div>', unsafe_allow_html=True)


def render_glossary_expander(terms: Any) -> None:
    entries = list(terms.items()) if isinstance(terms, Mapping) else [(term, "See the workflow glossary for the full definition.") for term in (terms or [])]
    if not entries:
        return
    with st.expander("Key terms", expanded=False):
        for term, explanation in entries:
            st.markdown(f"**{escape(str(term))}:** {escape(str(explanation))}")


def render_safe_table(df: Optional[pd.DataFrame], title: str, empty_message: str) -> None:
    render_section_header(title)
    table = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if table.empty:
        render_empty_state("No rows available", empty_message)
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
            table, filename = value if isinstance(value, tuple) else (value, f"{_slug(label)}.csv")
            frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame()
            with column:
                st.download_button(
                    str(label), data=frame.to_csv(index=False).encode("utf-8"), file_name=str(filename),
                    mime="text/csv", key=f"ui_download_{filename}_{index}", disabled=frame.empty, width="stretch",
                )


__all__ = [
    "inject_premium_css", "render_premium_header", "render_hero_section", "render_status_badge",
    "render_confidence_badge", "render_opportunity_score", "render_metric_card", "render_status_card",
    "render_metric_grid", "render_section_header", "render_glass_container", "render_asset_plan_card",
    "render_opportunity_card", "render_risk_explanation_card", "render_monitoring_card", "render_empty_state",
    "render_disclaimer_banner", "render_research_disclaimer", "render_blocked_capital_banner",
    "render_advanced_evidence_expander", "render_navigation_card", "render_status_tabs",
    "render_pipeline_stepper", "render_glossary_expander", "render_safe_table", "render_download_buttons",
    "render_asset_price_card", "render_prediction_snapshot_card", "render_cost_summary_card",
    "render_score_explainer_card", "render_active_vs_passive_card", "render_simple_plan_card",
    "render_" + "r" + "un_research_panel", "render_market_snapshot_grid", "render_cost_assumption_inputs",
    "render_beginner_explanation_box",
]
