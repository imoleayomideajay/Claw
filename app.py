"""Streamlit app for interactive AI fairness auditing simulations."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

from src.baseline import fit_chamberlain_baseline
from src.fairness_metrics import dataset_summary, marginal_and_intersection_metrics
from src.frequentist_model import fit_frequentist_fairness_model
from src.ias import compute_bayesian_ias_interval, compute_ias
from src.inclusion_exclusion import compute_union_metrics_pie
from src.simulate_data import (
    build_audit_outcome,
    simulate_algorithm,
    simulate_ground_truth,
    simulate_population,
)
from src.utils import SCENARIO_ORDER  # noqa: F401


HAS_BAYES_STACK = find_spec("pymc") is not None and find_spec("arviz") is not None

st.set_page_config(
    page_title="AI Fairness Audit Lab",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded",
)

# ── Palette ───────────────────────────────────────────────────────────────────

P = {
    "primary": "#6366f1",
    "primary_light": "#818cf8",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "dark": "#0f172a",
    "slate": "#1e293b",
    "muted": "#64748b",
    "subtle": "#94a3b8",
    "border": "#e2e8f0",
    "surface": "#f8fafc",
    "card": "#ffffff",
}

# ── Labels ────────────────────────────────────────────────────────────────────

SCENARIO_LABELS = {
    "fair_algorithm":               "Fair Algorithm",
    "marginal_bias":                "Biased Algorithm (Marginal Bias)",
    "intersectional_bias":          "Intersectional Bias",
    "proxy_feature_bias":           "Proxy Feature Bias",
    "differential_error_rate_bias": "Differential Error Rate Bias",
}

AUDIT_LABELS = {
    "positive_decision": "Positive Decisions (approvals, hires, etc.)",
    "false_positive":    "False Positives (wrongly flagged as risky)",
    "false_negative":    "False Negatives (missed true positives)",
}

FRIENDLY_GROUP_NAMES = {
    "gender=1":                     "Women",
    "race=1":                       "Minority Race",
    "disability=1":                 "People with Disabilities",
    "gender=1,race=1":              "Women + Minority Race",
    "gender=1,disability=1":        "Women + Disabled",
    "race=1,disability=1":          "Minority Race + Disabled",
    "gender=1,race=1,disability=1": "Women + Minority Race + Disabled",
}

FRIENDLY_UNION_NAMES = {
    "A\u222aB":        "Women or Minority Race",
    "A\u222aC":        "Women or Disabled",
    "B\u222aC":        "Minority Race or Disabled",
    "A\u222aB\u222aC": "Any Protected Group",
}

# ── Global CSS ────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.main .block-container {
    background: #f8fafc;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}

/* ── Sidebar dark theme ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1a2744 100%) !important;
    border-right: none !important;
    box-shadow: 2px 0 20px rgba(0,0,0,0.25) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stSidebar"] h1 {
    color: #f1f5f9 !important;
    font-size: 1.15rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #1e3a5f !important;
    margin: 0.9rem 0 !important;
    opacity: 0.7 !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    height: 2.9rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.45) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    box-shadow: 0 6px 24px rgba(99,102,241,0.6) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #64748b !important;
    padding: 0.65rem 1.3rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #6366f1 !important;
}
[data-testid="stTabPanel"] {
    padding-top: 1.2rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: white;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.03);
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    line-height: 1.1 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06) !important;
}

/* ── Expander ── */
details {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    background: white !important;
    overflow: hidden !important;
}
details summary {
    padding: 0.7rem 1rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_interactive_pipeline(
    scenario: str,
    audit_outcome: str,
    n: int,
    seed: int,
    threshold: float,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame,
]:
    pop = simulate_population(n=n, seed=seed)
    truth = simulate_ground_truth(pop, seed=seed)
    scored = simulate_algorithm(truth, scenario=scenario, seed=seed, base_threshold=threshold)

    z_col = f"Z_{audit_outcome}"
    scored[z_col] = build_audit_outcome(scored, audit_outcome)

    _, with_pi0 = fit_chamberlain_baseline(scored, z_col=z_col)
    marginal, intersection = marginal_and_intersection_metrics(with_pi0, z_col=z_col)
    union = compute_union_metrics_pie(with_pi0, z_col=z_col)
    freq_fit, freq_table = fit_frequentist_fairness_model(with_pi0, z_col=z_col)
    ias = compute_ias(with_pi0, dict(freq_fit.params))
    summary = dataset_summary(with_pi0)
    return with_pi0, summary, marginal, intersection, union, ias, freq_table


# ── Data helpers ──────────────────────────────────────────────────────────────

def round_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    float_cols = df.select_dtypes(include="float").columns
    result = df.copy()
    result[float_cols] = result[float_cols].round(decimals)
    return result


def round_dict(d: dict, decimals: int = 2) -> dict:
    return {k: round(float(v), decimals) for k, v in d.items()}


def fairness_verdict(ias_score: float) -> tuple[str, str]:
    if ias_score < 0.05:
        return (
            "Likely Fair",
            "Identity-related factors (gender, race, disability) account for very little "
            "of the outcome variation. The algorithm appears to be treating people fairly.",
        )
    elif ias_score < 0.15:
        return (
            "Borderline",
            "A moderate amount of outcome variation is tied to identity characteristics. "
            "The algorithm warrants monitoring, but is not a clear-cut case of bias.",
        )
    else:
        return (
            "Likely Unfair",
            "A substantial portion of outcomes appear to be driven by who someone is rather "
            "than what they have done or achieved. This warrants urgent investigation.",
        )


def friendly_marginal_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group"] = out["group"].map(FRIENDLY_GROUP_NAMES).fillna(out["group"])
    return round_df(
        out.rename(columns={
            "group": "Group", "n_group": "Group Size",
            "observed_count": "Actual Outcomes", "expected_count": "Expected Outcomes",
            "raw_gap": "Fairness Gap", "standardised_gap": "Gap (Std.)",
            "rate_disparity": "Rate vs. Others",
        })
    )


def friendly_union_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["union"] = out["union"].map(FRIENDLY_UNION_NAMES).fillna(out["union"])
    return round_df(
        out.rename(columns={
            "union": "Group Combination", "prevalence": "Population Share",
            "n_union": "Group Size", "n_complement": "Everyone Else",
            "observed_count": "Actual Outcomes", "expected_count": "Expected Outcomes",
            "fairness_gap": "Fairness Gap", "union_minus_complement_rate": "Rate vs. Non-Group",
        })
    )


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _chart_style(ax: plt.Axes, title: str, ylabel: str = "") -> None:
    ax.set_facecolor("#ffffff")
    ax.figure.patch.set_facecolor("#ffffff")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#f1f5f9", linewidth=1.3, linestyle="solid")
    ax.xaxis.grid(False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", which="both", length=0, labelsize=9, labelcolor="#64748b")
    ax.set_title(title, fontsize=13, fontweight="bold", color="#0f172a", pad=14, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#94a3b8", labelpad=8)


def make_gap_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label_map: dict | None = None,
) -> plt.Figure:
    labels = (
        df[x_col].map(x_label_map).fillna(df[x_col]).tolist()
        if x_label_map
        else df[x_col].tolist()
    )
    values = df[y_col].values
    n = len(values)
    fig_w = max(7.5, n * 1.55)

    fig, ax = plt.subplots(figsize=(fig_w, 5.0))

    bar_colors = [P["success"] if v >= 0 else P["danger"] for v in values]
    bars = ax.bar(
        range(n), values,
        color=bar_colors, alpha=0.88, width=0.58, zorder=3,
        linewidth=0, edgecolor="none",
    )

    ax.axhline(0, color=P["dark"], linewidth=1.2, alpha=0.18, zorder=2)

    for bar, val in zip(bars, values):
        offset = abs(val) * 0.10 + 0.18
        y_pos = val + offset if val >= 0 else val - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"{val:+.2f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9, fontweight="700",
            color=P["success"] if val >= 0 else P["danger"],
        )

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9.5)

    _chart_style(ax, title, ylabel="Gap (Observed \u2212 Expected)")

    green_p = mpatches.Patch(color=P["success"], alpha=0.88, label="Above expected (favorable)")
    red_p   = mpatches.Patch(color=P["danger"],  alpha=0.88, label="Below expected (unfavorable)")
    ax.legend(
        handles=[green_p, red_p], fontsize=8.5, frameon=True,
        framealpha=0.92, edgecolor=P["border"], loc="upper right",
        borderpad=0.9, labelcolor="#374151",
    )

    fig.tight_layout(pad=1.5)
    return fig


def make_semicircle_gauge(score: float) -> plt.Figure:
    score = float(min(max(score, 0.0), 1.0))

    if score < 0.05:
        fill_color, verdict_label = P["success"], "Likely Fair"
    elif score < 0.15:
        fill_color, verdict_label = P["warning"], "Borderline"
    else:
        fill_color, verdict_label = P["danger"],  "Likely Unfair"

    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    fig.patch.set_facecolor("#ffffff")
    ax.set_aspect("equal")
    ax.axis("off")

    lw = 24
    theta_full = np.linspace(0.0, np.pi, 400)

    ax.plot(
        np.cos(theta_full), np.sin(theta_full),
        color="#f1f5f9", linewidth=lw, solid_capstyle="round", zorder=1,
    )

    zone_specs = [
        (0.00, 0.05, P["success"]),
        (0.05, 0.15, P["warning"]),
        (0.15, 1.00, P["danger"]),
    ]
    for z_start, z_end, z_color in zone_specs:
        a0 = np.pi * (1.0 - z_end)
        a1 = np.pi * (1.0 - z_start)
        th = np.linspace(a0, a1, 200)
        ax.plot(
            np.cos(th), np.sin(th),
            color=z_color, alpha=0.18, linewidth=lw,
            solid_capstyle="butt", zorder=2,
        )

    for pct in (0.05, 0.15):
        a = np.pi * (1.0 - pct)
        ax.plot(
            [0.86 * np.cos(a), 1.04 * np.cos(a)],
            [0.86 * np.sin(a), 1.04 * np.sin(a)],
            color="#cbd5e1", linewidth=1.8, zorder=4,
        )
        ax.text(
            1.18 * np.cos(a), 1.18 * np.sin(a),
            f"{int(pct * 100)}%", ha="center", va="center",
            fontsize=7, color="#94a3b8",
        )

    if score > 0.002:
        end_angle = np.pi * (1.0 - score)
        theta_fill = np.linspace(end_angle, np.pi, 400)
        ax.plot(
            np.cos(theta_fill), np.sin(theta_fill),
            color=fill_color, linewidth=lw, solid_capstyle="round", zorder=3,
        )

    needle_angle = np.pi * (1.0 - score)
    ax.annotate(
        "",
        xy=(0.70 * np.cos(needle_angle), 0.70 * np.sin(needle_angle)),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="#1e293b", lw=2.2, mutation_scale=14),
        zorder=5,
    )
    ax.scatter([0], [0], color="#1e293b", s=90, zorder=6)

    ax.text(
        0, 0.15, f"{score * 100:.1f}%",
        ha="center", va="center",
        fontsize=28, fontweight="black",
        color=fill_color, zorder=7,
    )
    ax.text(
        0, -0.10, verdict_label,
        ha="center", va="center",
        fontsize=10, fontweight="600",
        color="#64748b", zorder=7,
    )

    ax.text(-1.18, -0.05, "0%\nFair",     ha="center", fontsize=7.5, color="#94a3b8", linespacing=1.4)
    ax.text( 1.18, -0.05, "100%\nBiased", ha="center", fontsize=7.5, color="#94a3b8", linespacing=1.4)
    ax.text( 0,    1.25,  "IAS Gauge",   ha="center", fontsize=9,   color="#94a3b8", fontweight="500")

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.28, 1.40)
    fig.tight_layout(pad=0.1)
    return fig


# ── HTML components ───────────────────────────────────────────────────────────

def _card(icon: str, label: str, value: str, subtitle: str = "", accent: str = "#6366f1") -> str:
    sub_html = (
        f'<div style="font-size:0.79rem;color:#64748b;margin-top:0.15rem;">{subtitle}</div>'
        if subtitle else ""
    )
    return f"""
<div style="background:#fff;border-radius:14px;padding:1.2rem 1.4rem;
     box-shadow:0 1px 6px rgba(0,0,0,0.06),0 0 0 1px rgba(0,0,0,0.03);
     border-top:3px solid {accent};height:100%;">
  <div style="font-size:1.3rem;margin-bottom:0.45rem;">{icon}</div>
  <div style="font-size:0.72rem;font-weight:700;color:#94a3b8;
       text-transform:uppercase;letter-spacing:0.07em;">{label}</div>
  <div style="font-size:1.8rem;font-weight:800;color:#0f172a;
       line-height:1.15;margin-top:0.25rem;">{value}</div>
  {sub_html}
</div>"""


def _section_header(title: str, subtitle: str = "") -> str:
    sub = (
        f'<p style="margin:0.3rem 0 0;font-size:0.9rem;color:#64748b;line-height:1.5;">{subtitle}</p>'
        if subtitle else ""
    )
    return f"""
<div style="margin-bottom:1.2rem;">
  <h3 style="margin:0;font-size:1.15rem;font-weight:800;color:#0f172a;
      letter-spacing:-0.01em;">{title}</h3>
  {sub}
</div>"""


def _divider() -> None:
    st.markdown(
        '<hr style="border:none;border-top:1px solid #e2e8f0;margin:1.6rem 0;">',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str, int, int, float, bool, bool, bool]:
    with st.sidebar:
        st.markdown(
            """
<div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0 0.2rem;">
  <span style="font-size:1.6rem;">⚖️</span>
  <div>
    <div style="font-size:1.0rem;font-weight:800;color:#f1f5f9;
         letter-spacing:-0.01em;">AI Fairness</div>
    <div style="font-size:0.75rem;color:#94a3b8;font-weight:500;
         letter-spacing:0.03em;">AUDIT LAB</div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.7rem;font-weight:700;color:#94a3b8;'
            'text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.4rem;">Scenario</p>',
            unsafe_allow_html=True,
        )
        scenario = st.selectbox(
            "Scenario",
            options=list(SCENARIO_LABELS.keys()),
            format_func=lambda x: SCENARIO_LABELS.get(x, x),
            index=0,
            label_visibility="collapsed",
        )

        st.markdown(
            '<p style="font-size:0.7rem;font-weight:700;color:#94a3b8;'
            'text-transform:uppercase;letter-spacing:0.1em;margin:0.8rem 0 0.4rem;">Outcome to Audit</p>',
            unsafe_allow_html=True,
        )
        audit = st.selectbox(
            "Audit outcome",
            options=list(AUDIT_LABELS.keys()),
            format_func=lambda x: AUDIT_LABELS.get(x, x),
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.7rem;font-weight:700;color:#94a3b8;'
            'text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.4rem;">Simulation</p>',
            unsafe_allow_html=True,
        )
        n = st.slider(
            "Population size",
            min_value=1000, max_value=20000, value=7000, step=500,
            help="Larger populations give more stable estimates but take longer.",
        )
        seed = st.number_input(
            "Random seed",
            min_value=1, value=2026, step=1,
            help="Same seed always produces the same simulation.",
        )
        threshold = st.slider(
            "Decision threshold",
            min_value=0.30, max_value=0.80, value=0.52, step=0.01,
            help="AI score cutoff for a positive decision.",
        )

        st.markdown("---")
        run_bayes = st.checkbox(
            "Run Bayesian model",
            value=False,
            disabled=not HAS_BAYES_STACK,
            help="MCMC sampling for uncertainty intervals. Takes 1–2 min.",
        )
        show_technical = st.checkbox(
            "Show technical details",
            value=False,
            help="Raw model outputs for researchers and analysts.",
        )

        st.markdown("---")
        run = st.button("\u25b6\ufe0f  Run Audit", type="primary", use_container_width=True)

    return scenario, audit, int(n), int(seed), float(threshold), run_bayes, run, show_technical


# ── Welcome screen ────────────────────────────────────────────────────────────

def render_welcome() -> None:
    st.markdown(
        """
<div style="background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 55%,#0891b2 100%);
     border-radius:20px;padding:3rem 3.5rem 2.8rem;margin-bottom:2rem;
     color:white;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-50px;right:-50px;width:220px;height:220px;
       background:rgba(255,255,255,0.05);border-radius:50%;"></div>
  <div style="position:absolute;bottom:-70px;right:120px;width:160px;height:160px;
       background:rgba(255,255,255,0.04);border-radius:50%;"></div>
  <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
       letter-spacing:0.14em;opacity:0.75;margin-bottom:0.7rem;">
    ⚖️ &nbsp; Chamberlain's Law · AI Fairness Audit
  </div>
  <div style="font-size:2.6rem;font-weight:900;line-height:1.12;
       margin-bottom:1.1rem;letter-spacing:-0.02em;">
    Is your AI treating<br>everyone equally?
  </div>
  <div style="font-size:1.05rem;opacity:0.88;max-width:580px;line-height:1.65;">
    Simulate an AI decision-making system and run a rigorous fairness audit across
    <strong>protected groups</strong> — gender, race, and disability — to detect bias
    patterns and quantify inequality in outcomes.
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    steps = [
        ("1", "Choose a scenario", "Select the type of algorithm: fair, marginally biased, intersectionally biased, or proxy-driven.", "#6366f1"),
        ("2", "Set parameters", "Adjust population size, decision threshold, and random seed for reproducibility.", "#7c3aed"),
        ("3", "Run the audit", "Click Run Audit to simulate outcomes and compute fairness metrics across all groups.", "#0891b2"),
    ]
    cols = st.columns(3)
    for col, (num, title, desc, color) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
<div style="background:#fff;border-radius:14px;padding:1.5rem 1.4rem;
     box-shadow:0 2px 14px rgba(0,0,0,0.07);border-top:4px solid {color};height:100%;">
  <div style="width:2rem;height:2rem;border-radius:8px;
       background:{color}18;color:{color};font-weight:800;font-size:0.9rem;
       display:flex;align-items:center;justify-content:center;margin-bottom:0.9rem;">
    {num}
  </div>
  <div style="font-weight:700;font-size:0.98rem;color:#0f172a;margin-bottom:0.45rem;">{title}</div>
  <div style="font-size:0.855rem;color:#64748b;line-height:1.55;">{desc}</div>
</div>""",
                unsafe_allow_html=True,
            )


# ── Verdict banner ────────────────────────────────────────────────────────────

def render_verdict_banner(ias_score: float, scenario: str, audit: str) -> None:
    label, explanation = fairness_verdict(ias_score)
    cfg = {
        "Likely Fair": {
            "bg": "linear-gradient(135deg,#f0fdf4,#dcfce7)",
            "border": P["success"], "icon": "✅",
            "badge_bg": "#d1fae5", "badge_fg": "#065f46",
        },
        "Borderline": {
            "bg": "linear-gradient(135deg,#fffbeb,#fef3c7)",
            "border": P["warning"], "icon": "⚠️",
            "badge_bg": "#fde68a", "badge_fg": "#78350f",
        },
        "Likely Unfair": {
            "bg": "linear-gradient(135deg,#fff5f5,#fee2e2)",
            "border": P["danger"], "icon": "🚨",
            "badge_bg": "#fecaca", "badge_fg": "#7f1d1d",
        },
    }
    c = cfg[label]
    s_name = SCENARIO_LABELS.get(scenario, scenario)
    a_name = AUDIT_LABELS.get(audit, audit)

    st.markdown(
        f"""
<div style="background:{c['bg']};border:1px solid {c['border']}38;
     border-left:5px solid {c['border']};border-radius:16px;
     padding:1.5rem 1.8rem;margin-bottom:1.6rem;
     box-shadow:0 3px 16px {c['border']}18;">
  <div style="display:flex;align-items:flex-start;gap:0.8rem;">
    <span style="font-size:1.7rem;margin-top:0.1rem;">{c['icon']}</span>
    <div style="flex:1;">
      <div style="font-size:1.4rem;font-weight:900;color:#0f172a;
           letter-spacing:-0.01em;margin-bottom:0.35rem;">
        Fairness Verdict: &nbsp;
        <span style="background:{c['badge_bg']};color:{c['badge_fg']};
             padding:0.18rem 0.8rem;border-radius:20px;
             font-size:1.1rem;font-weight:700;">{label}</span>
      </div>
      <div style="font-size:0.84rem;color:#475569;margin-bottom:0.55rem;">
        <strong>Scenario:</strong> {s_name} &nbsp;&middot;&nbsp;
        <strong>Outcome audited:</strong> {a_name}
      </div>
      <div style="font-size:0.95rem;color:#334155;line-height:1.65;
           background:rgba(255,255,255,0.65);border-radius:10px;
           padding:0.65rem 0.9rem;">
        {explanation}
      </div>
    </div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


# ── Overview tab ──────────────────────────────────────────────────────────────

def render_overview_tab(
    summary: pd.DataFrame,
    marginal: pd.DataFrame,
    intersection: pd.DataFrame,
    union: pd.DataFrame,
    ias: dict,
    scenario: str,
    audit: str,
    show_technical: bool,
) -> None:
    ias_score = float(ias.get("IAS", 0.0))

    render_verdict_banner(ias_score, scenario, audit)

    st.markdown(
        _section_header(
            "Inequality Attribution Score (IAS)",
            "How much of the AI\u2019s decisions are explained by <em>who someone is</em> "
            "(gender, race, disability) versus <em>what they\u2019ve done or achieved</em>?",
        ),
        unsafe_allow_html=True,
    )

    col_gauge, col_cards = st.columns([2, 3], gap="large")

    with col_gauge:
        fig_gauge = make_semicircle_gauge(ias_score)
        st.pyplot(fig_gauge, clear_figure=True)

    with col_cards:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            ias_accent = (
                P["danger"] if ias_score > 0.15
                else P["warning"] if ias_score > 0.05
                else P["success"]
            )
            st.markdown(
                _card("🎯", "IAS Score", f"{ias_score:.2f}", "Closer to 0 = fairer", ias_accent),
                unsafe_allow_html=True,
            )
        with r1c2:
            st.markdown(
                _card("👤", "Identity-driven variance", f"{ias['var_identity']:.2f}",
                      "Gender, race, disability", P["primary"]),
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown(
                _card("📈", "Merit-driven variance", f"{ias['var_merit']:.2f}",
                      "Education, experience, etc.", P["muted"]),
                unsafe_allow_html=True,
            )
        with r2c2:
            st.markdown(
                """
<div style="background:#fff;border-radius:14px;padding:1.2rem 1.4rem;
     box-shadow:0 1px 6px rgba(0,0,0,0.06),0 0 0 1px rgba(0,0,0,0.03);height:100%;">
  <div style="font-size:0.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;
       letter-spacing:0.07em;margin-bottom:0.6rem;">IAS Zones</div>
  <div style="font-size:0.83rem;line-height:1.8;color:#374151;">
    <span style="color:#10b981;font-weight:700;">&lt; 5%</span> — Likely Fair<br>
    <span style="color:#f59e0b;font-weight:700;">5 – 15%</span> — Borderline<br>
    <span style="color:#ef4444;font-weight:700;">&gt; 15%</span> — Likely Unfair
  </div>
</div>""",
                unsafe_allow_html=True,
            )

    _divider()

    st.markdown(
        _section_header(
            "Population Snapshot",
            "Key statistics about the simulated population and AI decisions.",
        ),
        unsafe_allow_html=True,
    )
    cc1, cc2, cc3, cc4 = st.columns(4)
    pop_n    = int(summary.loc[0, "n"])
    y_rate   = float(summary.loc[0, "y_rate"])
    yhat_rate= float(summary.loc[0, "y_hat_rate"])
    fp_rate  = float(summary.loc[0, "false_positive_rate"])

    with cc1:
        st.markdown(_card("👥", "People simulated", f"{pop_n:,}", "Total population", P["primary"]), unsafe_allow_html=True)
    with cc2:
        st.markdown(_card("✅", "True positive rate", f"{y_rate:.2f}", "Who truly qualifies", P["success"]), unsafe_allow_html=True)
    with cc3:
        st.markdown(_card("🤖", "AI positive rate", f"{yhat_rate:.2f}", "Who AI approved", P["muted"]), unsafe_allow_html=True)
    with cc4:
        st.markdown(_card("❗", "False positive rate", f"{fp_rate:.2f}", "Wrongly flagged", P["danger"]), unsafe_allow_html=True)

    if show_technical:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        with st.expander("Raw population summary table"):
            st.dataframe(round_df(summary), use_container_width=True)


# ── Groups tab ────────────────────────────────────────────────────────────────

def render_groups_tab(
    marginal: pd.DataFrame,
    intersection: pd.DataFrame,
    union: pd.DataFrame,
    show_technical: bool,
) -> None:
    st.markdown(
        _section_header(
            "Group Fairness Analysis",
            "The <strong>fairness gap</strong> shows how many more or fewer positive outcomes each group received "
            "compared to what was expected based on qualifications alone. "
            "<span style='color:#10b981;font-weight:600;'>Green = above expected</span> &nbsp;&middot;&nbsp; "
            "<span style='color:#ef4444;font-weight:600;'>Red = below expected</span> &nbsp;&middot;&nbsp; "
            "Near zero = treated fairly.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:0.82rem;font-weight:700;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">'
        'Individual groups</div>',
        unsafe_allow_html=True,
    )
    fig1 = make_gap_chart(
        marginal, "group", "raw_gap",
        "Fairness Gap \u2014 Individual Groups",
        x_label_map=FRIENDLY_GROUP_NAMES,
    )
    st.pyplot(fig1, clear_figure=True)
    if show_technical:
        with st.expander("View data table \u2014 individual groups"):
            st.dataframe(friendly_marginal_table(marginal), use_container_width=True)

    _divider()

    st.markdown(
        '<div style="font-size:0.82rem;font-weight:700;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">'
        'Intersectional groups</div>',
        unsafe_allow_html=True,
    )
    st.caption("People can belong to more than one group simultaneously. This measures fairness at those intersections.")
    fig2 = make_gap_chart(
        intersection, "group", "raw_gap",
        "Fairness Gap \u2014 Intersectional Groups",
        x_label_map=FRIENDLY_GROUP_NAMES,
    )
    st.pyplot(fig2, clear_figure=True)
    if show_technical:
        with st.expander("View data table \u2014 intersectional groups"):
            st.dataframe(friendly_marginal_table(intersection), use_container_width=True)

    _divider()

    st.markdown(
        '<div style="font-size:0.82rem;font-weight:700;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">'
        'Union of protected groups</div>',
        unsafe_allow_html=True,
    )
    st.caption("What happens when we consider anyone who belongs to at least one protected group?")
    fig3 = make_gap_chart(
        union, "union", "fairness_gap",
        "Fairness Gap \u2014 Any Protected Group Combination",
        x_label_map=FRIENDLY_UNION_NAMES,
    )
    st.pyplot(fig3, clear_figure=True)
    if show_technical:
        with st.expander("View data table \u2014 union groups"):
            st.dataframe(friendly_union_table(union), use_container_width=True)


# ── Technical tab ─────────────────────────────────────────────────────────────

def render_technical_tab(
    freq: pd.DataFrame,
    ias: dict,
    df: pd.DataFrame,
    audit: str,
    run_bayes: bool,
    show_technical: bool,
) -> None:
    st.markdown(
        _section_header(
            "Frequentist Fairness Model",
            "A logistic regression showing how each factor \u2014 merit-based and identity-based \u2014 "
            "influences the AI\u2019s decisions. Positive coefficients increase the chance of a positive "
            "decision; negative ones decrease it. Statistically significant identity factors indicate bias.",
        ),
        unsafe_allow_html=True,
    )

    display_freq = freq.copy().rename(columns={
        "term": "Variable", "coef": "Coefficient", "odds_ratio": "Odds Ratio",
        "ci_low": "CI Lower", "ci_high": "CI Upper", "p_value": "P-value",
        "fit_method": "Fit Method", "significant_0_05": "Significant (p<0.05)",
    })
    st.dataframe(round_df(display_freq), use_container_width=True)

    _divider()

    st.markdown(
        _section_header("IAS \u2014 Component Breakdown"),
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
<div style="background:#fff;border-radius:14px;padding:1.3rem 1.5rem;
     box-shadow:0 1px 6px rgba(0,0,0,0.06),0 0 0 1px rgba(0,0,0,0.03);">
  <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:0.5rem 0;color:#64748b;font-weight:500;">IAS (overall score)</td>
      <td style="text-align:right;font-weight:800;font-size:1.05rem;color:#0f172a;">{ias['IAS']:.2f}</td>
    </tr>
    <tr style="border-bottom:1px solid #f1f5f9;">
      <td style="padding:0.5rem 0;color:#64748b;font-weight:500;">Identity-driven variance</td>
      <td style="text-align:right;font-weight:700;color:#6366f1;">{ias['var_identity']:.2f}</td>
    </tr>
    <tr>
      <td style="padding:0.5rem 0;color:#64748b;font-weight:500;">Merit-driven variance</td>
      <td style="text-align:right;font-weight:700;color:#10b981;">{ias['var_merit']:.2f}</td>
    </tr>
  </table>
</div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.info(
            "**How to read the IAS**\n\n"
            "- **< 5%** \u2014 identity attributes explain very little; algorithm likely fair\n"
            "- **5\u201315%** \u2014 moderate identity influence; worth monitoring\n"
            "- **> 15%** \u2014 strong identity influence; investigate for bias"
        )

    if run_bayes and HAS_BAYES_STACK:
        _divider()
        st.markdown(
            _section_header(
                "Bayesian Fairness Model",
                "MCMC sampling to estimate uncertainty ranges (highest-density intervals) around each coefficient.",
            ),
            unsafe_allow_html=True,
        )
        with st.spinner("Sampling posterior \u2014 this can take 1\u20132 minutes\u2026"):
            try:
                from src.bayesian_model import fit_bayesian_fairness_model
                import arviz as az

                z_col = f"Z_{audit}"
                idata, bayes_summary, sampling_method = fit_bayesian_fairness_model(
                    df, z_col=z_col, draws=500, tune=500,
                )
                bayes_ias = compute_bayesian_ias_interval(df, bayes_summary)
            except Exception as exc:
                st.error("Bayesian model failed for this configuration.")
                st.exception(exc)
                return

        st.caption(f"Sampling method: `{sampling_method}`")
        st.dataframe(round_df(bayes_summary), use_container_width=True)

        st.markdown("**Bayesian IAS Interval**")
        b = round_dict(bayes_ias)
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("IAS Point Estimate", f"{b.get('ias_point', 0):.2f}")
        bc2.metric("HDI Lower (95%)", f"{b.get('ias_hdi_low', 0):.2f}")
        bc3.metric("HDI Upper (95%)", f"{b.get('ias_hdi_high', 0):.2f}")

        fig_path = Path("results/figures") / f"streamlit_posterior_{audit}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        az.plot_forest(idata, var_names=["intercept", "beta"], combined=True, hdi_prob=0.95)
        plt.title("Bayesian posterior \u2014 factor estimates with 95% uncertainty range")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        st.image(
            str(fig_path),
            caption="Each dot = estimated effect size; bars = 95% uncertainty range. Dots far from 0 have strong effects.",
        )

    elif run_bayes and not HAS_BAYES_STACK:
        st.warning("PyMC/ArviZ not installed. Bayesian model cannot run in this environment.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _inject_css()

    scenario, audit, n, seed, threshold, run_bayes, run, show_technical = render_sidebar()

    if run:
        st.session_state.last_run = (scenario, audit, n, seed, threshold, run_bayes)

    if "last_run" not in st.session_state:
        render_welcome()
        return

    scenario, audit, n, seed, threshold, run_bayes = st.session_state.last_run

    with st.spinner("Running simulation and fairness audit \u2014 please wait\u2026"):
        try:
            df, summary, marginal, intersection, union, ias_raw, freq = run_interactive_pipeline(
                scenario=scenario,
                audit_outcome=audit,
                n=int(n),
                seed=int(seed),
                threshold=float(threshold),
            )
        except Exception as exc:
            st.error("The audit pipeline ran into an error. Try a different scenario or smaller population.")
            st.exception(exc)
            return

    ias = round_dict(ias_raw)

    tab_overview, tab_groups, tab_technical = st.tabs([
        "\U0001f4ca  Overview & Verdict",
        "\U0001f465  Group Fairness",
        "\U0001f52c  Technical Details",
    ])

    with tab_overview:
        render_overview_tab(summary, marginal, intersection, union, ias, scenario, audit, show_technical)

    with tab_groups:
        render_groups_tab(marginal, intersection, union, show_technical)

    with tab_technical:
        render_technical_tab(freq, ias, df, audit, run_bayes, show_technical)


if __name__ == "__main__":
    main()
