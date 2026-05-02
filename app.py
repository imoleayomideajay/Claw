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
from src.simulate_data import build_audit_outcome, simulate_algorithm, simulate_ground_truth, simulate_population
from src.utils import SCENARIO_ORDER


HAS_BAYES_STACK = find_spec("pymc") is not None and find_spec("arviz") is not None

st.set_page_config(page_title="AI Fairness Audit Lab", layout="wide", page_icon="⚖️")

SCENARIO_LABELS = {
    "fair_algorithm": "Fair Algorithm",
    "biased_algorithm": "Biased Algorithm",
    "proxy_bias": "Proxy Bias",
    "intersectional_bias": "Intersectional Bias",
}

AUDIT_LABELS = {
    "positive_decision": "Positive Decisions (e.g. approvals, hires)",
    "false_positive": "False Positives (wrongly flagged as risky)",
    "false_negative": "False Negatives (missed true positives)",
}

FRIENDLY_GROUP_NAMES = {
    "gender=1": "Women",
    "race=1": "Minority Race",
    "disability=1": "People with Disabilities",
    "gender=1,race=1": "Women + Minority Race",
    "gender=1,disability=1": "Women + Disabled",
    "race=1,disability=1": "Minority Race + Disabled",
    "gender=1,race=1,disability=1": "Women + Minority Race + Disabled",
}

FRIENDLY_UNION_NAMES = {
    "A∪B": "Women or Minority Race",
    "A∪C": "Women or Disabled",
    "B∪C": "Minority Race or Disabled",
    "A∪B∪C": "Any Protected Group",
}


@st.cache_data(show_spinner=False)
def run_interactive_pipeline(
    scenario: str,
    audit_outcome: str,
    n: int,
    seed: int,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Run one scenario/outcome and return all key tables."""
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


def round_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    return df.round(decimals)


def round_dict(d: dict, decimals: int = 2) -> dict:
    return {k: round(float(v), decimals) for k, v in d.items()}


def fairness_verdict(ias_score: float) -> tuple[str, str, str]:
    """Return (label, color, explanation) based on IAS score."""
    if ias_score < 0.05:
        return "Likely Fair", "normal", "Identity-related factors (gender, race, disability) account for very little of the outcome variation. This looks fair."
    elif ias_score < 0.15:
        return "Borderline", "off", "Some outcome variation is tied to identity. Worth monitoring, but not a major red flag."
    else:
        return "Likely Unfair", "inverse", "A significant portion of outcomes appear driven by identity rather than merit. This warrants investigation."


def gap_verdict(gap: float, threshold: float = 5.0) -> str:
    if abs(gap) < threshold:
        return "Within range"
    return "Needs attention" if gap < 0 else "Above average"


def make_gap_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, x_label_map: dict | None = None) -> plt.Figure:
    """Color-coded fairness gap bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = df[x_col].map(x_label_map).fillna(df[x_col]) if x_label_map else df[x_col]
    values = df[y_col].values
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in values]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=1.2, linestyle="--")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.3 if val >= 0 else -0.5),
            f"{val:.2f}",
            ha="center", va="bottom" if val >= 0 else "top",
            fontsize=9, fontweight="bold",
        )

    red_patch = mpatches.Patch(color="#d62728", label="Below expected (unfavorable)")
    green_patch = mpatches.Patch(color="#2ca02c", label="Above expected (favorable)")
    ax.legend(handles=[red_patch, green_patch], fontsize=8, loc="upper right")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Gap (Observed − Expected)", fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def render_sidebar() -> tuple[str, str, int, int, float, bool, bool]:
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/scales.png", width=60)
        st.title("Audit Settings")
        st.markdown("---")

        st.markdown("**Which scenario should we test?**")
        scenario_display = st.selectbox(
            "Scenario",
            options=list(SCENARIO_LABELS.keys()),
            format_func=lambda x: SCENARIO_LABELS.get(x, x),
            index=0,
            label_visibility="collapsed",
            help="Choose the type of algorithm to simulate.",
        )

        st.markdown("**What outcome are we auditing?**")
        audit_display = st.selectbox(
            "Audit outcome",
            options=list(AUDIT_LABELS.keys()),
            format_func=lambda x: AUDIT_LABELS.get(x, x),
            index=0,
            label_visibility="collapsed",
            help="The decision or error type to measure fairness on.",
        )

        st.markdown("---")
        st.markdown("**Simulation settings**")

        n = st.slider(
            "Population size",
            min_value=1000, max_value=20000, value=7000, step=500,
            help="How many people to simulate. Larger = more reliable results, slower.",
        )
        seed = st.number_input(
            "Random seed",
            min_value=1, value=2026, step=1,
            help="Controls randomness. Same seed = same results.",
        )
        threshold = st.slider(
            "Decision threshold",
            min_value=0.30, max_value=0.80, value=0.52, step=0.01,
            help="Score cutoff above which the AI makes a positive decision.",
        )

        st.markdown("---")
        run_bayes = st.checkbox(
            "Run advanced Bayesian model",
            value=False,
            disabled=not HAS_BAYES_STACK,
            help="Uses statistical sampling to estimate uncertainty ranges. Takes longer.",
        )
        show_technical = st.checkbox(
            "Show technical details",
            value=False,
            help="Display raw model outputs for researchers and analysts.",
        )

        st.markdown("---")
        run = st.button("Run Audit", type="primary", use_container_width=True)

    return scenario_display, audit_display, int(n), int(seed), float(threshold), run_bayes, run, show_technical


def render_welcome():
    st.markdown(
        """
        <div style="padding: 2rem; background: linear-gradient(135deg, #f0f4ff, #e8f4e8);
             border-radius: 12px; margin-bottom: 1.5rem;">
            <h2 style="margin:0; color:#1a237e;">⚖️ AI Fairness Audit Lab</h2>
            <p style="color:#444; margin-top:0.5rem; font-size:1.05rem;">
                This tool simulates an AI decision-making system and checks whether it treats
                different groups of people <strong>equally and fairly</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1 — Choose a scenario**\nPick the type of AI algorithm you want to test in the sidebar.")
    with col2:
        st.info("**Step 2 — Set parameters**\nAdjust the population size and decision threshold.")
    with col3:
        st.info("**Step 3 — Run the audit**\nClick **Run Audit** to see the fairness results.")


def render_verdict_banner(ias_score: float, scenario: str, audit: str):
    label, state, explanation = fairness_verdict(ias_score)
    colors = {
        "Likely Fair": ("#e8f5e9", "#2e7d32", "✅"),
        "Borderline": ("#fff8e1", "#f57f17", "⚠️"),
        "Likely Unfair": ("#ffebee", "#c62828", "🚨"),
    }
    bg, fg, icon = colors[label]
    scenario_name = SCENARIO_LABELS.get(scenario, scenario)
    audit_name = AUDIT_LABELS.get(audit, audit)

    st.markdown(
        f"""
        <div style="padding:1.2rem 1.5rem; background:{bg}; border-left: 5px solid {fg};
             border-radius:8px; margin-bottom:1.2rem;">
            <h3 style="margin:0; color:{fg};">{icon} Fairness Verdict: {label}</h3>
            <p style="margin:0.4rem 0 0 0; color:#333;">
                <strong>Scenario:</strong> {scenario_name} &nbsp;|&nbsp;
                <strong>Outcome audited:</strong> {audit_name}
            </p>
            <p style="margin:0.4rem 0 0 0; color:#555;">{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ias_gauge(ias_score: float):
    pct = min(ias_score * 100, 100)
    color = "#2ca02c" if pct < 5 else ("#f0a500" if pct < 15 else "#d62728")
    st.markdown(
        f"""
        <div style="margin-bottom:0.5rem;">
            <span style="font-size:0.9rem; color:#555;">
                0% = Fully fair &nbsp;&nbsp;&nbsp; 100% = Fully identity-driven
            </span>
        </div>
        <div style="background:#e0e0e0; border-radius:20px; height:22px; width:100%;">
            <div style="width:{pct:.1f}%; background:{color}; height:22px; border-radius:20px;
                 display:flex; align-items:center; justify-content:flex-end; padding-right:8px;">
                <span style="color:white; font-size:0.8rem; font-weight:bold;">{pct:.1f}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def friendly_marginal_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["group"] = out["group"].map(FRIENDLY_GROUP_NAMES).fillna(out["group"])
    out = out.rename(columns={
        "group": "Group",
        "n_group": "Group Size",
        "observed_count": "Actual Outcomes",
        "expected_count": "Expected Outcomes",
        "raw_gap": "Fairness Gap",
        "standardised_gap": "Gap (Standardised)",
        "rate_disparity": "Rate Difference vs Others",
    })
    return round_df(out)


def friendly_union_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["union"] = out["union"].map(FRIENDLY_UNION_NAMES).fillna(out["union"])
    out = out.rename(columns={
        "union": "Group Combination",
        "prevalence": "Share of Population",
        "n_union": "Group Size",
        "n_complement": "Everyone Else",
        "observed_count": "Actual Outcomes",
        "expected_count": "Expected Outcomes",
        "fairness_gap": "Fairness Gap",
        "union_minus_complement_rate": "Rate vs Non-Group",
    })
    return round_df(out)


def render_overview_tab(summary, marginal, intersection, union, ias, scenario, audit, show_technical):
    ias_score = ias.get("IAS", 0.0)

    render_verdict_banner(ias_score, scenario, audit)

    st.markdown("### Inequality Attribution Score (IAS)")
    st.markdown(
        "The IAS answers: **How much of the AI's decisions are explained by who someone is "
        "(gender, race, disability) vs. what they've done or achieved?**"
    )
    render_ias_gauge(ias_score)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Identity-driven variation",
        f"{ias['var_identity']:.2f}",
        help="How much outcome variation is explained by protected identity attributes.",
    )
    c2.metric(
        "Merit-driven variation",
        f"{ias['var_merit']:.2f}",
        help="How much outcome variation is explained by legitimate factors (education, experience, etc.).",
    )
    c3.metric(
        "IAS Score",
        f"{ias_score:.2f}",
        help="Ratio of identity-driven to total variation. Closer to 0 is fairer.",
    )

    st.markdown("---")
    st.markdown("### Snapshot: How this population was simulated")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("People simulated", f"{int(summary.loc[0, 'n']):,}", help="Total population in the simulation.")
    c2.metric("Actual positive rate", f"{summary.loc[0, 'y_rate']:.2f}", help="Share of people who truly qualify for a positive outcome.")
    c3.metric("AI positive rate", f"{summary.loc[0, 'y_hat_rate']:.2f}", help="Share of people the AI gave a positive outcome to.")
    c4.metric(
        "False positive rate",
        f"{summary.loc[0, 'false_positive_rate']:.2f}",
        help="Share incorrectly flagged as positive.",
    )

    if show_technical:
        with st.expander("Raw population summary table"):
            st.dataframe(round_df(summary), use_container_width=True)


def render_groups_tab(marginal, intersection, union, show_technical):
    st.markdown(
        """
        ### How fairly did the AI treat each group?
        The charts below show the **fairness gap** for each group — the difference between
        how many positive outcomes the group actually received versus how many they were expected
        to receive based on their qualifications alone.

        - **Red bars** = group received *fewer* outcomes than expected (potential disadvantage)
        - **Green bars** = group received *more* outcomes than expected
        - **Bars near zero** = treated fairly
        """
    )

    st.markdown("#### Individual groups")
    friendly_m = friendly_marginal_table(marginal)
    fig1 = make_gap_chart(
        marginal, "group", "raw_gap",
        "Fairness Gap by Group",
        x_label_map=FRIENDLY_GROUP_NAMES,
    )
    st.pyplot(fig1, clear_figure=True)
    if show_technical:
        with st.expander("View group data table"):
            st.dataframe(friendly_m, use_container_width=True)

    st.markdown("#### Combinations of groups (intersectional)")
    st.caption("People can belong to more than one group. This checks fairness at the intersection.")
    fig2 = make_gap_chart(
        intersection, "group", "raw_gap",
        "Fairness Gap — Intersectional Groups",
        x_label_map=FRIENDLY_GROUP_NAMES,
    )
    st.pyplot(fig2, clear_figure=True)
    if show_technical:
        with st.expander("View intersectional data table"):
            st.dataframe(friendly_marginal_table(intersection), use_container_width=True)

    st.markdown("#### Union of any protected group")
    st.caption("What if we consider anyone who belongs to at least one protected group?")
    fig3 = make_gap_chart(
        union, "union", "fairness_gap",
        "Fairness Gap — Any Protected Group Combination",
        x_label_map=FRIENDLY_UNION_NAMES,
    )
    st.pyplot(fig3, clear_figure=True)
    if show_technical:
        with st.expander("View union data table"):
            st.dataframe(friendly_union_table(union), use_container_width=True)


def render_technical_tab(freq, ias, show_technical):
    st.markdown("### Frequentist Fairness Model")
    st.markdown(
        "This logistic regression model estimates how each factor — merit-based and identity-based — "
        "contributes to the AI's decisions. A positive coefficient means that factor increases the "
        "chance of a positive decision; negative means it decreases it."
    )
    display_freq = freq.copy().rename(columns={
        "term": "Variable",
        "coef": "Coefficient",
        "odds_ratio": "Odds Ratio",
        "ci_low": "CI Lower",
        "ci_high": "CI Upper",
        "p_value": "P-value",
        "fit_method": "Fit Method",
        "significant_0_05": "Significant (p<0.05)",
    })
    st.dataframe(round_df(display_freq), use_container_width=True)

    st.markdown("---")
    st.markdown("### Inequality Attribution Score — Detail")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            | Component | Value |
            |---|---|
            | IAS (overall score) | **{ias['IAS']:.2f}** |
            | Identity-driven variance | {ias['var_identity']:.2f} |
            | Merit-driven variance | {ias['var_merit']:.2f} |
            """
        )
    with col2:
        st.info(
            "**How to read IAS:**\n\n"
            "- **< 5%** — identity attributes explain very little; algorithm is likely fair\n"
            "- **5–15%** — moderate identity influence; worth monitoring\n"
            "- **> 15%** — strong identity influence; investigate for bias"
        )


def main() -> None:
    """Render Streamlit dashboard."""

    scenario_display, audit_display, n, seed, threshold, run_bayes, run, show_technical = render_sidebar()

    if run:
        st.session_state.last_run = (scenario_display, audit_display, n, seed, threshold, run_bayes)

    if "last_run" not in st.session_state:
        render_welcome()
        return

    scenario, audit, n, seed, threshold, run_bayes = st.session_state.last_run

    with st.spinner("Running simulation and fairness audit — this may take a few seconds..."):
        try:
            df, summary, marginal, intersection, union, ias, freq = run_interactive_pipeline(
                scenario=scenario,
                audit_outcome=audit,
                n=int(n),
                seed=int(seed),
                threshold=float(threshold),
            )
        except Exception as exc:
            st.error("The audit pipeline ran into an error. Try a different scenario or smaller population size.")
            st.exception(exc)
            return

    ias = round_dict(ias)

    tab_overview, tab_groups, tab_technical = st.tabs([
        "📊 Overview & Verdict",
        "👥 Group Fairness",
        "🔬 Technical Details",
    ])

    with tab_overview:
        render_overview_tab(summary, marginal, intersection, union, ias, scenario, audit, show_technical)

    with tab_groups:
        render_groups_tab(marginal, intersection, union, show_technical)

    with tab_technical:
        render_technical_tab(freq, ias, show_technical)

        if run_bayes and HAS_BAYES_STACK:
            st.markdown("---")
            st.markdown("### Bayesian Fairness Model")
            st.caption("Uses statistical sampling (MCMC) to estimate uncertainty ranges around each coefficient.")
            with st.spinner("Sampling posterior — this can take 1–2 minutes..."):
                try:
                    from src.bayesian_model import fit_bayesian_fairness_model
                    import arviz as az

                    z_col = f"Z_{audit}"
                    idata, bayes_summary, sampling_method = fit_bayesian_fairness_model(df, z_col=z_col, draws=500, tune=500)
                    bayes_ias = compute_bayesian_ias_interval(df, bayes_summary)
                except Exception as exc:
                    st.error("Bayesian model failed for this configuration.")
                    st.exception(exc)
                    return

            st.markdown(f"Sampling method: `{sampling_method}`")
            st.dataframe(round_df(bayes_summary), use_container_width=True)

            st.markdown("**Bayesian IAS Interval**")
            b = round_dict(bayes_ias)
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("IAS Point Estimate", f"{b.get('ias_point', 0):.2f}")
            bc2.metric("HDI Lower (95%)", f"{b.get('ias_hdi_low', 0):.2f}")
            bc3.metric("HDI Upper (95%)", f"{b.get('ias_hdi_high', 0):.2f}")

            fig_path = Path("results/figures") / f"streamlit_{scenario}_{audit}_posterior.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            az.plot_forest(idata, var_names=["intercept", "beta"], combined=True, hdi_prob=0.95)
            plt.title("Bayesian posterior: factor estimates with 95% uncertainty range")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()
            st.image(str(fig_path), caption="Each dot is the estimated effect; bars show uncertainty. Dots far from 0 have strong effects.")

        elif run_bayes and not HAS_BAYES_STACK:
            st.warning("PyMC/ArviZ not available. Bayesian model cannot run in this environment.")


if __name__ == "__main__":
    main()
