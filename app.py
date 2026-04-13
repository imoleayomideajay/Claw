"""Streamlit app for interactive AI fairness auditing simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.baseline import fit_chamberlain_baseline
from src.bayesian_model import fit_bayesian_fairness_model
from src.fairness_metrics import dataset_summary, marginal_and_intersection_metrics
from src.frequentist_model import fit_frequentist_fairness_model
from src.ias import compute_bayesian_ias_interval, compute_ias
from src.inclusion_exclusion import compute_union_metrics_pie
from src.simulate_data import build_audit_outcome, simulate_algorithm, simulate_ground_truth, simulate_population
from src.utils import SCENARIO_ORDER


st.set_page_config(page_title="AI Fairness Audit Lab", layout="wide")


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


def make_gap_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> plt.Figure:
    """Simple fairness-gap bar chart for Streamlit rendering."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="black", linewidth=1)
    ax.bar(df[x_col], df[y_col], color="#1f77b4")
    ax.set_title(title)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def main() -> None:
    """Render Streamlit dashboard."""
    st.title("Extending Chamberlain’s Law for AI Fairness")
    st.caption("Interactive simulation + fairness auditing dashboard")

    with st.sidebar:
        st.header("Simulation controls")
        scenario = st.selectbox("Scenario", SCENARIO_ORDER, index=0)
        audit = st.selectbox("Audit outcome", ["positive_decision", "false_positive", "false_negative"], index=0)
        n = st.slider("Population size", min_value=1000, max_value=20000, value=7000, step=500)
        seed = st.number_input("Random seed", min_value=1, value=2026, step=1)
        threshold = st.slider("Decision threshold", min_value=0.30, max_value=0.80, value=0.52, step=0.01)
        run_bayes = st.checkbox("Run Bayesian model", value=False, help="Can take longer for large n.")
        run = st.button("Run audit", type="primary")

    if run:
        st.session_state.last_run = (scenario, audit, int(n), int(seed), float(threshold), run_bayes)

    if "last_run" not in st.session_state:
        st.info("Configure parameters in the sidebar, then click **Run audit**.")
        return

    scenario, audit, n, seed, threshold, run_bayes = st.session_state.last_run

    with st.spinner("Running simulation and fairness audit..."):
        try:
            df, summary, marginal, intersection, union, ias, freq = run_interactive_pipeline(
                scenario=scenario,
                audit_outcome=audit,
                n=int(n),
                seed=int(seed),
                threshold=float(threshold),
            )
        except Exception as exc:
            st.error("Pipeline failed. Try a different outcome/scenario or smaller N.")
            st.exception(exc)
            return

    c1, c2, c3 = st.columns(3)
    c1.metric("N", f"{int(summary.loc[0, 'n']):,}")
    c2.metric("Y rate", f"{summary.loc[0, 'y_rate']:.3f}")
    c3.metric("Y-hat rate", f"{summary.loc[0, 'y_hat_rate']:.3f}")

    st.subheader("Population summary")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Marginal fairness metrics")
    st.dataframe(marginal, use_container_width=True)

    st.subheader("Intersectional fairness metrics")
    st.dataframe(intersection, use_container_width=True)

    st.subheader("Union-level metrics (inclusion–exclusion)")
    st.dataframe(union, use_container_width=True)

    st.subheader("Frequentist fairness model")
    st.dataframe(freq, use_container_width=True)

    st.subheader("Inequality Attribution Score (IAS)")
    st.json(ias)

    p1, p2 = st.columns(2)
    with p1:
        st.pyplot(make_gap_plot(marginal, "group", "raw_gap", "Marginal fairness gaps"), clear_figure=True)
    with p2:
        st.pyplot(make_gap_plot(intersection, "group", "raw_gap", "Intersectional fairness gaps"), clear_figure=True)

    st.pyplot(make_gap_plot(union, "union", "fairness_gap", "Union fairness gaps"), clear_figure=True)

    if run_bayes:
        st.subheader("Bayesian fairness model")
        with st.spinner("Sampling posterior..."):
            try:
                z_col = f"Z_{audit}"
                idata, bayes_summary, sampling_method = fit_bayesian_fairness_model(df, z_col=z_col, draws=500, tune=500)
                bayes_ias = compute_bayesian_ias_interval(df, bayes_summary)
            except Exception as exc:
                st.error("Bayesian model failed for this configuration.")
                st.exception(exc)
                return
        st.write(f"Sampling method: `{sampling_method}`")
        st.dataframe(bayes_summary, use_container_width=True)
        st.json(bayes_ias)

        fig_path = Path("results/figures") / f"streamlit_{scenario}_{audit}_posterior.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        import arviz as az

        az.plot_forest(idata, var_names=["intercept", "beta"], combined=True, hdi_prob=0.95)
        plt.title("Bayesian posterior forest plot")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        st.image(str(fig_path), caption="Posterior forest plot")


if __name__ == "__main__":
    main()
