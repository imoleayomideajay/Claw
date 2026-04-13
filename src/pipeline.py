"""End-to-end execution pipeline for all fairness scenarios."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.baseline import fit_chamberlain_baseline
from src.bayesian_model import fit_bayesian_fairness_model, save_posterior_forest_plot
from src.fairness_metrics import dataset_summary, marginal_and_intersection_metrics
from src.frequentist_model import fit_frequentist_fairness_model
from src.ias import compute_bayesian_ias_interval, compute_ias
from src.inclusion_exclusion import compute_union_metrics_pie
from src.plots import (
    plot_error_rate_disparities,
    plot_fairness_gap,
    plot_ias_across_scenarios,
    plot_observed_expected,
    plot_union_gap,
)
from src.simulate_data import SimulationConfig, build_audit_outcome, simulate_algorithm, simulate_ground_truth, simulate_population
from src.tables import export_all_tables
from src.utils import SCENARIO_ORDER, ensure_directories


AUDIT_OUTCOMES = ["positive_decision", "false_positive", "false_negative"]


def generate_all_figures(
    scenario: str,
    marginal_metrics: pd.DataFrame,
    intersection_metrics: pd.DataFrame,
    union_metrics: pd.DataFrame,
    scenario_df: pd.DataFrame,
    figure_dir: Path,
    bayes_idata,
) -> None:
    """Create all required per-scenario figures."""
    plot_observed_expected(
        marginal_metrics,
        f"Observed vs Expected ({scenario})",
        figure_dir / f"{scenario}_observed_vs_expected.png",
    )
    plot_fairness_gap(
        intersection_metrics,
        f"Subgroup fairness gaps ({scenario})",
        figure_dir / f"{scenario}_fairness_gap.png",
    )
    plot_union_gap(
        union_metrics,
        f"Union vs complement gap ({scenario})",
        figure_dir / f"{scenario}_union_gap.png",
    )
    save_posterior_forest_plot(
        bayes_idata,
        figure_dir / f"{scenario}_posterior_forest.png",
        title=f"Posterior forest plot ({scenario})",
    )
    plot_error_rate_disparities(scenario_df, figure_dir / f"{scenario}_error_rate_disparities.png")


def run_scenario(
    scenario: str,
    config: SimulationConfig,
    data_dir: Path,
    figure_dir: Path,
    table_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Run full pipeline for one scenario and persist outputs."""
    pop = simulate_population(config.n, config.seed)
    truth = simulate_ground_truth(pop, config.seed)
    scenario_df = simulate_algorithm(truth, scenario, config.seed, config.base_threshold)

    scenario_df.to_csv(data_dir / f"{scenario}_dataset.csv", index=False)

    scenario_results: dict[str, pd.DataFrame] = {}
    for audit in AUDIT_OUTCOMES:
        z_col = f"Z_{audit}"
        scenario_df[z_col] = build_audit_outcome(scenario_df, audit)

        _, with_pi0 = fit_chamberlain_baseline(scenario_df, z_col)
        marginal_metrics, intersection_metrics = marginal_and_intersection_metrics(with_pi0, z_col)
        union_metrics = compute_union_metrics_pie(with_pi0, z_col)

        freq_fit, freq_table = fit_frequentist_fairness_model(with_pi0, z_col)
        coef_map = dict(freq_fit.params)
        ias_point = compute_ias(with_pi0, coef_map)

        bayes_idata, bayes_summary, sampling_method = fit_bayesian_fairness_model(with_pi0, z_col)
        ias_bayes = compute_bayesian_ias_interval(with_pi0, bayes_summary)

        prefix = f"{scenario}_{audit}"
        tables = {
            f"{prefix}_population_summary": dataset_summary(with_pi0),
            f"{prefix}_marginal_metrics": marginal_metrics,
            f"{prefix}_intersection_metrics": intersection_metrics,
            f"{prefix}_union_metrics": union_metrics,
            f"{prefix}_frequentist_summary": freq_table,
            f"{prefix}_bayesian_summary": bayes_summary,
            f"{prefix}_ias_summary": pd.DataFrame(
                [
                    {
                        "scenario": scenario,
                        "audit_outcome": audit,
                        "sampling_method": sampling_method,
                        **ias_point,
                        **ias_bayes,
                    }
                ]
            ),
        }
        export_all_tables(tables, table_dir)

        if audit == "positive_decision":
            generate_all_figures(
                scenario,
                marginal_metrics,
                intersection_metrics,
                union_metrics,
                with_pi0,
                figure_dir,
                bayes_idata,
            )

        scenario_results[f"{prefix}_ias_summary"] = tables[f"{prefix}_ias_summary"]

    return scenario_results


def run_all_scenarios(
    base_dir: Path,
    n: int = 8000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all required scenarios and return scenario-level IAS summary."""
    data_dir = base_dir / "data"
    figure_dir = base_dir / "results" / "figures"
    table_dir = base_dir / "results" / "tables"
    ensure_directories([data_dir, figure_dir, table_dir])

    config = SimulationConfig(n=n, seed=seed)

    ias_rows = []
    for i, scenario in enumerate(SCENARIO_ORDER):
        scenario_seeded = SimulationConfig(n=n, seed=seed + 100 * i)
        scenario_out = run_scenario(scenario, scenario_seeded, data_dir, figure_dir, table_dir)
        ias_rows.append(scenario_out[f"{scenario}_positive_decision_ias_summary"])

    ias_table = pd.concat(ias_rows, ignore_index=True)
    ias_table.to_csv(table_dir / "scenario_comparison_summary.csv", index=False)
    plot_ias_across_scenarios(ias_table, figure_dir / "ias_comparison_across_scenarios.png")
    return ias_table
