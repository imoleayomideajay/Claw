"""Publication-quality plots for fairness audit outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_observed_expected(metrics: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(metrics))
    ax.bar(x, metrics["observed_count"], width=0.4, label="Observed", alpha=0.8)
    ax.bar([i + 0.4 for i in x], metrics["expected_count"], width=0.4, label="Expected", alpha=0.8)
    ax.set_xticks([i + 0.2 for i in x], metrics["group"], rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.legend()
    _save(fig, output_path)


def plot_fairness_gap(metrics: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="black", linewidth=1)
    ax.bar(metrics["group"], metrics["raw_gap"], color="#1f77b4")
    ax.set_xticklabels(metrics["group"], rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Observed - Expected")
    _save(fig, output_path)


def plot_union_gap(union_metrics: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(0, color="black", linewidth=1)
    ax.bar(union_metrics["union"], union_metrics["union_minus_complement_rate"], color="#ff7f0e")
    ax.set_title(title)
    ax.set_ylabel("Rate(U) - Rate(U^c)")
    _save(fig, output_path)


def plot_ias_across_scenarios(ias_table: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(ias_table["scenario"], ias_table["ias_point"], marker="o", linewidth=2)
    if {"ias_hdi_low", "ias_hdi_high"}.issubset(ias_table.columns):
        ax.fill_between(
            ias_table["scenario"],
            ias_table["ias_hdi_low"],
            ias_table["ias_hdi_high"],
            alpha=0.2,
            color="tab:blue",
            label="Approx. interval",
        )
        ax.legend()
    ax.set_ylim(0, 1)
    ax.set_ylabel("IAS")
    ax.set_title("IAS comparison across scenarios")
    ax.set_xticklabels(ias_table["scenario"], rotation=25, ha="right")
    _save(fig, output_path)


def plot_error_rate_disparities(df: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for group, mask in {
        "gender=1": df["gender"] == 1,
        "race=1": df["race"] == 1,
        "disability=1": df["disability"] == 1,
    }.items():
        fp = ((df["Y_hat"] == 1) & (df["Y"] == 0) & mask).mean()
        fn = ((df["Y_hat"] == 0) & (df["Y"] == 1) & mask).mean()
        rows.append({"group": group, "false_positive": fp, "false_negative": fn})

    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(d["group"], d["false_positive"], alpha=0.8, label="FP")
    ax.bar(d["group"], d["false_negative"], alpha=0.8, bottom=d["false_positive"], label="FN")
    ax.set_title("Error-rate disparities")
    ax.set_ylabel("Rate")
    ax.legend()
    _save(fig, output_path)
