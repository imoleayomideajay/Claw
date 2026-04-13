"""Fairness-metric utilities and scenario summaries."""

from __future__ import annotations

import pandas as pd

from src.baseline import compute_group_fairness_metrics


def marginal_and_intersection_metrics(df: pd.DataFrame, z_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return marginal and intersectional fairness metrics tables."""
    marginal_defs = {
        "gender=1": df["gender"] == 1,
        "race=1": df["race"] == 1,
        "disability=1": df["disability"] == 1,
    }
    inter_defs = {
        "gender=1,race=1": (df["gender"] == 1) & (df["race"] == 1),
        "gender=1,disability=1": (df["gender"] == 1) & (df["disability"] == 1),
        "race=1,disability=1": (df["race"] == 1) & (df["disability"] == 1),
        "gender=1,race=1,disability=1": (df["gender"] == 1) & (df["race"] == 1) & (df["disability"] == 1),
    }
    return (
        compute_group_fairness_metrics(df, z_col, marginal_defs),
        compute_group_fairness_metrics(df, z_col, inter_defs),
    )


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level summary table."""
    summary = {
        "n": len(df),
        "y_rate": df["Y"].mean(),
        "y_hat_rate": df["Y_hat"].mean(),
        "false_positive_rate": ((df["Y_hat"] == 1) & (df["Y"] == 0)).mean(),
        "false_negative_rate": ((df["Y_hat"] == 0) & (df["Y"] == 1)).mean(),
        "gender_share": df["gender"].mean(),
        "race_share": df["race"].mean(),
        "disability_share": df["disability"].mean(),
    }
    return pd.DataFrame([summary])
