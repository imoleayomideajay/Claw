"""Chamberlain expected-outcome baseline for subgroup fairness metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


LEGITIMATE_FEATURES = ["education", "experience", "performance_score", "income_risk_score"]


def fit_chamberlain_baseline(df: pd.DataFrame, z_col: str) -> tuple[sm.GLM, pd.DataFrame]:
    """Fit baseline logistic model with legitimate covariates only and return pi0."""
    X = sm.add_constant(df[LEGITIMATE_FEATURES], has_constant="add")
    model = sm.GLM(df[z_col], X, family=sm.families.Binomial())
    result = model.fit()
    out = df.copy()
    out["pi0"] = result.predict(X)
    return result, out


def _group_metrics(df: pd.DataFrame, mask: pd.Series, z_col: str) -> dict[str, float]:
    n_u = int(mask.sum())
    o_u = float(df.loc[mask, z_col].sum())
    e_u = float(df.loc[mask, "pi0"].sum())
    g_u = o_u - e_u

    var_u = float((df.loc[mask, "pi0"] * (1 - df.loc[mask, "pi0"])).sum())
    std_gap = g_u / np.sqrt(var_u + 1e-8)

    p_u = float(df.loc[mask, z_col].mean()) if n_u > 0 else np.nan
    p_c = float(df.loc[~mask, z_col].mean()) if (~mask).sum() > 0 else np.nan
    delta = p_u - p_c

    return {
        "n_group": n_u,
        "observed_count": o_u,
        "expected_count": e_u,
        "raw_gap": g_u,
        "standardised_gap": std_gap,
        "rate_disparity": delta,
    }


def compute_group_fairness_metrics(
    df: pd.DataFrame,
    z_col: str,
    group_definitions: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Compute observed/expected fairness metrics for groups or intersections."""
    if group_definitions is None:
        group_definitions = {
            "gender=1": df["gender"] == 1,
            "race=1": df["race"] == 1,
            "disability=1": df["disability"] == 1,
            "gender=0": df["gender"] == 0,
            "race=0": df["race"] == 0,
            "disability=0": df["disability"] == 0,
            "gender=1,race=1": (df["gender"] == 1) & (df["race"] == 1),
            "gender=1,disability=1": (df["gender"] == 1) & (df["disability"] == 1),
            "race=1,disability=1": (df["race"] == 1) & (df["disability"] == 1),
            "gender=1,race=1,disability=1": (df["gender"] == 1) & (df["race"] == 1) & (df["disability"] == 1),
        }

    rows = []
    for name, mask in group_definitions.items():
        metrics = _group_metrics(df, mask, z_col)
        metrics["group"] = name
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
