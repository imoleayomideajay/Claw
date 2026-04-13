"""Inequality Attribution Score computations."""

from __future__ import annotations

import numpy as np
import pandas as pd


MERIT_COLS = ["education", "experience", "performance_score", "income_risk_score"]
IDENTITY_COLS = ["gender", "race", "disability", "gender_race", "gender_disability", "race_disability"]


def compute_ias(df: pd.DataFrame, coef_map: dict[str, float]) -> dict[str, float]:
    """Compute IAS from linear predictor decomposition using provided coefficients."""
    x = df.copy()
    x["gender_race"] = x["gender"] * x["race"]
    x["gender_disability"] = x["gender"] * x["disability"]
    x["race_disability"] = x["race"] * x["disability"]

    merit_comp = np.zeros(len(x))
    identity_comp = np.zeros(len(x))

    for c in MERIT_COLS:
        merit_comp += coef_map.get(c, 0.0) * x[c].values
    for c in IDENTITY_COLS:
        identity_comp += coef_map.get(c, 0.0) * x[c].values

    v_merit = float(np.var(merit_comp))
    v_identity = float(np.var(identity_comp))
    ias = v_identity / (v_identity + v_merit + 1e-12)
    return {"IAS": ias, "var_identity": v_identity, "var_merit": v_merit}


def compute_bayesian_ias_interval(df: pd.DataFrame, posterior_summary: pd.DataFrame) -> dict[str, float]:
    """Approximate IAS interval from posterior mean and 95% HDI bounds."""
    term_to_mean = dict(zip(posterior_summary["term"], posterior_summary["mean"]))
    term_to_low = dict(zip(posterior_summary["term"], posterior_summary["hdi_2.5%"], strict=False))
    term_to_high = dict(zip(posterior_summary["term"], posterior_summary["hdi_97.5%"], strict=False))

    point = compute_ias(df, term_to_mean)["IAS"]
    low = compute_ias(df, term_to_low)["IAS"]
    high = compute_ias(df, term_to_high)["IAS"]
    return {"ias_point": point, "ias_hdi_low": min(low, high), "ias_hdi_high": max(low, high)}
