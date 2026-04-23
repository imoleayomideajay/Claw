"""Bayesian fairness model with PyMC and ArviZ."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


FEATURE_COLUMNS = [
    "education",
    "experience",
    "performance_score",
    "income_risk_score",
    "gender",
    "race",
    "disability",
    "gender_race",
    "gender_disability",
    "race_disability",
]


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["gender_race"] = x["gender"] * x["race"]
    x["gender_disability"] = x["gender"] * x["disability"]
    x["race_disability"] = x["race"] * x["disability"]
    return x


def fit_bayesian_fairness_model(
    df: pd.DataFrame,
    z_col: str,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    target_accept: float = 0.9,
) -> tuple[az.InferenceData, pd.DataFrame, str]:
    """Fit Bayesian logistic regression and return posterior summary table.

    Falls back to ADVI if NUTS sampling fails.
    """
    design = _design_matrix(df)
    X = design[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = design[z_col].to_numpy(dtype=int)

    x_mean = X.mean(axis=0)
    x_std = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    Xs = (X - x_mean) / x_std

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=Xs.shape[1])
        logits = intercept + pm.math.dot(Xs, beta)
        pm.Bernoulli("obs", logit_p=logits, observed=y)

        sampling_method = "NUTS"
        try:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=1,
                random_seed=42,
                target_accept=target_accept,
                progressbar=False,
            )
        except Exception:
            approx = pm.fit(method="advi", n=25000, random_seed=42, progressbar=False)
            idata = approx.sample(draws=draws)
            sampling_method = "ADVI fallback"

    summary = az.summary(idata, var_names=["intercept", "beta"], hdi_prob=0.95).reset_index()
    name_map = {f"beta[{i}]": FEATURE_COLUMNS[i] for i in range(len(FEATURE_COLUMNS))}
    summary["term"] = summary["index"].map(lambda x: name_map.get(x, x))
    return idata, summary, sampling_method


def save_posterior_forest_plot(idata: az.InferenceData, output_path: Path, title: str) -> None:
    """Create and save posterior forest plot."""
    ax = az.plot_forest(idata, var_names=["intercept", "beta"], combined=True, hdi_prob=0.95)
    fig = ax.ravel()[0].figure
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    fig.clf()
