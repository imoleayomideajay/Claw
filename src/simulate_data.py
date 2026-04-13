"""Synthetic data generation for fairness-auditing scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils import logistic


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for simulation design."""

    n: int = 8000
    seed: int = 42
    base_threshold: float = 0.52


def simulate_population(n: int, seed: int) -> pd.DataFrame:
    """Generate a synthetic population with dependent protected attributes and merit features."""
    rng = np.random.default_rng(seed)
    latent_social = rng.normal(0, 1, n)

    gender_prob = logistic(-0.25 + 0.65 * latent_social)
    gender = rng.binomial(1, gender_prob)

    race_prob = logistic(-0.15 + 0.75 * latent_social + 0.35 * gender)
    race = rng.binomial(1, race_prob)

    disability_prob = logistic(-1.10 + 0.45 * latent_social + 0.30 * race)
    disability = rng.binomial(1, disability_prob)

    education = np.clip(12 + 1.7 * latent_social + rng.normal(0, 1.2, n), 8, 21)
    experience = np.clip(8 + 2.8 * latent_social + 0.3 * education + rng.normal(0, 2.0, n), 0, 45)
    performance = np.clip(55 + 8.0 * latent_social + 1.5 * education + 0.4 * experience + rng.normal(0, 10, n), 0, 100)
    income_risk = np.clip(620 + 35 * latent_social + 3 * education + 0.5 * experience + rng.normal(0, 40, n), 300, 850)

    return pd.DataFrame(
        {
            "gender": gender,
            "race": race,
            "disability": disability,
            "education": education,
            "experience": experience,
            "performance_score": performance,
            "income_risk_score": income_risk,
        }
    )


def simulate_ground_truth(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Generate true outcome using legitimate covariates only."""
    rng = np.random.default_rng(seed + 1)
    linear = (
        -7.0
        + 0.19 * df["education"].values
        + 0.055 * df["experience"].values
        + 0.07 * df["performance_score"].values
        + 0.0055 * df["income_risk_score"].values
    )
    p_true = logistic(linear)
    y_true = rng.binomial(1, p_true)

    out = df.copy()
    out["p_true"] = p_true
    out["Y"] = y_true
    return out


def simulate_algorithm(df: pd.DataFrame, scenario: str, seed: int, base_threshold: float) -> pd.DataFrame:
    """Generate algorithmic score and decision under specific fairness scenarios."""
    rng = np.random.default_rng(seed + 7)
    merit_signal = (
        -6.4
        + 0.18 * df["education"].values
        + 0.05 * df["experience"].values
        + 0.068 * df["performance_score"].values
        + 0.0052 * df["income_risk_score"].values
    )

    gender = df["gender"].values
    race = df["race"].values
    disability = df["disability"].values

    bias_term = np.zeros(df.shape[0])
    proxy = (100 - df["performance_score"].values) / 100

    if scenario == "fair_algorithm":
        bias_term += 0.0
    elif scenario == "marginal_bias":
        bias_term += -0.55 * race
    elif scenario == "intersectional_bias":
        bias_term += -0.95 * (gender * race * disability) - 0.25 * (race * disability)
    elif scenario == "proxy_feature_bias":
        bias_term += -0.40 * proxy - 0.25 * (proxy * race)
    elif scenario == "differential_error_rate_bias":
        bias_term += -0.20 * race
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    raw_score = logistic(merit_signal + bias_term + rng.normal(0, 0.30, size=df.shape[0]))

    threshold = np.full(df.shape[0], base_threshold)
    if scenario == "differential_error_rate_bias":
        threshold += 0.08 * race + 0.06 * disability

    y_hat = (raw_score >= threshold).astype(int)

    out = df.copy()
    out["algorithm_score"] = raw_score
    out["Y_hat"] = y_hat
    return out


def build_audit_outcome(df: pd.DataFrame, outcome: str) -> pd.Series:
    """Build binary audit outcome Z for fairness auditing."""
    if outcome == "positive_decision":
        return df["Y_hat"].astype(int)
    if outcome == "false_positive":
        return ((df["Y_hat"] == 1) & (df["Y"] == 0)).astype(int)
    if outcome == "false_negative":
        return ((df["Y_hat"] == 0) & (df["Y"] == 1)).astype(int)
    raise ValueError(f"Unsupported audit outcome: {outcome}")
