"""Utility helpers for reproducible fairness-auditing experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


SCENARIO_ORDER = [
    "fair_algorithm",
    "marginal_bias",
    "intersectional_bias",
    "proxy_feature_bias",
    "differential_error_rate_bias",
]


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def get_rng(seed: int) -> np.random.Generator:
    """Return a deterministic random number generator."""
    return np.random.default_rng(seed)


def logistic(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform."""
    x_clip = np.clip(x, -25, 25)
    return 1.0 / (1.0 + np.exp(-x_clip))
