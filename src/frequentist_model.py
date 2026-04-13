"""Frequentist fairness model using logistic regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_frequentist_fairness_model(df: pd.DataFrame, z_col: str) -> tuple[object, pd.DataFrame]:
    """Fit fairness regression with automatic fallback when Logit is unstable."""
    formula = (
        f"{z_col} ~ education + experience + performance_score + income_risk_score + "
        "gender + race + disability + gender:race + gender:disability + race:disability"
    )

    backend = "logit"
    try:
        result = smf.logit(formula=formula, data=df).fit(disp=0, maxiter=200)
    except Exception:
        backend = "glm_binomial_fallback"
        result = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit(maxiter=200)

    params = result.params
    conf = result.conf_int()
    pvals = result.pvalues

    table = pd.DataFrame(
        {
            "term": params.index,
            "coef": params.values,
            "odds_ratio": np.exp(params.values),
            "ci_low": np.exp(conf[0].values),
            "ci_high": np.exp(conf[1].values),
            "p_value": pvals.values,
            "model_backend": backend,
        }
    )
    table["significant_0_05"] = table["p_value"] < 0.05
    return result, table.sort_values("p_value")
