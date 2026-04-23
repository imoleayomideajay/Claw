"""Frequentist fairness model using logistic regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def _safe_exp(values: np.ndarray) -> np.ndarray:
    """Exponentiate values with clipping to avoid overflow warnings."""
    return np.exp(np.clip(values, -700, 700))


def _summary_table_from_result(result: object, fit_method: str) -> pd.DataFrame:
    """Construct a tidy coefficient summary for model outputs."""
    params = result.params
    try:
        conf = result.conf_int()
    except Exception:
        conf = pd.DataFrame(index=params.index, data={0: np.nan, 1: np.nan})

    try:
        pvals = result.pvalues
    except Exception:
        pvals = pd.Series(np.nan, index=params.index)

    table = pd.DataFrame(
        {
            "term": params.index,
            "coef": params.values,
            "odds_ratio": _safe_exp(params.values),
            "ci_low": _safe_exp(conf[0].values),
            "ci_high": _safe_exp(conf[1].values),
            "p_value": pvals.values,
            "fit_method": fit_method,
        }
    )
    table["significant_0_05"] = table["p_value"] < 0.05
    return table.sort_values(["p_value", "term"], na_position="last").reset_index(drop=True)


def fit_frequentist_fairness_model(df: pd.DataFrame, z_col: str) -> tuple[object, pd.DataFrame]:
    """Fit multivariable logistic regression with protected interactions and return tidy summary."""
    formula = (
        f"{z_col} ~ education + experience + performance_score + income_risk_score + "
        "gender + race + disability + gender:race + gender:disability + race:disability"
    )
    try:
        result = smf.logit(formula=formula, data=df).fit(disp=0, maxiter=200)
        return result, _summary_table_from_result(result, fit_method="logit_mle")
    except (np.linalg.LinAlgError, PerfectSeparationError):
        try:
            result = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit(maxiter=200)
            return result, _summary_table_from_result(result, fit_method="glm_binomial_fallback")
        except Exception:
            result = smf.logit(formula=formula, data=df).fit_regularized(
                alpha=1e-4,
                L1_wt=0.0,
                maxiter=500,
                disp=0,
            )
            return result, _summary_table_from_result(result, fit_method="logit_ridge_fallback")
