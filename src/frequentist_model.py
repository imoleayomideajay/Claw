"""Frequentist fairness model using logistic regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


FORMULA_SUFFIX = (
    "education + experience + performance_score + income_risk_score + "
    "gender + race + disability + gender:race + gender:disability + race:disability"
)


def _build_summary_table(result: object, backend: str) -> pd.DataFrame:
    """Create a standard coefficient summary table from a statsmodels result."""
    params = result.params

    try:
        conf = result.conf_int()
        ci_low = np.exp(conf[0].values)
        ci_high = np.exp(conf[1].values)
    except Exception:
        ci_low = np.full(len(params), np.nan)
        ci_high = np.full(len(params), np.nan)

    try:
        pvals = result.pvalues
    except Exception:
        pvals = pd.Series(np.nan, index=params.index)

    table = pd.DataFrame(
        {
            "term": params.index,
            "coef": params.values,
            "odds_ratio": np.exp(params.values),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": pvals.values,
            "model_backend": backend,
        }
    )
    table["significant_0_05"] = table["p_value"] < 0.05
    return table.sort_values(["p_value", "term"], na_position="last")


def fit_frequentist_fairness_model(df: pd.DataFrame, z_col: str) -> tuple[object, pd.DataFrame]:
    """Fit fairness regression with robust multi-stage fallbacks for singular designs."""
    formula = f"{z_col} ~ {FORMULA_SUFFIX}"

    # 1) Preferred: MLE logit.
    try:
        result = smf.logit(formula=formula, data=df).fit(disp=0, maxiter=200)
        return result, _build_summary_table(result, backend="logit_mle")
    except Exception:
        pass

    # 2) Fallback: GLM binomial.
    try:
        result = smf.glm(formula=formula, data=df, family=sm.families.Binomial()).fit(maxiter=200)
        return result, _build_summary_table(result, backend="glm_binomial")
    except Exception:
        pass

    # 3) Final fallback: regularized logit (handles singular Hessians).
    model = smf.logit(formula=formula, data=df)
    result = model.fit_regularized(alpha=1e-4, maxiter=500, disp=0)
    return result, _build_summary_table(result, backend="logit_regularized")
