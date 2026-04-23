"""Inclusion-exclusion based auditing for unions of protected groups."""

from __future__ import annotations

import pandas as pd


def _ie_count_from_sets(df: pd.DataFrame, sets: list[pd.Series], value_col: str) -> float:
    """Compute union total via inclusion-exclusion on value column."""
    if len(sets) == 2:
        a, b = sets
        return float(df.loc[a, value_col].sum() + df.loc[b, value_col].sum() - df.loc[a & b, value_col].sum())
    if len(sets) == 3:
        a, b, c = sets
        return float(
            df.loc[a, value_col].sum()
            + df.loc[b, value_col].sum()
            + df.loc[c, value_col].sum()
            - df.loc[a & b, value_col].sum()
            - df.loc[a & c, value_col].sum()
            - df.loc[b & c, value_col].sum()
            + df.loc[a & b & c, value_col].sum()
        )
    raise ValueError("Only 2-way and 3-way unions are supported")


def compute_union_metrics_pie(df: pd.DataFrame, z_col: str) -> pd.DataFrame:
    """Compute fairness metrics for unions using inclusion-exclusion to avoid double counting."""
    groups = {
        "A:gender=1": df["gender"] == 1,
        "B:race=1": df["race"] == 1,
        "C:disability=1": df["disability"] == 1,
    }

    union_defs = {
        "A∪B": [groups["A:gender=1"], groups["B:race=1"]],
        "A∪C": [groups["A:gender=1"], groups["C:disability=1"]],
        "B∪C": [groups["B:race=1"], groups["C:disability=1"]],
        "A∪B∪C": [groups["A:gender=1"], groups["B:race=1"], groups["C:disability=1"]],
    }

    rows = []
    n = len(df)
    for union_name, masks in union_defs.items():
        union_mask = masks[0].copy()
        for m in masks[1:]:
            union_mask = union_mask | m

        prevalence = float(union_mask.mean())
        obs = _ie_count_from_sets(df, masks, z_col)
        exp = _ie_count_from_sets(df, masks, "pi0")
        gap = obs - exp
        p_union = float(df.loc[union_mask, z_col].mean())
        p_comp = float(df.loc[~union_mask, z_col].mean())

        rows.append(
            {
                "union": union_name,
                "prevalence": prevalence,
                "n_union": int(union_mask.sum()),
                "n_complement": int(n - union_mask.sum()),
                "observed_count": obs,
                "expected_count": exp,
                "fairness_gap": gap,
                "union_minus_complement_rate": p_union - p_comp,
            }
        )

    return pd.DataFrame(rows)
