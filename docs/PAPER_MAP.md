# Paper-to-Code Map

This document maps core methodological sections of the paper to concrete repository modules, helping reviewers and collaborators trace claims to implementation.

## Mapping table

| Paper section / concept | Repository file / module | Purpose | Key outputs |
|---|---|---|---|
| Problem setup: structural disparities in algorithmic decisions | `src/simulate_data.py` | Defines synthetic population, protected-attribute dependencies, and decision-generating process under scenarios | Scenario-specific datasets; simulated `Y`, scores, and `Y_hat` |
| Chamberlain expected-outcome baseline | `src/baseline.py` | Fits baseline model with legitimate covariates and computes expected outcomes (`pi0`) | Baseline predictions and observed-vs-expected components |
| Observed vs expected subgroup disparity | `src/fairness_metrics.py`, `src/baseline.py` | Computes subgroup-level fairness gaps and rate disparities for marginal/intersectional groups | Marginal/intersectional fairness metric tables |
| Frequentist fairness model | `src/frequentist_model.py` | Estimates logistic model with protected-attribute interactions and robust fallback fitting paths | Frequentist coefficient table with ORs, CIs, p-values, and fit method |
| Bayesian fairness model | `src/bayesian_model.py` | Fits Bayesian logistic model (NUTS with ADVI fallback) and summarizes posterior | Posterior summary table and forest-plot-ready inference data |
| Inclusion–exclusion union analysis | `src/inclusion_exclusion.py` | Computes union-level fairness quantities using inclusion–exclusion logic | Union-level fairness metrics |
| Inequality Attribution Score (IAS) | `src/ias.py` | Decomposes variance contributions into identity-attributable and merit-attributable components | IAS point estimates and Bayesian interval summaries |
| Multi-scenario experiment orchestration | `src/pipeline.py` | Runs all required scenarios/endpoints and coordinates data, model fits, tables, and figures | Scenario comparison table; saved artifacts in `results/` |
| Figure generation for manuscript and reporting | `src/plots.py` | Produces publication-ready visuals for subgroup/union disparities and IAS comparisons | `.png` figures in `results/figures/` |
| Table export for manuscript appendix/supplement | `src/tables.py` | Exports model and fairness tables in reproducible CSV form | `.csv` tables in `results/tables/` |
| End-to-end reproducible run | `main.py` | Primary entrypoint for full pipeline execution | Full set of datasets, tables, and figures |
| Interactive demonstration interface | `app.py` | Streamlit UI for scenario exploration and live auditing outputs | Browser-based dashboard with tables/plots and optional Bayesian run |
| Shared constants/utilities | `src/utils.py` | Scenario ordering, directory handling, and common utilities | Stable execution order and output directory preparation |

## How to use this map during review

1. Start from paper claims (e.g., baseline construction, IAS interpretation).
2. Locate the corresponding module above.
3. Run `python main.py` to regenerate artifacts.
4. Match generated tables/figures to manuscript sections and supplements.

## Notes

- This map is intentionally implementation-facing; it complements (not replaces) methodological narrative in the manuscript.
- If module responsibilities change, update this file in the same PR to preserve traceability.
