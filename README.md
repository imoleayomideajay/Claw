# Extending Chamberlain’s Law for AI Fairness

[![Status](https://img.shields.io/badge/status-active%20research-blue)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)

**Tagline:** A reproducible simulation and auditing pipeline for quantifying structural disparities in algorithmic decision-making using Chamberlain-style expected outcomes, frequentist/Bayesian models, and inequality attribution.

---

## Overview

This repository accompanies the methodological work:

> **“Extending Chamberlain’s Law for AI Fairness: A Probabilistic Framework for Quantifying Structural Disparities in Algorithmic Decision-Making.”**

The codebase provides a complete research workflow to:
- generate synthetic populations with dependent protected attributes,
- simulate algorithmic decisions under multiple bias regimes,
- estimate observed-vs-expected subgroup disparities,
- quantify marginal/intersectional/union-level fairness gaps,
- fit frequentist and Bayesian fairness models, and
- compute an **Inequality Attribution Score (IAS)** that separates identity-attributable from merit-attributable variance.

## Why this project matters

Most fairness analyses stop at a small set of group metrics. This project extends that perspective by combining:
1. **Normative baseline modeling** (expected outcomes under legitimate covariates),
2. **Structured subgroup and union analysis** (inclusion–exclusion logic), and
3. **Variance decomposition via IAS** for interpretable disparity attribution.

This makes the repository suitable for research replication, method comparison, and reviewer-facing evidence generation.

## Methodology at a glance

1. **Population simulation** with correlated identities (`gender`, `race`, `disability`) and legitimate predictors.
2. **Ground truth generation** from legitimate predictors only.
3. **Algorithm simulation** under configurable fairness/bias scenarios.
4. **Audit outcomes** for positive decisions, false positives, and false negatives.
5. **Chamberlain baseline** estimation (`pi0`) and subgroup fairness gaps (observed minus expected).
6. **Frequentist modeling** with protected-attribute interactions.
7. **Bayesian modeling** (NUTS with ADVI fallback).
8. **Inclusion–exclusion union metrics** and **IAS** calculations.
9. **Figure/table export** plus optional interactive inspection in Streamlit.

## Mapping to the paper

A dedicated section-to-code map is provided in [`docs/PAPER_MAP.md`](docs/PAPER_MAP.md).

## Repository structure

```text
.
├── app.py                     # Streamlit interactive dashboard
├── main.py                    # End-to-end pipeline entrypoint
├── requirements.txt           # Python dependencies
├── src/
│   ├── simulate_data.py       # Synthetic population, outcomes, scenario simulation
│   ├── baseline.py            # Chamberlain expected-outcome baseline
│   ├── fairness_metrics.py    # Marginal and intersectional disparity metrics
│   ├── frequentist_model.py   # Frequentist fairness modeling
│   ├── bayesian_model.py      # Bayesian fairness modeling + posterior summaries
│   ├── inclusion_exclusion.py # Union-level subgroup analysis
│   ├── ias.py                 # Inequality Attribution Score (IAS)
│   ├── plots.py               # Figure generation
│   ├── tables.py              # Table export utilities
│   ├── pipeline.py            # Scenario orchestration
│   └── utils.py               # Shared constants/utilities
├── docs/
│   └── PAPER_MAP.md           # Paper-to-code mapping for reviewers
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── RELEASE_CHECKLIST.md
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

Run the complete experiment suite:

```bash
python main.py
```

Launch interactive dashboard:

```bash
streamlit run app.py
```

## Running simulations

The default pipeline runs all configured scenarios and writes outputs to:
- `data/`
- `results/figures/`
- `results/tables/`

You can adapt sample size and seeds by editing runtime parameters in `main.py`/`src/pipeline.py`.

## Streamlit app usage

The app exposes controls for:
- scenario selection,
- audit outcome selection,
- sample size, seed, and threshold,
- optional Bayesian posterior sampling.

It is intended for exploratory analysis and demonstration; publication artifacts should still be generated via `main.py` for consistency.

## Simulation scenarios

Current scenarios include:
1. `fair_algorithm`
2. `marginal_bias`
3. `intersectional_bias`
4. `proxy_feature_bias`
5. `differential_error_rate_bias`

## Outputs generated

After `python main.py`, expected outputs include:
- Scenario datasets (`data/*_dataset.csv`)
- Fairness figures (`results/figures/*.png`), including observed-vs-expected and IAS comparison plots
- Summary tables (`results/tables/*.csv`) for population, frequentist/Bayesian model summaries, union metrics, and IAS

## Reproducibility notes

- Deterministic seeds are used in simulation and fitting entrypoints.
- Scenarios are run in a fixed order for consistent comparison outputs.
- Bayesian fitting defaults to NUTS and falls back to ADVI where needed for robustness.
- For paper-grade replication, pin dependency versions and archive outputs with commit hash metadata.

## Citation

If you use this repository, please cite the project metadata in [`CITATION.cff`](CITATION.cff).

## Limitations

- Synthetic-data conclusions are scenario-dependent and not a substitute for real-world causal claims.
- Fairness definitions are metric- and modeling-choice dependent.
- Numerical optimization may require fallback strategies under severe separation or near-collinearity.
- Bayesian posterior quality is sensitive to tuning/sampling settings and compute resources.

## Roadmap

- Add benchmark comparisons against alternative fairness auditing frameworks.
- Add robustness and sensitivity notebooks.
- Expand documentation on interpretation of IAS in policy-oriented contexts.
- Add optional CLI configuration for scenario, seed, and output directories.

## Contributing

Contributions are welcome. Please read:
- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

## License

This repository is released under the MIT License (see [`LICENSE`](LICENSE)).
A licensing rationale note is available in [`LICENSE_RECOMMENDATION.md`](LICENSE_RECOMMENDATION.md).
