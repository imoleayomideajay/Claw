# AI Fairness Audit Lab

## Overview

A Streamlit-based interactive dashboard for AI fairness auditing simulations, implementing "Extending Chamberlain's Law for AI Fairness." Users can simulate algorithmic decision-making across different scenarios and audit outcomes, then inspect fairness metrics at marginal, intersectional, and union levels.

## Architecture

- **Frontend/App**: Streamlit (`app.py`) — single-page interactive dashboard
- **Entry point (batch)**: `main.py` — runs all scenarios as a pipeline (non-interactive)
- **Source modules** (`src/`):
  - `simulate_data.py` — population and algorithm simulation
  - `baseline.py` — Chamberlain baseline model
  - `fairness_metrics.py` — marginal and intersectional metrics
  - `frequentist_model.py` — frequentist fairness regression
  - `bayesian_model.py` — PyMC-based Bayesian fairness model
  - `ias.py` — Inequality Attribution Score (IAS)
  - `inclusion_exclusion.py` — union-level metrics via inclusion–exclusion
  - `pipeline.py` — batch pipeline runner
  - `plots.py` — plotting utilities
  - `tables.py` — table utilities
  - `utils.py` — shared constants (SCENARIO_ORDER, etc.)

## Stack

- **Language**: Python 3.12
- **Key dependencies**: streamlit, numpy, pandas, scipy, matplotlib, statsmodels, scikit-learn, pymc, arviz

## Running

The app runs via the "Start application" workflow:
```
streamlit run app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true
```

## Deployment

- **Target**: Autoscale
- **Run command**: `streamlit run app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true`
