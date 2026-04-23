# Extending Chamberlain's Law for AI Fairness (Simulation Pipeline)

This repository provides a complete, reproducible Python research pipeline for auditing algorithmic fairness under controlled simulation scenarios.

## What this pipeline does

- Simulates a synthetic population with dependent protected attributes (`gender`, `race`, `disability`) and legitimate predictors.
- Generates ground truth outcome `Y` from legitimate features only.
- Generates algorithmic score and decision `Y_hat` under 5 scenarios:
  1. `fair_algorithm`
  2. `marginal_bias`
  3. `intersectional_bias`
  4. `proxy_feature_bias`
  5. `differential_error_rate_bias`
- Audits three outcomes:
  - positive decisions
  - false positives
  - false negatives
- Implements Chamberlain expected-outcome baseline and subgroup fairness gaps.
- Fits frequentist and Bayesian fairness models.
- Computes inclusion–exclusion union metrics and IAS.
- Exports publication-ready figures and tables.

## Project structure

- `src/`: modular pipeline code
- `data/`: generated scenario datasets
- `results/figures/`: saved figures
- `results/tables/`: saved CSV tables
- `main.py`: end-to-end entrypoint
- `app.py`: Streamlit interactive dashboard
- `requirements.txt`: Python dependencies

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run everything

```bash
python main.py
```


Or launch the interactive dashboard:

```bash
streamlit run app.py
```

## Outputs

After running `python main.py`, you will get:

- One dataset per scenario in `data/`.
- Per-scenario figures in `results/figures/`:
  - observed vs expected counts
  - subgroup fairness gap
  - union vs complement gap
  - Bayesian posterior forest plot
  - error-rate disparity plot
- Cross-scenario IAS figure:
  - `ias_comparison_across_scenarios.png`
- Tables in `results/tables/` including:
  - simulated population summaries
  - frequentist model summaries
  - Bayesian posterior summaries
  - union-level fairness metrics
  - scenario comparison summary including IAS

## Reproducibility notes

- Deterministic seeds are used throughout.
- All scenario runs are orchestrated in `src/pipeline.py`.
- Bayesian model uses NUTS by default and falls back to ADVI if sampling fails.

  
