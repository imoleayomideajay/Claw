"""Entry point for the Chamberlain fairness auditing simulation pipeline."""

from __future__ import annotations

from pathlib import Path

from src.pipeline import run_all_scenarios


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    ias = run_all_scenarios(project_root, n=7000, seed=2026)
    print("Pipeline completed. Scenario IAS summary:")
    print(ias[["scenario", "audit_outcome", "IAS", "ias_point", "ias_hdi_low", "ias_hdi_high"]])
