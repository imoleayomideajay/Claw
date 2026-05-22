"""Batch entrypoint for running all fairness scenarios and saving outputs."""

from __future__ import annotations

from pathlib import Path

from src.pipeline import run_all_scenarios


def main() -> None:
    """Run the full multi-scenario pipeline and print output locations."""
    base_dir = Path(__file__).resolve().parent
    ias_table = run_all_scenarios(base_dir=base_dir)

    print("Completed all scenarios.")
    print(f"Saved scenario comparison summary with {len(ias_table)} rows.")
    print(f"Data directory: {base_dir / 'data'}")
    print(f"Figures directory: {base_dir / 'results' / 'figures'}")
    print(f"Tables directory: {base_dir / 'results' / 'tables'}")


if __name__ == "__main__":
    main()
