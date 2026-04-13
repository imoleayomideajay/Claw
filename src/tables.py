"""Table export utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_table(df: pd.DataFrame, output_path: Path, index: bool = False) -> None:
    """Save table as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)


def export_all_tables(table_map: dict[str, pd.DataFrame], table_dir: Path) -> None:
    """Export a dictionary of named tables to disk."""
    for name, table in table_map.items():
        save_table(table, table_dir / f"{name}.csv", index=False)
