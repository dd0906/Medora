from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(DATA_DIR.glob("*_train.pcap.csv"))

if not files:
    raise SystemExit(f"No training CSV files found in {DATA_DIR}")

file_summaries = []
column_reports = []

print("=" * 70)
print("MEDORA - FULL DATA QUALITY CHECK")
print("=" * 70)

for file_path in files:
    print(f"\nInspecting: {file_path.name}")

    row_count = 0
    missing_by_column = None
    infinite_by_column = None
    invalid_by_column = None
    columns = None

    # Process 100,000 rows at a time.
    for chunk in pd.read_csv(file_path, chunksize=100_000):
        row_count += len(chunk)

        if columns is None:
            columns = list(chunk.columns)
            missing_by_column = pd.Series(0, index=columns, dtype="int64")
            infinite_by_column = pd.Series(0, index=columns, dtype="int64")
            invalid_by_column = pd.Series(0, index=columns, dtype="int64")

        missing_by_column += chunk.isna().sum()

        # Detect non-numeric values and infinities.
        numeric = chunk.apply(pd.to_numeric, errors="coerce")

        invalid_mask = chunk.notna() & numeric.isna()
        invalid_by_column += invalid_mask.sum()

        numeric_array = numeric.to_numpy(dtype=np.float64)
        infinite_mask = pd.DataFrame(
            np.isinf(numeric_array),
            columns=columns,
        )
        infinite_by_column += infinite_mask.sum()

    total_missing = int(missing_by_column.sum())
    total_infinite = int(infinite_by_column.sum())
    total_invalid = int(invalid_by_column.sum())

    constant_or_unchecked = "Requires separate check"

    file_summaries.append({
        "file": file_path.name,
        "rows": row_count,
        "columns": len(columns),
        "missing_cells": total_missing,
        "infinite_cells": total_infinite,
        "nonnumeric_cells": total_invalid,
        "size_MB": round(file_path.stat().st_size / (1024 * 1024), 2),
    })

    for column in columns:
        column_reports.append({
            "file": file_path.name,
            "column": column,
            "missing_values": int(missing_by_column[column]),
            "infinite_values": int(infinite_by_column[column]),
            "nonnumeric_values": int(invalid_by_column[column]),
        })

    print(f"Rows: {row_count:,}")
    print(f"Columns: {len(columns)}")
    print(f"Missing cells: {total_missing:,}")
    print(f"Infinite cells: {total_infinite:,}")
    print(f"Non-numeric cells: {total_invalid:,}")

summary = pd.DataFrame(file_summaries)
column_report = pd.DataFrame(column_reports)

summary_path = OUTPUT_DIR / "dataset_profile.csv"
columns_path = OUTPUT_DIR / "column_quality.csv"

summary.to_csv(summary_path, index=False)
column_report.to_csv(columns_path, index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(summary.to_string(index=False))

print(f"\nSaved file summary: {summary_path}")
print(f"Saved column report: {columns_path}")
print("\nData-quality inspection complete.")