from pathlib import Path

import pandas as pd


# Locate the Medora project and its raw data folder.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"

# Find the downloaded training CSV files.
files = sorted(DATA_DIR.glob("*_train.pcap.csv"))

if not files:
    print(f"No training CSV files found in: {DATA_DIR}")
    raise SystemExit(1)

print("=" * 70)
print("MEDORA - IoMT DATASET INSPECTION")
print("=" * 70)
print(f"Data directory: {DATA_DIR}")
print(f"Training files found: {len(files)}")

for file_path in files:
    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print(f"Size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")

    try:
        # Read only five rows for the initial inspection.
        sample = pd.read_csv(file_path, nrows=5)

        print(f"\nNumber of columns: {len(sample.columns)}")
        print("\nColumn names and inferred data types:")

        for column in sample.columns:
            print(f"  {column}: {sample[column].dtype}")

        print("\nFirst two sample rows:")
        print(sample.head(2).to_string(index=False))

    except Exception as error:
        print(f"Error inspecting file: {error}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)