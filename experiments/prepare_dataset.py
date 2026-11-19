from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The source filename determines the class label.
CLASS_FILES = {
    "Benign": "Benign",
    "ARP_Spoofing": "ARP_Spoofing",
    "MQTT_DDoS_Connect_Flood": "MQTT-DDoS-Connect_Flood",
    "Recon_Port_Scan": "Recon-Port_Scan",
}

SPLITS = ("train", "test")
CHUNK_SIZE = 50_000


def expected_filename(source_name, split):
    return f"{source_name}_{split}.pcap.csv"


def main():
    # Confirm that all eight files exist before preparing any data.
    files = {}

    for split in SPLITS:
        for label, source_name in CLASS_FILES.items():
            path = RAW_DIR / expected_filename(source_name, split)

            if not path.is_file():
                raise FileNotFoundError(
                    f"Required file not found: {path}\n"
                    f"Check the filename in {RAW_DIR}."
                )

            files[(split, label)] = path

    # Verify that every file has the same feature names and order.
    reference_columns = None

    for key, path in files.items():
        columns = pd.read_csv(path, nrows=0).columns.tolist()

        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            raise ValueError(
                f"Feature columns differ in {path.name}."
            )

    if "label" in reference_columns:
        raise ValueError(
            "An existing 'label' column was found. "
            "Inspect the source schema before proceeding."
        )

    print(f"Verified {len(files)} CSV files.")
    print(f"Number of features: {len(reference_columns)}")
    print("Feature names and order match across all files.")

    summaries = []

    # Generate independent processed training and testing files.
    for split in SPLITS:
        output_path = OUTPUT_DIR / f"{split}_processed.csv"

        if output_path.exists():
            output_path.unlink()

        first_chunk = True
        split_rows = 0

        for label in CLASS_FILES:
            source_path = files[(split, label)]
            class_rows = 0

            print(f"\nProcessing {source_path.name}")

            for chunk in pd.read_csv(
                source_path,
                chunksize=CHUNK_SIZE,
            ):
                # Reject missing values.
                if chunk.isna().any().any():
                    raise ValueError(
                        f"Missing values found in {source_path.name}"
                    )

                # Ensure all features are numeric and finite.
                numeric = chunk.apply(pd.to_numeric, errors="coerce")

                if numeric.isna().any().any():
                    raise ValueError(
                        f"Non-numeric values found in {source_path.name}"
                    )

                values = numeric.to_numpy(dtype=np.float64)

                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Infinite values found in {source_path.name}"
                    )

                # Add the target label derived from the source file.
                numeric["label"] = label

                numeric.to_csv(
                    output_path,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    index=False,
                )

                first_chunk = False
                class_rows += len(numeric)
                split_rows += len(numeric)

            summaries.append({
                "split": split,
                "label": label,
                "rows": class_rows,
            })

            print(f"Rows processed: {class_rows:,}")

        print(f"\nSaved: {output_path}")
        print(f"Total {split} rows: {split_rows:,}")

    summary = pd.DataFrame(summaries)
    summary_path = OUTPUT_DIR / "label_distribution.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nTraining data: {OUTPUT_DIR / 'train_processed.csv'}")
    print(f"Testing data:  {OUTPUT_DIR / 'test_processed.csv'}")
    print(f"Label report:  {summary_path}")


if __name__ == "__main__":
    main()