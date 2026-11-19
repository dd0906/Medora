from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "experiments" / "results"

TRAIN_PATH = DATA_DIR / "train_processed.csv"
TEST_PATH = DATA_DIR / "test_processed.csv"

CHUNK_SIZE = 50_000


def main():
    print("=" * 65)
    print("MEDORA - TRAIN/TEST DATA AUDIT")
    print("=" * 65)

    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Processed train/test files not found.")

    train_columns = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
    test_columns = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()

    if train_columns != test_columns:
        raise ValueError("Training and testing columns do not match.")

    if "label" not in train_columns:
        raise ValueError("Target label column is missing.")

    features = [c for c in train_columns if c != "label"]

    print(f"Features: {len(features)}")
    print("Train/test feature names and order match.")

    # Track exact feature-vector duplicates using row hashes.
    # Labels are excluded so we can also detect conflicting labels.
    train_seen = set()
    train_label_by_hash = {}
    train_label_conflicts = set()
    train_duplicate_rows = 0
    train_rows = 0

    feature_min = None
    feature_max = None

    print("\nAuditing training data...")

    for chunk in pd.read_csv(TRAIN_PATH, chunksize=CHUNK_SIZE):
        X = chunk[features]

        if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
            raise ValueError("Non-finite values found in training data.")

        chunk_min = X.min()
        chunk_max = X.max()

        if feature_min is None:
            feature_min = chunk_min
            feature_max = chunk_max
        else:
            feature_min = pd.concat(
                [feature_min, chunk_min], axis=1
            ).min(axis=1)

            feature_max = pd.concat(
                [feature_max, chunk_max], axis=1
            ).max(axis=1)

        hashes = pd.util.hash_pandas_object(
            X, index=False
        ).to_numpy(dtype=np.uint64)

        labels = chunk["label"].astype(str).to_numpy()

        for row_hash, label in zip(hashes, labels):
            key = int(row_hash)

            if key in train_seen:
                train_duplicate_rows += 1
            else:
                train_seen.add(key)

            previous_label = train_label_by_hash.get(key)

            if previous_label is None:
                train_label_by_hash[key] = label
            elif previous_label != label:
                train_label_conflicts.add(key)

        train_rows += len(chunk)

    # Flag features that are constant across all training rows.
    constant_features = [
        feature
        for feature in features
        if feature_min[feature] == feature_max[feature]
    ]

    test_seen = set()
    test_duplicate_rows = 0
    test_rows = 0

    overlap_rows = 0
    overlap_hashes = set()
    conflicting_test_labels = 0

    print("Auditing testing data...")

    for chunk in pd.read_csv(TEST_PATH, chunksize=CHUNK_SIZE):
        X = chunk[features]

        if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
            raise ValueError("Non-finite values found in testing data.")

        hashes = pd.util.hash_pandas_object(
            X, index=False
        ).to_numpy(dtype=np.uint64)

        labels = chunk["label"].astype(str).to_numpy()

        for row_hash, label in zip(hashes, labels):
            key = int(row_hash)

            if key in test_seen:
                test_duplicate_rows += 1
            else:
                test_seen.add(key)

            if key in train_seen:
                overlap_rows += 1
                overlap_hashes.add(key)

                if (
                    train_label_by_hash[key] != label
                    or key in train_label_conflicts
                ):
                    conflicting_test_labels += 1

        test_rows += len(chunk)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([{
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_duplicate_feature_rows": train_duplicate_rows,
        "test_duplicate_feature_rows": test_duplicate_rows,
        "test_rows_matching_train_features": overlap_rows,
        "unique_overlapping_feature_hashes": len(overlap_hashes),
        "train_feature_hashes_with_conflicting_labels": len(
            train_label_conflicts
        ),
        "overlapping_test_rows_with_label_conflicts": (
            conflicting_test_labels
        ),
        "constant_features": len(constant_features),
    }])

    summary_path = RESULTS_DIR / "split_audit.csv"
    summary.to_csv(summary_path, index=False)

    constant_path = RESULTS_DIR / "constant_features.csv"
    pd.DataFrame({
        "constant_feature": constant_features
    }).to_csv(constant_path, index=False)

    print("\n" + "=" * 65)
    print("AUDIT SUMMARY")
    print("=" * 65)
    print(summary.to_string(index=False))

    print("\nConstant features:")
    print(constant_features if constant_features else "None found")

    print("\nSaved:", summary_path)
    print("Saved:", constant_path)

    print(
        "\nNote: Matching row hashes are a screening signal, "
        "not proof of data leakage."
    )


if __name__ == "__main__":
    main()