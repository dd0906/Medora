"""Check whether Medora's features carry over to CICIoT2023. Usage: python -m scripts.external_check

The active model can't be scored there, because seven of its features are absent
and several shared column names hold differently computed values. This measures
that mismatch, then fits a throwaway model on the features that do line up. It
doesn't change or save any model.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier

from detection.config import CLASSES, DATA_DIR, EVALUATION_DIR, MODEL_FEATURE_COLUMNS, SEED
from detection.evaluation import compute_metrics
from detection.feature_validator import validate_features
from detection.preprocessing import load_split

logger = logging.getLogger("medora.external_check")

EXTERNAL_DIR = DATA_DIR / "external" / "ciciot2023"
# CICIoT2023 folder name, and the Medora class it corresponds to.
FOLDER_CLASSES = {
    "Benign_Final": "Benign",
    "MITM-ArpSpoofing": "ARP_Spoofing",
    "Recon-PortScan": "Recon_Port_Scan",
}
# Features whose benign distributions differ by more than this are treated as
# not comparable between the two datasets.
MAX_BENIGN_KS = 0.2
RF_PARAMS = {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2, "random_state": SEED, "n_jobs": -1}


def load_external(directory: Path, shared: list[str]) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Load labelled records, taking each label from its folder, and refuse bad rows."""
    frames, labels, rejected = [], [], {}
    for folder, label in FOLDER_CLASSES.items():
        raw = pd.concat([pd.read_csv(path) for path in sorted((directory / folder).glob("*.csv"))], ignore_index=True)
        result = validate_features(raw[shared], expected=shared)
        frames.append(result.features)
        labels.append(pd.Series(label, index=result.features.index))
        rejected[folder] = len(result.rejected)
    return pd.concat(frames, ignore_index=True), pd.concat(labels, ignore_index=True), rejected


def benign_drift(source: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """KS distance between the two datasets' benign records, for each feature."""
    return pd.Series(
        {column: ks_2samp(source[column], target[column]).statistic for column in source.columns}
    ).sort_values()


def summarize(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = compute_metrics(y_true, y_pred)
    per_class = {name: metrics["per_class"][name] for name in CLASSES if metrics["per_class"][name]["support"]}
    return {
        "rows": metrics["rows"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1_present_classes": float(np.mean([values["f1"] for values in per_class.values()])),
        "benign_false_alarm_rate": metrics["benign_false_alarm_rate"],
        "attack_detection_rate": metrics["attack_detection_rate"],
        "per_class": per_class,
        "confusion_matrix": {"classes": CLASSES, "matrix": metrics["confusion_matrix"]},
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = EVALUATION_DIR / f"external_ciciot2023_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)

    header = pd.read_csv(next((EXTERNAL_DIR / "Benign_Final").glob("*.csv")), nrows=0).columns
    shared = [column for column in MODEL_FEATURE_COLUMNS if column in header]
    missing = [column for column in MODEL_FEATURE_COLUMNS if column not in header]

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    X_external, y_external, rejected = load_external(EXTERNAL_DIR, shared)

    drift = benign_drift(X_train.loc[y_train == "Benign", shared], X_external[y_external == "Benign"])
    drift.rename("benign_ks").to_csv(run_dir / "benign_feature_drift.csv", index_label="feature")
    comparable = drift[drift <= MAX_BENIGN_KS].index.tolist()

    results = {
        "missing_model_features": missing,
        "shared_features": len(shared),
        "rejected_rows_by_folder": rejected,
        "max_benign_ks": MAX_BENIGN_KS,
        "comparable_features": comparable,
    }
    for name, columns in (("comparable_features_model", comparable), ("all_shared_features_model", shared)):
        if not columns:
            continue
        model = RandomForestClassifier(**RF_PARAMS).fit(X_train[columns], y_train)
        results[name] = {
            "feature_count": len(columns),
            "ciciomt2024_test": summarize(y_test, model.predict(X_test[columns])),
            "ciciot2023": summarize(y_external, model.predict(X_external[columns])),
        }

    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Finished: %s", run_dir)


if __name__ == "__main__":
    main()
