"""Search for better ARP Spoofing detection.

python -m scripts.improve_arp
    Reads the training split only. Every candidate is fitted on earlier records
    of each class and validated on later ones, which is stricter than random
    folds. The test split is not loaded, so nothing here can be tuned to it.

python -m scripts.improve_arp --finalize
    Fits the candidate chosen by that search on all training data, scores it on
    the test split, and saves it as a new versioned model. It doesn't activate
    the model or overwrite anything.
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier

from detection.config import (
    CLASSES,
    DATASET_ID,
    EVALUATION_DIR,
    EXCLUDED_FEATURES,
    MODEL_DIR,
    MODEL_FEATURE_COLUMNS,
    PROCESSED_DIR,
    SEED,
)
from detection.evaluation import align_probabilities, compute_metrics, measure_inference
from detection.preprocessing import feature_hashes, load_split
from scripts.run_protocol import score, sha256, write_json

logger = logging.getLogger("medora.improve_arp")

ARP = "ARP_Spoofing"
FORWARD_FOLDS = ((0.4, 0.6), (0.6, 0.8), (0.8, 1.0))
ARP_MULTIPLIERS = (0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
# A candidate must beat the baseline's ARP F1 by this much to be worth adopting.
MIN_ARP_F1_GAIN = 0.01

BASE_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "random_state": SEED,
    "n_jobs": -1,
}
RATIO_PAIRS = (
    ("Header_Length", "Tot sum"),
    ("syn_count", "ack_count"),
    ("rst_count", "ack_count"),
    ("fin_count", "ack_count"),
    ("Std", "AVG"),
    ("Max", "AVG"),
)


@dataclass(frozen=True)
class Candidate:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    # Returns the feature columns to fit on, using the fitting rows only.
    select: Callable[[pd.DataFrame, pd.Series], list[str]] | None = None
    add_ratios: bool = False


def with_ratios(features: pd.DataFrame) -> pd.DataFrame:
    """Add ratios a tree can't easily express with single-feature splits."""
    ratios = {
        f"{top}_per_{bottom}": features[top] / (features[bottom] + 1.0)
        for top, bottom in RATIO_PAIRS
    }
    return features.assign(**ratios)


def stable_features(max_drift: float) -> Callable[[pd.DataFrame, pd.Series], list[str]]:
    """Keep features whose Benign and ARP values don't shift between early and late rows."""

    def select(features: pd.DataFrame, labels: pd.Series) -> list[str]:
        keep = []
        for column in features.columns:
            drift = 0.0
            for name in ("Benign", ARP):
                values = features.loc[labels == name, column].to_numpy()
                middle = len(values) // 2
                drift = max(drift, ks_2samp(values[:middle], values[middle:]).statistic)
            if drift <= max_drift:
                keep.append(column)
        return keep

    return select


def build_candidates() -> list[Candidate]:
    return [
        Candidate("baseline"),
        Candidate("leaf_5", {"min_samples_leaf": 5}),
        Candidate("leaf_20", {"min_samples_leaf": 20}),
        Candidate("leaf_50", {"min_samples_leaf": 50}),
        Candidate("depth_12", {"max_depth": 12}),
        Candidate("unlimited_depth", {"max_depth": None}),
        Candidate("trees_300", {"n_estimators": 300}),
        Candidate("max_features_0.3", {"max_features": 0.3}),
        Candidate("arp_weight_2", {"class_weight": {**dict.fromkeys(CLASSES, 1.0), ARP: 2.0}}),
        Candidate("stable_drift_0.2", select=stable_features(0.2)),
        Candidate("stable_drift_0.1", select=stable_features(0.1)),
        Candidate("ratios", add_ratios=True),
        Candidate("ratios_leaf_20", {"min_samples_leaf": 20}, add_ratios=True),
    ]


def forward_splits(
    labels: pd.Series,
    hashes: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split each class by file position, and drop validation copies of fitting rows."""
    splits = []
    for fit_end, validation_end in FORWARD_FOLDS:
        fit, validation = [], []
        for name in CLASSES:
            positions = np.flatnonzero(labels.to_numpy() == name)
            fit.append(positions[: int(len(positions) * fit_end)])
            validation.append(positions[int(len(positions) * fit_end) : int(len(positions) * validation_end)])
        fit, validation = np.concatenate(fit), np.concatenate(validation)
        validation = validation[~np.isin(hashes[validation], hashes[fit])]
        splits.append((fit, validation))
    return splits


def apply_multiplier(probabilities: np.ndarray, multiplier: float) -> np.ndarray:
    """Pick classes after scaling ARP's probability, where columns follow CLASSES."""
    scaled = probabilities.copy()
    scaled[:, CLASSES.index(ARP)] *= multiplier
    return np.array(CLASSES)[scaled.argmax(axis=1)]


def order_evidence(features: pd.DataFrame, raw_iat: pd.Series, labels: pd.Series) -> dict[str, float]:
    """Share of large IAT readings that rise with file order, by class."""
    evidence = {}
    for name in CLASSES:
        values = raw_iat[(labels == name) & (raw_iat > 1e6)].to_numpy()
        evidence[name] = float((np.diff(values) >= 0).mean())
    return evidence


def evaluate(candidate: Candidate, X: pd.DataFrame, y: pd.Series, splits: list) -> list[dict[str, Any]]:
    rows = []
    for fold, (fit, validation) in enumerate(splits):
        X_fit, X_validation = X.iloc[fit], X.iloc[validation]
        columns = candidate.select(X_fit, y.iloc[fit]) if candidate.select else list(X.columns)
        if candidate.add_ratios:
            X_fit, X_validation = with_ratios(X_fit), with_ratios(X_validation)
            columns = list(X_fit.columns)

        model = RandomForestClassifier(**{**BASE_PARAMS, **candidate.params})
        model.fit(X_fit[columns], y.iloc[fit])
        probabilities = align_probabilities(model.predict_proba(X_validation[columns]), list(model.classes_))

        truth = y.iloc[validation]
        for multiplier in ARP_MULTIPLIERS:
            metrics = compute_metrics(truth, apply_multiplier(probabilities, multiplier))
            rows.append({
                "candidate": candidate.name,
                "arp_multiplier": multiplier,
                "fold": fold,
                "feature_count": len(columns),
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "arp_precision": metrics["per_class"][ARP]["precision"],
                "arp_recall": metrics["per_class"][ARP]["recall"],
                "arp_f1": metrics["per_class"][ARP]["f1"],
                "benign_false_alarm_rate": metrics["benign_false_alarm_rate"],
            })
        logger.info("%s fold %d: %d features done", candidate.name, fold, len(columns))
    return rows


def finalize() -> None:
    """Score the latest search's choice once on the test split and save it."""
    searches = sorted(EVALUATION_DIR.glob("arp_search_*"))
    decision = json.loads((searches[-1] / "decision.json").read_text(encoding="utf-8"))
    best = decision["best"]
    if not decision["adopt"]:
        sys.exit("The latest search found no candidate worth adopting.")
    if best["arp_multiplier"] != 1.0:
        sys.exit("The chosen candidate needs a decision multiplier, which the Detector doesn't support.")

    candidate = next(c for c in build_candidates() if c.name == best["candidate"])
    if candidate.select or candidate.add_ratios:
        sys.exit("The chosen candidate changes the features, which the Detector doesn't support.")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = EVALUATION_DIR / f"arp_final_{run_id}"
    run_dir.mkdir(parents=True)

    columns = list(MODEL_FEATURE_COLUMNS)
    X_train, y_train = load_split("train")
    params = {**BASE_PARAMS, **candidate.params}
    model = RandomForestClassifier(**params).fit(X_train[columns], y_train)

    X_test, y_test = load_split("test")
    metrics = score(model, X_test[columns], y_test)
    metrics["inference"] = measure_inference(model, X_test[columns])
    write_json(run_dir / "test_metrics.json", {"candidate": candidate.name, "params": params, **metrics})

    version = f"medora_random_forest_{run_id}"
    metadata = {
        "model_name": "random_forest",
        "model_version": version,
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "dataset": DATASET_ID,
        "dataset_sha256": sha256(PROCESSED_DIR / "train_processed.csv"),
        "training_rows": int(len(X_train)),
        "feature_columns": columns,
        "excluded_features": list(EXCLUDED_FEATURES),
        "class_order": list(CLASSES),
        "hyperparameters": model.get_params(deep=False),
        "preprocessing": "validate_features only: schema check, excluded features dropped, float64, rows with invalid values refused, no imputation",
        "validation": "Forward validation on train (earlier records fit, later records validate) for selection; supplied test split for final metrics",
        "evaluation_run": run_id,
        "metrics_on_supplied_test": {
            k: metrics[k]
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "benign_false_alarm_rate")
        },
        "versions": {"scikit-learn": sklearn.__version__},
    }
    joblib.dump(
        {"model": model, "feature_columns": columns, "class_order": list(CLASSES), "metadata": metadata},
        MODEL_DIR / f"{version}.joblib",
    )
    write_json(MODEL_DIR / f"{version}.metadata.json", metadata)
    logger.info("Finished: %s | Model: %s.joblib", run_dir, version)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Search for better ARP Spoofing detection.")
    parser.add_argument("--finalize", action="store_true", help="Score the search's choice on the test split.")
    if parser.parse_args().finalize:
        finalize()
        return

    run_dir = EVALUATION_DIR / f"arp_search_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)

    features, labels = load_split("train")
    X = features[list(MODEL_FEATURE_COLUMNS)]
    splits = forward_splits(labels, feature_hashes(X))

    rows = []
    for candidate in build_candidates():
        rows += evaluate(candidate, X, labels, splits)
    folds = pd.DataFrame(rows)
    folds.to_csv(run_dir / "forward_folds.csv", index=False)

    summary = (
        folds.drop(columns="fold")
        .groupby(["candidate", "arp_multiplier"])
        .mean()
        .sort_values("arp_f1", ascending=False)
        .reset_index()
    )
    summary.to_csv(run_dir / "forward_summary.csv", index=False)

    baseline = summary.query("candidate == 'baseline' and arp_multiplier == 1.0").iloc[0]
    eligible = summary[summary["macro_f1"] >= baseline["macro_f1"]]
    best = eligible.iloc[0]
    decision = {
        "rule": (
            f"Highest mean forward ARP F1 among candidates with macro F1 at least the "
            f"baseline's; adopt only if it gains more than {MIN_ARP_F1_GAIN}."
        ),
        "baseline": baseline.to_dict(),
        "best": best.to_dict(),
        "arp_f1_gain": float(best["arp_f1"] - baseline["arp_f1"]),
        "adopt": bool(best["arp_f1"] - baseline["arp_f1"] > MIN_ARP_F1_GAIN),
        "test_split_used": False,
        "file_order_evidence": order_evidence(X, features["IAT"], labels),
    }
    (run_dir / "decision.json").write_text(json.dumps(decision, indent=2, default=float), encoding="utf-8")
    logger.info("Finished: %s", run_dir)


if __name__ == "__main__":
    main()
