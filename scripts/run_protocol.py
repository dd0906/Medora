"""Run Medora's evaluation protocol. Usage: python -m scripts.run_protocol

1. Cross-validate every candidate on the training split only. Exact duplicate
   feature vectors are grouped so that a copy can never cross folds.
2. Select a model with a fixed rule and write the decision to disk.
3. Only then refit on all training data and score the supplied test split.
4. Save the selected model as a new versioned file with its metadata.

Each run gets its own folder under experiments/evaluation. Existing models and
experiment results are never overwritten.
"""

import hashlib
import json
import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from detection.config import (
    CLASSES,
    DATASET_ID,
    EVALUATION_DIR,
    EXCLUDED_FEATURES,
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_FEATURE_COLUMNS,
    PROCESSED_DIR,
    SEED,
)
from detection.evaluation import (
    align_probabilities,
    class_report_rows,
    compute_metrics,
    measure_inference,
)
from detection.preprocessing import feature_hashes, load_split
from scripts.protocol_figures import plot_comparison, plot_confusion

logger = logging.getLogger("medora.protocol")

CV_FOLDS = 5
PRIMARY_MODEL = "random_forest"
SELECTION_METRIC = "macro_f1"
# Random Forest stays primary unless a candidate clearly beats it, as a gap
# smaller than this is within the fold-to-fold noise seen in earlier runs.
PRIMARY_TOLERANCE = 0.005

# Hyperparameters are the ones from the earlier experiments/ scripts, so results
# are comparable with those runs. None of them were tuned on the test split here.
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "random_state": SEED,
    "n_jobs": -1,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: Any
    selectable: bool = True
    features: tuple[str, ...] = field(default=MODEL_FEATURE_COLUMNS)
    purpose: str = "Candidate model"


def build_candidates() -> list[Candidate]:
    return [
        Candidate(
            "random_forest",
            RandomForestClassifier(class_weight=None, **RF_PARAMS),
        ),
        Candidate(
            "logistic_regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)),
            ]),
        ),
        Candidate(
            "mlp",
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    batch_size=2048,
                    learning_rate_init=0.001,
                    max_iter=100,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=8,
                    random_state=SEED,
                )),
            ]),
        ),
        Candidate(
            "lightgbm",
            LGBMClassifier(
                objective="multiclass",
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                importance_type="gain",
                random_state=SEED,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
        Candidate(
            "rf_balanced_subsample",
            RandomForestClassifier(class_weight="balanced_subsample", **RF_PARAMS),
            purpose="Class-imbalance option; also the configuration of the recorded 97.94% run",
        ),
        Candidate(
            "rf_with_iat",
            RandomForestClassifier(class_weight=None, **RF_PARAMS),
            selectable=False,
            features=FEATURE_COLUMNS,
            purpose="Diagnostic: the earlier configuration, showing how much scores rely on IAT",
        ),
    ]


def score(model: Any, features: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
    probabilities = align_probabilities(model.predict_proba(features), list(model.classes_))
    return compute_metrics(labels, model.predict(features), probabilities)


def cross_validate(
    candidates: list[Candidate],
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Score each candidate on identical duplicate-safe folds of the training split."""
    splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(X, y, groups=feature_hashes(X)))

    rows = []
    for candidate in candidates:
        for fold, (train_idx, val_idx) in enumerate(folds):
            start = time.perf_counter()
            model = clone(candidate.estimator)
            model.fit(X.iloc[train_idx][list(candidate.features)], y.iloc[train_idx])
            metrics = score(model, X.iloc[val_idx][list(candidate.features)], y.iloc[val_idx])
            seconds = time.perf_counter() - start
            rows.append({
                "model": candidate.name,
                "fold": fold,
                "seconds": seconds,
                **{k: metrics[k] for k in (
                    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                    "macro_roc_auc", "macro_pr_auc", "benign_false_alarm_rate",
                )},
                "arp_precision": metrics["per_class"]["ARP_Spoofing"]["precision"],
                "arp_recall": metrics["per_class"]["ARP_Spoofing"]["recall"],
                "arp_f1": metrics["per_class"]["ARP_Spoofing"]["f1"],
            })
            logger.info(
                "CV %s fold %d: macro_f1=%.4f (%.0fs)",
                candidate.name, fold, metrics["macro_f1"], seconds,
            )
    return pd.DataFrame(rows)


def select_model(summary: pd.DataFrame, candidates: list[Candidate]) -> dict[str, Any]:
    """Apply the fixed selection rule using cross-validation results only."""
    allowed = [c.name for c in candidates if c.selectable]
    scores = summary.loc[allowed, SELECTION_METRIC]
    best = str(scores.idxmax())
    keep_primary = scores[PRIMARY_MODEL] >= scores[best] - PRIMARY_TOLERANCE
    return {
        "selected": PRIMARY_MODEL if keep_primary else best,
        "rule": (
            f"Highest mean cross-validated {SELECTION_METRIC}; {PRIMARY_MODEL} is kept "
            f"unless beaten by more than {PRIMARY_TOLERANCE}."
        ),
        "cv_scores": {k: float(v) for k, v in scores.items()},
        "test_split_used": False,
    }


def feature_shift(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """KS distance between train and test for each feature and class.

    Run after selection, as a description of how the two splits differ.
    """
    rows = []
    for name in CLASSES:
        train_rows = X_train[y_train == name]
        test_rows = X_test[y_test == name]
        for feature in FEATURE_COLUMNS:
            result = ks_2samp(train_rows[feature], test_rows[feature])
            rows.append({"class": name, "feature": feature, "ks_statistic": result.statistic})
    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = EVALUATION_DIR / run_id
    run_dir.mkdir(parents=True)

    X_train, y_train = load_split("train")
    candidates = build_candidates()

    write_json(run_dir / "config.json", {
        "run_id": run_id,
        "dataset": DATASET_ID,
        "dataset_sha256": {
            name: sha256(PROCESSED_DIR / name)
            for name in ("train_processed.csv", "test_processed.csv")
        },
        "seed": SEED,
        "cv": {"folds": CV_FOLDS, "method": "StratifiedGroupKFold, grouped by exact feature vector"},
        "selection_metric": SELECTION_METRIC,
        "primary_tolerance": PRIMARY_TOLERANCE,
        "excluded_features": EXCLUDED_FEATURES,
        "candidates": {
            c.name: {
                "params": c.estimator.get_params(deep=True),
                "selectable": c.selectable,
                "feature_count": len(c.features),
                "purpose": c.purpose,
            }
            for c in candidates
        },
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    })

    # Steps 1 and 2 happen before the test split is even loaded.
    cv_folds = cross_validate(candidates, X_train, y_train)
    cv_folds.to_csv(run_dir / "cv_folds.csv", index=False)
    numeric = cv_folds.drop(columns="fold").groupby("model")
    cv_summary = numeric.mean()
    cv_summary.join(numeric.std().add_suffix("_std")).to_csv(run_dir / "cv_summary.csv")

    selection = select_model(cv_summary, candidates)
    write_json(run_dir / "selection.json", selection)
    logger.info("Selected %s before scoring the test split.", selection["selected"])

    # Step 3.
    X_test, y_test = load_split("test")
    overlap = np.isin(feature_hashes(X_test), feature_hashes(X_train))

    test_results, class_rows, confusion_rows, fitted = {}, [], {}, {}
    for candidate in candidates:
        columns = list(candidate.features)
        start = time.perf_counter()
        model = clone(candidate.estimator)
        model.fit(X_train[columns], y_train)
        training_seconds = time.perf_counter() - start

        metrics = score(model, X_test[columns], y_test)
        metrics["without_rows_also_in_train"] = {
            k: v for k, v in score(model, X_test.loc[~overlap, columns], y_test[~overlap]).items()
            if k in ("rows", "accuracy", "macro_f1")
        }
        metrics["training_seconds"] = training_seconds
        metrics["inference"] = measure_inference(model, X_test[columns])

        test_results[candidate.name] = metrics
        fitted[candidate.name] = model
        class_rows += class_report_rows(candidate.name, metrics)
        confusion_rows[candidate.name] = metrics["confusion_matrix"]
        logger.info(
            "TEST %s: acc=%.4f macro_f1=%.4f",
            candidate.name, metrics["accuracy"], metrics["macro_f1"],
        )

    write_json(run_dir / "test_metrics.json", test_results)
    write_json(run_dir / "test_confusion_matrices.json", {"classes": CLASSES, "matrices": confusion_rows})
    pd.DataFrame(class_rows).to_csv(run_dir / "test_class_reports.csv", index=False)
    comparison = pd.DataFrame([
        {
            "model": name,
            **{k: m[k] for k in (
                "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
                "macro_f1", "weighted_f1", "macro_roc_auc", "macro_pr_auc",
                "benign_false_alarm_rate", "attack_detection_rate", "alert_precision",
                "training_seconds",
            )},
            **m["inference"],
            "arp_precision": m["per_class"]["ARP_Spoofing"]["precision"],
            "arp_recall": m["per_class"]["ARP_Spoofing"]["recall"],
            "arp_f1": m["per_class"]["ARP_Spoofing"]["f1"],
            "cv_macro_f1": cv_summary.loc[name, "macro_f1"],
            "cv_arp_f1": cv_summary.loc[name, "arp_f1"],
            "cv_benign_false_alarm_rate": cv_summary.loc[name, "benign_false_alarm_rate"],
        }
        for name, m in test_results.items()
    ])
    comparison.to_csv(run_dir / "test_comparison.csv", index=False)

    feature_shift(X_train, y_train, X_test, y_test).to_csv(run_dir / "feature_shift.csv", index=False)

    plot_comparison(comparison, run_dir / "model_comparison.png")
    plot_confusion(
        np.array(test_results[selection["selected"]]["confusion_matrix"]),
        CLASSES,
        selection["selected"],
        run_dir / "selected_confusion_matrix.png",
    )

    # Step 4.
    name = selection["selected"]
    candidate = next(c for c in candidates if c.name == name)
    version = f"medora_{name}_{run_id}"
    metadata = {
        "model_name": name,
        "model_version": version,
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "dataset": DATASET_ID,
        "dataset_sha256": sha256(PROCESSED_DIR / "train_processed.csv"),
        "training_rows": int(len(X_train)),
        "feature_columns": list(candidate.features),
        "excluded_features": list(EXCLUDED_FEATURES),
        "class_order": list(CLASSES),
        "hyperparameters": candidate.estimator.get_params(deep=False)
        if name in ("random_forest", "lightgbm", "rf_balanced_subsample") else str(candidate.estimator),
        "preprocessing": "validate_features only: schema check, excluded features dropped, float64, rows with invalid values refused, no imputation",
        "validation": "5-fold StratifiedGroupKFold on train for selection; supplied test split for final metrics",
        "evaluation_run": run_id,
        "metrics_on_supplied_test": {
            k: test_results[name][k]
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "benign_false_alarm_rate")
        },
        "versions": {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__},
    }
    joblib.dump(
        {
            "model": fitted[name],
            "feature_columns": list(candidate.features),
            "class_order": list(CLASSES),
            "metadata": metadata,
        },
        MODEL_DIR / f"{version}.joblib",
    )
    write_json(MODEL_DIR / f"{version}.metadata.json", metadata)
    logger.info("Finished. Results: %s | Model: %s.joblib", run_dir, version)


if __name__ == "__main__":
    main()
