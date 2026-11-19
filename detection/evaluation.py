"""Classification metrics computed the same way for every Medora model."""

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from detection.config import BENIGN_CLASS, CLASSES


def align_probabilities(
    probabilities: np.ndarray,
    model_classes: Sequence[str],
    classes: Sequence[str] = CLASSES,
) -> np.ndarray:
    """Reorder probability columns from the model's class order to `classes`."""
    positions = {name: index for index, name in enumerate(model_classes)}
    return probabilities[:, [positions[name] for name in classes]]


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: np.ndarray | None = None,
    classes: Sequence[str] = CLASSES,
) -> dict[str, Any]:
    """Return JSON-ready metrics. `probabilities` must be ordered as `classes`.

    ROC-AUC and PR-AUC are one-vs-rest per class with an unweighted macro mean.
    They are omitted without probabilities, or when a class is absent from
    `y_true` and the score would be undefined.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = list(classes)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(classes)
    }

    is_benign = y_true == BENIGN_CLASS
    predicted_attack = y_pred != BENIGN_CLASS

    metrics = {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
        # Share of benign rows raised as any attack: the alert-noise rate.
        "benign_false_alarm_rate": _safe_mean(predicted_attack[is_benign]),
        # Share of attack rows raised as any attack, right class or not.
        "attack_detection_rate": _safe_mean(predicted_attack[~is_benign]),
        # Share of raised rows that really were attacks.
        "alert_precision": _safe_mean(~is_benign[predicted_attack]),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }

    if probabilities is not None and support.all():
        for i, name in enumerate(classes):
            is_class = (y_true == name).astype(int)
            per_class[name]["roc_auc"] = float(roc_auc_score(is_class, probabilities[:, i]))
            per_class[name]["pr_auc"] = float(average_precision_score(is_class, probabilities[:, i]))
        metrics["macro_roc_auc"] = float(np.mean([per_class[n]["roc_auc"] for n in classes]))
        metrics["macro_pr_auc"] = float(np.mean([per_class[n]["pr_auc"] for n in classes]))

    return metrics


def _safe_mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def measure_inference(model: Any, features: pd.DataFrame, single_rows: int = 200) -> dict[str, float]:
    """Time batch prediction and one-row-at-a-time scoring on one thread.

    One thread is forced for the single-row figure because that is how a
    per-record service would run it: thread pool startup otherwise dominates
    the time for small inputs.
    """
    start = time.perf_counter()
    model.predict(features)
    batch_seconds = time.perf_counter() - start

    original_jobs = getattr(model, "n_jobs", None)
    if original_jobs is not None:
        model.set_params(n_jobs=1)
    sample = [features.iloc[[i]] for i in range(min(single_rows, len(features)))]
    start = time.perf_counter()
    for row in sample:
        model.predict_proba(row)
    single_seconds = time.perf_counter() - start
    if original_jobs is not None:
        model.set_params(n_jobs=original_jobs)

    return {
        "batch_microseconds_per_row": batch_seconds / len(features) * 1e6,
        "single_row_milliseconds": single_seconds / len(sample) * 1e3,
    }


def class_report_rows(model_name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten per-class metrics into rows for a CSV report."""
    return [{"model": model_name, "class": name, **values} for name, values in metrics["per_class"].items()]
