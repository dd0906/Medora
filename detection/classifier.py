"""Detection engine: loads a saved model once and scores validated traffic records."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from detection.config import BENIGN_CLASS, FEATURE_COLUMNS, LABEL_COLUMN
from detection.evaluation import align_probabilities
from detection.feature_validator import validate_features

logger = logging.getLogger("medora.detection")


class ModelLoadError(RuntimeError):
    """The model file is missing or is not a usable Medora model bundle."""


@dataclass(frozen=True)
class Detection:
    """One model prediction. A prediction is not a confirmed security incident."""

    row: Any
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]

    @property
    def is_attack(self) -> bool:
        return self.predicted_class != BENIGN_CLASS


@dataclass(frozen=True)
class DetectionBatch:
    detections: list[Detection]
    rejected: dict[Any, str]


class Detector:
    """Scores records with a saved model bundle. Never trains."""

    def __init__(self, model_path: Path) -> None:
        try:
            bundle = joblib.load(model_path)
            self._model = bundle["model"]
            self.feature_columns = tuple(bundle["feature_columns"])
            self.classes = tuple(bundle["class_order"])
        except (OSError, KeyError, TypeError, ValueError, EOFError) as error:
            raise ModelLoadError(f"Could not load model: {type(error).__name__}") from error

        if set(self.classes) != set(self._model.classes_):
            raise ModelLoadError("Model classes do not match the bundle's class_order.")

        # Callers may send complete dataset records. Known columns this model
        # doesn't use are ignored, but unknown columns are still refused.
        self._ignored_columns = (
            LABEL_COLUMN,
            *(c for c in FEATURE_COLUMNS if c not in self.feature_columns),
        )

        self.metadata: dict[str, Any] = bundle.get("metadata", {})
        self.version: str = self.metadata.get("model_version", Path(model_path).stem)

        # Scoring small requests on one thread avoids thread pool startup cost,
        # which otherwise dominates the time for single records.
        if "n_jobs" in self._model.get_params(deep=False):
            self._model.set_params(n_jobs=1)

        logger.info("Loaded model %s with %d features", self.version, len(self.feature_columns))

    def detect(self, frame: pd.DataFrame) -> DetectionBatch:
        """Validate and score records. Invalid rows are returned, not scored."""
        validated = validate_features(frame, expected=self.feature_columns, ignore=self._ignored_columns)
        if validated.features.empty:
            return DetectionBatch([], validated.rejected)

        probabilities = align_probabilities(
            self._model.predict_proba(validated.features),
            list(self._model.classes_),
            self.classes,
        )
        detections = []
        for row, scores in zip(validated.features.index, probabilities):
            best = int(np.argmax(scores))
            detections.append(Detection(
                row=row,
                predicted_class=self.classes[best],
                confidence=float(scores[best]),
                probabilities={c: float(p) for c, p in zip(self.classes, scores)},
            ))
        return DetectionBatch(detections, validated.rejected)

    def explain(self, record: pd.DataFrame, predicted_class: str, top_n: int = 5) -> list[dict[str, Any]]:
        """Return the features that pushed this record toward `predicted_class`.

        This describes how the model reached its output, not what caused the
        traffic. It is only implemented for Random Forest; other models return
        an empty list rather than an invented explanation.
        """
        if not isinstance(self._model, RandomForestClassifier):
            return []

        ordered = validate_features(record, expected=self.feature_columns, ignore=self._ignored_columns).features
        _, contributions = path_contributions(self._model, ordered.to_numpy()[0], predicted_class)
        ranked = sorted(enumerate(contributions), key=lambda item: item[1], reverse=True)
        return [
            {
                "feature": self.feature_columns[index],
                "value": float(ordered.iloc[0, index]),
                "contribution": float(contribution),
            }
            for index, contribution in ranked[:top_n]
            if contribution > 0
        ]

    def contributions(self, records: pd.DataFrame, class_name: str) -> np.ndarray | None:
        """Each feature's effect on `class_name`'s probability, one row per valid record.

        Returns None for models other than Random Forest, which have no such breakdown here.
        """
        if not isinstance(self._model, RandomForestClassifier):
            return None
        ordered = validate_features(records, expected=self.feature_columns, ignore=self._ignored_columns).features
        return np.array([path_contributions(self._model, row, class_name)[1] for row in ordered.to_numpy()])

    def info(self) -> dict[str, Any]:
        """Describe the model without exposing filesystem locations."""
        return {
            "version": self.version,
            "algorithm": type(self._model).__name__,
            "classes": list(self.classes),
            "feature_columns": list(self.feature_columns),
            "metadata": self.metadata,
        }


def path_contributions(
    forest: RandomForestClassifier,
    values: Sequence[float],
    class_name: str,
) -> tuple[float, np.ndarray]:
    """Split a forest's class probability into a base rate plus feature effects.

    Each split on a record's path moves the probability of the class. Crediting
    that change to the split's feature gives
    `base + contributions.sum() == predict_proba` for the class.
    """
    class_index = list(forest.classes_).index(class_name)
    contributions = np.zeros(forest.n_features_in_)
    base = 0.0

    for estimator in forest.estimators_:
        tree = estimator.tree_
        node_values = tree.value[:, 0, class_index] / tree.value[:, 0, :].sum(axis=1)
        node = 0
        base += node_values[0]
        while tree.children_left[node] != -1:
            feature = tree.feature[node]
            goes_left = values[feature] <= tree.threshold[node]
            child = tree.children_left[node] if goes_left else tree.children_right[node]
            contributions[feature] += node_values[child] - node_values[node]
            node = child

    count = len(forest.estimators_)
    return base / count, contributions / count
