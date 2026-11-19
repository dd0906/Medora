import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from detection.config import CLASSES, MODEL_FEATURE_COLUMNS


def synthetic_features(rows: int, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Separable data: each class has its own range of the Rate feature."""
    rng = np.random.default_rng(seed)
    labels = np.array(CLASSES)[np.arange(rows) % len(CLASSES)]
    frame = pd.DataFrame(rng.normal(size=(rows, len(MODEL_FEATURE_COLUMNS))), columns=list(MODEL_FEATURE_COLUMNS))
    frame["Rate"] = (np.arange(rows) % len(CLASSES)) * 100.0 + rng.normal(scale=1.0, size=rows)
    return frame, labels


def record_for(class_name: str) -> dict[str, float]:
    """A feature dict the synthetic model confidently assigns to `class_name`."""
    row = {name: 0.0 for name in MODEL_FEATURE_COLUMNS}
    row["Rate"] = CLASSES.index(class_name) * 100.0
    return row


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    """A small real Random Forest bundle, so tests don't need the dataset."""
    features, labels = synthetic_features(400)
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0).fit(features, labels)
    path = tmp_path_factory.mktemp("models") / "test_model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(MODEL_FEATURE_COLUMNS),
            "class_order": list(CLASSES),
            "metadata": {"model_version": "test-model"},
        },
        path,
    )
    return path
