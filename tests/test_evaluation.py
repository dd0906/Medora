import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from detection.config import CLASSES, FEATURE_COLUMNS, MODEL_DIR
from detection.evaluation import align_probabilities, compute_metrics, measure_inference

BENIGN, ARP, MQTT, RECON = CLASSES


def test_hand_checked_metrics():
    y_true = [BENIGN, BENIGN, BENIGN, BENIGN, ARP, ARP, MQTT, RECON]
    y_pred = [BENIGN, BENIGN, BENIGN, ARP, ARP, BENIGN, MQTT, RECON]
    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(6 / 8)
    assert metrics["balanced_accuracy"] == pytest.approx((3 / 4 + 1 / 2 + 1 + 1) / 4)
    assert metrics["benign_false_alarm_rate"] == pytest.approx(1 / 4)
    assert metrics["attack_detection_rate"] == pytest.approx(3 / 4)
    assert metrics["alert_precision"] == pytest.approx(3 / 4)
    assert metrics["per_class"][ARP]["precision"] == pytest.approx(1 / 2)
    assert metrics["per_class"][ARP]["recall"] == pytest.approx(1 / 2)
    assert metrics["per_class"][BENIGN]["support"] == 4
    assert metrics["confusion_matrix"][0] == [3, 1, 0, 0]
    assert metrics["confusion_matrix"][1] == [1, 1, 0, 0]


def test_wrong_attack_class_still_counts_as_detected():
    metrics = compute_metrics([MQTT, BENIGN], [RECON, BENIGN], classes=CLASSES)
    assert metrics["attack_detection_rate"] == 1.0
    assert metrics["accuracy"] == 0.5


def test_probabilities_are_realigned_by_class_name():
    # Sorted order, as a fitted sklearn model reports it.
    model_classes = sorted(CLASSES)
    assert model_classes != list(CLASSES)
    probabilities = np.array([[0.1, 0.2, 0.3, 0.4]])

    aligned = align_probabilities(probabilities, model_classes)

    for i, name in enumerate(CLASSES):
        assert aligned[0, i] == probabilities[0, model_classes.index(name)]


def test_perfect_scores_give_unit_auc():
    y_true = [BENIGN, ARP, MQTT, RECON] * 3
    probabilities = np.eye(4)[[0, 1, 2, 3] * 3]
    metrics = compute_metrics(y_true, y_true, probabilities)
    assert metrics["macro_roc_auc"] == 1.0
    assert metrics["macro_pr_auc"] == 1.0


def test_auc_is_omitted_when_a_class_is_absent():
    y_true = [BENIGN, ARP, MQTT]
    metrics = compute_metrics(y_true, y_true, np.eye(4)[[0, 1, 2]])
    assert "macro_roc_auc" not in metrics
    assert metrics["per_class"][RECON]["support"] == 0


def test_measure_inference_restores_model_threads():
    features = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    model = DummyClassifier(strategy="most_frequent").fit(features, ["x", "y", "y"])
    model.n_jobs = -1
    model.set_params = lambda **kw: setattr(model, "n_jobs", kw["n_jobs"])

    timing = measure_inference(model, features)

    assert model.n_jobs == -1
    assert timing["batch_microseconds_per_row"] > 0
    assert timing["single_row_milliseconds"] > 0


def test_saved_random_forest_loads_and_predicts_known_classes():
    joblib = pytest.importorskip("joblib")
    path = MODEL_DIR / "medora_rf.joblib"
    if not path.is_file():
        pytest.skip("Saved model is not available.")

    saved = joblib.load(path)
    assert saved["feature_columns"] == list(FEATURE_COLUMNS)

    row = pd.DataFrame([np.zeros(len(FEATURE_COLUMNS))], columns=list(FEATURE_COLUMNS))
    assert saved["model"].predict(row)[0] in CLASSES
    probabilities = saved["model"].predict_proba(row)
    assert probabilities.shape == (1, 4)
    assert probabilities.sum() == pytest.approx(1.0)
