import joblib
import numpy as np
import pandas as pd
import pytest

from detection.classifier import Detector, ModelLoadError, path_contributions
from detection.config import CLASSES, EXCLUDED_FEATURES, FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from detection.feature_validator import SchemaError
from tests.conftest import record_for


@pytest.fixture(scope="module")
def detector(model_path):
    return Detector(model_path)


@pytest.mark.parametrize("class_name", CLASSES)
def test_predicts_expected_class(detector, class_name):
    batch = detector.detect(pd.DataFrame([record_for(class_name)]))
    assert batch.detections[0].predicted_class == class_name


def test_probabilities_are_labelled_by_class_not_position(detector):
    """Model classes_ are sorted, so a positional mapping would mislabel these."""
    assert list(detector._model.classes_) != list(CLASSES)
    detection = detector.detect(pd.DataFrame([record_for("Benign")])).detections[0]
    assert max(detection.probabilities, key=detection.probabilities.get) == "Benign"
    assert detection.confidence == detection.probabilities["Benign"]
    assert sum(detection.probabilities.values()) == pytest.approx(1.0)


def test_benign_is_not_an_attack(detector):
    assert not detector.detect(pd.DataFrame([record_for("Benign")])).detections[0].is_attack
    assert detector.detect(pd.DataFrame([record_for("ARP_Spoofing")])).detections[0].is_attack


def test_column_order_does_not_change_the_result(detector):
    frame = pd.DataFrame([record_for("Recon_Port_Scan")])
    shuffled = frame[list(reversed(MODEL_FEATURE_COLUMNS))]
    assert detector.detect(frame).detections[0] == detector.detect(shuffled).detections[0]


def test_invalid_row_is_rejected_and_others_still_scored(detector):
    rows = [record_for("Benign"), {**record_for("Benign"), "Rate": float("nan")}, record_for("ARP_Spoofing")]
    batch = detector.detect(pd.DataFrame(rows))
    assert [d.row for d in batch.detections] == [0, 2]
    assert list(batch.rejected) == [1]


def test_all_rows_invalid_returns_no_detections(detector):
    batch = detector.detect(pd.DataFrame([{**record_for("Benign"), "Rate": None}]))
    assert batch.detections == []
    assert len(batch.rejected) == 1


def test_wrong_schema_raises(detector):
    with pytest.raises(SchemaError):
        detector.detect(pd.DataFrame([{"Rate": 1.0}]))


def test_contributions_add_up_to_the_model_probability(detector):
    forest = detector._model
    frame = pd.DataFrame([record_for("MQTT_DDoS_Connect_Flood")])[list(MODEL_FEATURE_COLUMNS)]
    for class_name in CLASSES:
        base, contributions = path_contributions(forest, frame.to_numpy()[0], class_name)
        probability = forest.predict_proba(frame)[0][list(forest.classes_).index(class_name)]
        assert base + contributions.sum() == pytest.approx(probability)


def test_iat_is_not_a_model_input():
    assert "IAT" in FEATURE_COLUMNS
    assert "IAT" in EXCLUDED_FEATURES
    assert "IAT" not in MODEL_FEATURE_COLUMNS
    assert len(MODEL_FEATURE_COLUMNS) == 44


def test_excluded_feature_in_input_is_ignored_and_cannot_change_the_result(detector):
    frame = pd.DataFrame([record_for("ARP_Spoofing")])
    expected = detector.detect(frame).detections[0]
    for value in (0.0, 169402413.0, "not-a-number"):
        assert detector.detect(frame.assign(IAT=value)).detections[0] == expected


def test_model_feature_missing_is_still_refused(detector):
    with pytest.raises(SchemaError, match="Rate"):
        detector.detect(pd.DataFrame([record_for("Benign")]).drop(columns="Rate"))


def test_explanation_lists_real_features_that_raised_the_class(detector):
    frame = pd.DataFrame([record_for("ARP_Spoofing")])
    explanation = detector.explain(frame, "ARP_Spoofing")
    assert explanation[0]["feature"] == "Rate"
    assert all(item["contribution"] > 0 for item in explanation)
    assert all(item["feature"] in MODEL_FEATURE_COLUMNS for item in explanation)


def test_non_forest_model_gives_no_invented_explanation(model_path, tmp_path):
    from sklearn.linear_model import LogisticRegression

    features = np.random.default_rng(0).normal(size=(40, len(MODEL_FEATURE_COLUMNS)))
    labels = np.array(CLASSES)[np.arange(40) % 4]
    path = tmp_path / "logistic.joblib"
    joblib.dump(
        {
            "model": LogisticRegression(max_iter=200).fit(pd.DataFrame(features, columns=list(MODEL_FEATURE_COLUMNS)), labels),
            "feature_columns": list(MODEL_FEATURE_COLUMNS),
            "class_order": list(CLASSES),
        },
        path,
    )
    assert Detector(path).explain(pd.DataFrame([record_for("Benign")]), "Benign") == []


def test_info_does_not_expose_the_file_location(detector, model_path):
    assert str(model_path.parent) not in str(detector.info())


@pytest.mark.parametrize("contents", [{"unexpected": 1}, "not a bundle"])
def test_bad_bundle_raises_load_error(tmp_path, contents):
    path = tmp_path / "bad.joblib"
    joblib.dump(contents, path)
    with pytest.raises(ModelLoadError):
        Detector(path)


def test_missing_model_raises_load_error(tmp_path):
    with pytest.raises(ModelLoadError):
        Detector(tmp_path / "absent.joblib")
