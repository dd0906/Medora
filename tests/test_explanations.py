import json

import pandas as pd
import pytest

from backend.evaluation_results import _feature_effects
from dashboard import charts, views
from detection.classifier import Detector
from detection.config import CLASSES, FEATURE_COLUMNS, LABEL_COLUMN
from detection.feature_descriptions import FEATURE_NAMES, feature_meaning, feature_name
from scripts.explain_model import summarize_effects
from tests.conftest import record_for

# Plain feature descriptions


def test_every_dataset_column_has_a_plain_name():
    assert set(FEATURE_COLUMNS) <= set(FEATURE_NAMES)
    assert all(feature_meaning(column) for column in FEATURE_COLUMNS)


def test_undefined_statistics_are_not_given_invented_meanings():
    assert "isn't defined further" in feature_meaning("Radius")
    assert "time-to-live" in feature_meaning("Duration")


def test_protocol_columns_read_as_yes_or_no_questions():
    assert feature_name("HTTP") == "Uses HTTP"
    assert feature_meaning("HTTP") == "Whether the traffic uses HTTP."


def test_unknown_columns_fall_back_to_their_own_name():
    assert feature_name("Something_New") == "Something_New"
    assert feature_meaning("Something_New") == "Not described by the dataset."


# Per-alert explanations


def effect(feature, push, value=1.0):
    return {"feature": feature, "contribution": push, "value": value}


def test_reason_sentence_names_the_top_features_in_plain_words():
    features = [effect("fin_flag_number", 0.2), effect("ack_count", 0.1), effect("syn_flag_number", 0.05), effect("Rate", 0.01)]
    sentence = views.reason_sentence("MQTT_DDoS_Connect_Flood", features)
    assert sentence == "The model chose MQTT DDoS Connect Flood mainly because of: FIN flags, ACK count and SYN flags."


def test_reason_sentence_handles_one_or_no_features():
    assert views.reason_sentence("ARP_Spoofing", [effect("Header_Length", 0.3)]).endswith("because of: Header length.")
    assert "can't show" in views.reason_sentence("ARP_Spoofing", [])


def test_effects_are_shown_in_percentage_points_with_meanings():
    frame = views.effects_frame([effect("fin_flag_number", 0.178, 0.19)])
    row = frame.iloc[0]
    assert row["Feature"] == "FIN flags"
    assert row["Push"] == pytest.approx(17.8)
    assert row["Column"] == "fin_flag_number"
    assert "close a connection" in row["Meaning"]
    assert charts.effect_bars(frame).to_dict()


def test_overall_effects_use_their_own_key():
    frame = views.effects_frame([{"feature": "ack_count", "average_effect": 0.05}], "average_effect")
    assert frame.iloc[0]["Push"] == pytest.approx(5.0)


def test_performance_summary_names_strong_and_weak_types():
    metrics = {"per_class": {
        "Benign": {"f1": 0.97}, "ARP_Spoofing": {"f1": 0.62},
        "MQTT_DDoS_Connect_Flood": {"f1": 1.0}, "Recon_Port_Scan": {"f1": 0.99},
    }}
    summary = views.performance_summary(metrics)
    assert "very good at spotting MQTT DDoS Connect Flood and Recon Port Scan" in summary
    assert "weaker at ARP Spoofing" in summary


def test_performance_summary_doesnt_invent_a_weakness():
    metrics = {"per_class": {name: {"f1": 0.99} for name in CLASSES}}
    assert "weaker" not in views.performance_summary(metrics)


# What the model looks at overall


def test_contributions_add_up_to_the_models_probability(model_path):
    detector = Detector(model_path)
    record = pd.DataFrame([record_for("Recon_Port_Scan")])
    effects = detector.contributions(record, "Recon_Port_Scan")
    probability = detector.detect(record).detections[0].probabilities["Recon_Port_Scan"]
    base = probability - effects[0].sum()
    assert 0 <= base <= 1
    assert effects.shape == (1, len(detector.feature_columns))


def test_overall_effects_cover_each_predicted_type_and_rank_features(model_path):
    detector = Detector(model_path)
    rows = []
    for class_name in CLASSES:
        rows += [{**record_for(class_name), LABEL_COLUMN: class_name}] * 4
    types = summarize_effects(detector, pd.DataFrame(rows), per_class=3)
    assert set(types) == set(CLASSES)
    for summary in types.values():
        assert summary["records"] == 3
        pushes = [item["average_effect"] for item in summary["features"]]
        assert pushes == sorted(pushes, reverse=True)
        assert all(push > 0 for push in pushes)
    assert types["Recon_Port_Scan"]["features"][0]["feature"] == "Rate"


def test_saved_effects_are_loaded_for_the_dashboard(tmp_path):
    (tmp_path / "feature_effects.json").write_text(
        json.dumps({"method": "m", "types": {"Benign": {"records": 1, "features": []}}, "extra": 1}),
        encoding="utf-8",
    )
    assert _feature_effects(tmp_path) == {"method": "m", "types": {"Benign": {"records": 1, "features": []}}}


def test_missing_or_damaged_effects_mean_none(tmp_path):
    assert _feature_effects(tmp_path) is None
    (tmp_path / "feature_effects.json").write_text("{}", encoding="utf-8")
    assert _feature_effects(tmp_path) is None
