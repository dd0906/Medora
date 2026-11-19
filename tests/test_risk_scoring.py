import json

import pytest

from detection.risk_scoring import RiskConfig, assess_risk, load_risk_config

CONFIG = RiskConfig()


def test_benign_is_informational_and_never_alerts_even_when_repeated():
    risk = assess_risk("Benign", 1.0, 1000, CONFIG)
    assert risk.score == 0
    assert risk.severity == "Informational"
    assert not risk.should_alert


def test_score_follows_the_documented_formula():
    risk = assess_risk("ARP_Spoofing", 0.8, 3, CONFIG)
    assert risk.score == pytest.approx(70 * (0.5 + 0.5 * 0.8) + 2 * 3)
    assert risk.components["base_score"] == 70
    assert risk.components["repeat_bonus"] == 6


def test_confidence_does_not_override_attack_severity():
    """A very sure port scan must still rank below a moderately sure flood."""
    scan = assess_risk("Recon_Port_Scan", 1.0, 0, CONFIG)
    flood = assess_risk("MQTT_DDoS_Connect_Flood", 0.6, 0, CONFIG)
    assert flood.score > scan.score


def test_repeat_bonus_is_capped():
    assert assess_risk("Recon_Port_Scan", 1.0, 10_000, CONFIG).components["repeat_bonus"] == CONFIG.repeat_cap


def test_score_never_exceeds_100():
    assert assess_risk("MQTT_DDoS_Connect_Flood", 1.0, 10_000, RiskConfig(repeat_cap=500)).score == 100


@pytest.mark.parametrize(
    ("class_name", "confidence", "repeats", "severity"),
    [
        ("Recon_Port_Scan", 0.0, 0, "Low"),
        ("Recon_Port_Scan", 1.0, 0, "Medium"),
        ("ARP_Spoofing", 1.0, 0, "High"),
        ("ARP_Spoofing", 1.0, 5, "Critical"),
    ],
)
def test_severity_bands(class_name, confidence, repeats, severity):
    assert assess_risk(class_name, confidence, repeats, CONFIG).severity == severity


def test_threshold_boundary_is_inclusive():
    config = RiskConfig(confidence_floor=1.0, base_scores={
        "Benign": 0, "ARP_Spoofing": 80, "MQTT_DDoS_Connect_Flood": 60, "Recon_Port_Scan": 20,
    })
    assert assess_risk("ARP_Spoofing", 0.5, 0, config).severity == "Critical"
    assert assess_risk("MQTT_DDoS_Connect_Flood", 0.5, 0, config).severity == "High"
    assert assess_risk("Recon_Port_Scan", 0.5, 0, config).severity == "Low"


def test_low_scores_do_not_alert():
    config = RiskConfig(alert_min_score=50)
    assert not assess_risk("Recon_Port_Scan", 0.1, 0, config).should_alert


@pytest.mark.parametrize(
    ("class_name", "confidence", "repeats"),
    [("Unknown", 0.5, 0), ("Benign", 1.5, 0), ("Benign", -0.1, 0), ("Benign", 0.5, -1)],
)
def test_invalid_inputs_are_refused(class_name, confidence, repeats):
    with pytest.raises(ValueError):
        assess_risk(class_name, confidence, repeats, CONFIG)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_scores": {"Benign": 0}},
        {"severity_thresholds": (50.0, 40.0, 60.0, 80.0)},
        {"confidence_floor": 2.0},
    ],
)
def test_invalid_config_is_refused(kwargs):
    with pytest.raises(ValueError):
        RiskConfig(**kwargs)


def test_config_can_be_overridden_from_a_file(tmp_path):
    path = tmp_path / "risk.json"
    path.write_text(json.dumps({"repeat_cap": 5, "severity_thresholds": [10, 20, 30, 40]}), encoding="utf-8")
    config = load_risk_config(path)
    assert config.repeat_cap == 5
    assert config.severity_thresholds == (10, 20, 30, 40)
    assert config.base_scores == CONFIG.base_scores
