import sqlite3

import pytest

from backend.database import Database


@pytest.fixture
def database(tmp_path):
    return Database(tmp_path / "nested" / "medora.db")


def alert(**overrides):
    return {
        "id": "a1",
        "created_at": "2026-01-01T00:00:00.000000+00:00",
        "last_seen_at": "2026-01-01T00:00:00.000000+00:00",
        "source": "replay",
        "predicted_class": "ARP_Spoofing",
        "event_count": 1,
        "max_confidence": 0.8,
        "risk_score": 60.0,
        "severity": "High",
        "src_ip": None,
        "dst_ip": None,
        "protocol": None,
        "status": "open",
        "explanation": {"summary": "text"},
        **overrides,
    }


def event(**overrides):
    return {
        "occurred_at": "2026-01-01T00:00:00.000000+00:00",
        "source": "replay",
        "predicted_class": "ARP_Spoofing",
        "is_attack": 1,
        "confidence": 0.8,
        "probabilities": {"ARP_Spoofing": 0.8},
        "risk_score": 60.0,
        "severity": "High",
        "src_ip": None,
        "dst_ip": None,
        "protocol": None,
        "ground_truth": None,
        "model_version": "v",
        "alert_id": None,
        **overrides,
    }


def test_alert_round_trip_restores_explanation(database):
    with database.session() as store:
        store.create_alert(alert())
    with database.session() as store:
        saved = store.get_alert("a1")
    assert saved["explanation"] == {"summary": "text"}
    assert saved["src_ip"] is None


def test_failed_session_rolls_back_everything(database):
    with pytest.raises(RuntimeError):
        with database.session() as store:
            store.create_alert(alert())
            raise RuntimeError("failure mid-batch")
    with database.session() as store:
        assert store.get_alert("a1") is None


def test_data_survives_reopening_the_database(tmp_path):
    path = tmp_path / "medora.db"
    with Database(path).session() as store:
        store.create_alert(alert())
    with Database(path).session() as store:
        assert store.get_alert("a1") is not None


def test_event_with_unknown_alert_is_refused(database):
    with pytest.raises(sqlite3.IntegrityError):
        with database.session() as store:
            store.insert_event(event(alert_id="missing"))


def test_extending_alert_raises_risk_but_never_lowers_it(database):
    with database.session() as store:
        store.create_alert(alert())
        store.extend_alert("a1", "2026-01-01T00:00:01.000000+00:00", 0.9, 85.0, "Critical")
        raised = store.get_alert("a1")
        store.extend_alert("a1", "2026-01-01T00:00:02.000000+00:00", 0.5, 30.0, "Low")
        kept = store.get_alert("a1")
    assert (raised["risk_score"], raised["severity"]) == (85.0, "Critical")
    assert (kept["risk_score"], kept["severity"]) == (85.0, "Critical")
    assert kept["event_count"] == 3
    assert kept["max_confidence"] == 0.9


def test_recent_count_respects_window_class_source_and_ip(database):
    with database.session() as store:
        store.insert_event(event())
        store.insert_event(event(occurred_at="2025-01-01T00:00:00.000000+00:00"))
        store.insert_event(event(predicted_class="Recon_Port_Scan"))
        store.insert_event(event(source="api"))
        store.insert_event(event(src_ip="10.0.0.1"))
        count = store.count_recent_events("replay", "ARP_Spoofing", None, "2025-12-31T00:00:00.000000+00:00")
        by_ip = store.count_recent_events("replay", "ARP_Spoofing", "10.0.0.1", "2025-12-31T00:00:00.000000+00:00")
    assert count == 1
    assert by_ip == 1


def test_resolved_alert_is_not_joined(database):
    since = "2025-12-31T00:00:00.000000+00:00"
    with database.session() as store:
        store.create_alert(alert(status="resolved"))
        assert store.find_active_alert("replay", "ARP_Spoofing", None, since) is None
        store.create_alert(alert(id="a2", status="acknowledged"))
        assert store.find_active_alert("replay", "ARP_Spoofing", None, since)["id"] == "a2"


def test_filter_values_cannot_inject_sql(database):
    with database.session() as store:
        store.create_alert(alert())
        assert store.list_alerts(predicted_class="x' OR '1'='1") == []
        assert store.get_alert("a1' OR '1'='1") is None
        assert len(store.list_alerts()) == 1


def test_statistics_counts(database):
    with database.session() as store:
        store.insert_event(event())
        store.insert_event(event(predicted_class="Benign", is_attack=0, severity="Informational", source="api"))
        store.create_alert(alert())
        stats = store.statistics()
        replay_only = store.statistics(source="replay")
    assert stats["events_total"] == 2
    assert stats["events_benign"] == 1
    assert stats["events_malicious"] == 1
    assert stats["alerts_by_severity"] == {"High": 1}
    assert replay_only["events_total"] == 1


def test_statistics_on_empty_database(database):
    with database.session() as store:
        stats = store.statistics()
    assert stats["events_total"] == 0
    assert stats["events_malicious"] == 0
