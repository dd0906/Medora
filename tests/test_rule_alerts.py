import json
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.database import Database
from backend.main import Settings, create_app
from dashboard import views
from scripts.analyze_pcap import send_findings

API = "/api/v1"
GATEWAY = "bb:bb:bb:bb:bb:bb"
ATTACKER = "ee:ee:ee:ee:ee:ee"


def finding(**overrides):
    return {
        "rule": "binding_flapping",
        "first_seen": 100.0,
        "last_seen": 160.0,
        "ip": "10.0.0.1",
        "macs": [GATEWAY, ATTACKER],
        "count": 7,
        "detail": "The MAC for 10.0.0.1 changed 4 times within 60 seconds.",
        **overrides,
    }


@pytest.fixture
def client(model_path, tmp_path):
    settings = Settings(model_path=model_path, database_path=tmp_path / "api.db", risk_config_path=None, api_key=None)
    with TestClient(create_app(settings), raise_server_exceptions=False) as test_client:
        yield test_client


def send(client, findings, capture="spoof.pcap"):
    return client.post(f"{API}/rule-findings", json={"capture": capture, "findings": findings})


def test_finding_becomes_a_labelled_lead_without_a_confidence_value(client):
    assert send(client, [finding()]).json() == {"created": 1, "duplicates": 0}

    alert = client.get(f"{API}/alerts", params={"source": "capture"}).json()[0]
    assert alert["detector"] == "arp_rules"
    assert alert["max_confidence"] is None
    assert alert["status"] == "open"
    assert alert["event_count"] == 7
    assert "not been verified" in alert["explanation"]["status_note"]
    assert alert["explanation"]["macs"] == [GATEWAY, ATTACKER]


@pytest.mark.parametrize(
    ("rule", "severity"),
    [
        ("binding_flapping", "Medium"),
        ("unsolicited_reply_burst", "Medium"),
        ("ip_mac_conflict", "Low"),
        ("mac_claims_many_addresses", "Low"),
    ],
)
def test_rule_alerts_stay_at_medium_or_below(client, rule, severity):
    send(client, [finding(rule=rule)])
    assert client.get(f"{API}/alerts").json()[0]["severity"] == severity


def test_sending_a_capture_twice_adds_no_duplicates(client):
    assert send(client, [finding(), finding(first_seen=120.0)]).json()["created"] == 2
    assert send(client, [finding(), finding(first_seen=120.0)]).json() == {"created": 0, "duplicates": 2}
    assert len(client.get(f"{API}/alerts").json()) == 2


def test_same_finding_from_another_capture_is_a_separate_alert(client):
    send(client, [finding()], capture="a.pcap")
    assert send(client, [finding()], capture="b.pcap").json()["created"] == 1


def test_rule_alerts_never_merge_with_model_alerts_or_count_as_events(client):
    send(client, [finding()])
    stats = client.get(f"{API}/stats").json()
    assert stats["events_total"] == 0
    assert stats["alerts_by_status"] == {"open": 1}
    assert client.get(f"{API}/stats", params={"source": "replay"}).json()["alerts_by_status"] == {}


def test_rule_alert_can_be_marked_false_positive(client):
    send(client, [finding()])
    alert_id = client.get(f"{API}/alerts").json()[0]["id"]
    response = client.patch(f"{API}/alerts/{alert_id}", json={"status": "false_positive"})
    assert response.json()["status"] == "false_positive"
    assert client.get(f"{API}/alerts/{alert_id}/outcomes").json() == {}


@pytest.mark.parametrize(
    "bad",
    [
        {"rule": "made_up_rule"},
        {"ip": "not-an-ip"},
        {"macs": ["not-a-mac"]},
        {"macs": []},
        {"macs": ["AA:AA:AA:AA:AA:AA"]},
        {"count": 0},
        {"first_seen": 500.0},
        {"first_seen": -1.0},
        {"detail": "x" * 501},
        {"unexpected": "field"},
    ],
)
def test_invalid_findings_are_refused_and_nothing_is_stored(client, bad):
    assert send(client, [finding(), finding(**bad)]).status_code == 422
    assert client.get(f"{API}/alerts").json() == []


@pytest.mark.parametrize("name", ["../../etc/passwd", r"C:\Windows\system.ini", "a/b.pcap", "", "x" * 256])
def test_capture_names_that_look_like_paths_are_refused(client, name):
    assert send(client, [finding()], capture=name).status_code == 422


def test_too_many_findings_are_refused(client):
    assert send(client, [finding()] * 1001).status_code == 422


def test_clearing_rule_findings_keeps_other_alerts(client):
    client.post(f"{API}/detect", json={"record": {"features": _attack_features(client)}})
    send(client, [finding()])
    assert client.delete(f"{API}/rule-findings").status_code == 204
    assert [a["source"] for a in client.get(f"{API}/alerts").json()] == ["api"]


def _attack_features(client):
    from tests.conftest import record_for

    return record_for("ARP_Spoofing")


def test_rule_writes_need_the_api_key(model_path, tmp_path):
    settings = Settings(model_path=model_path, database_path=tmp_path / "k.db", risk_config_path=None, api_key="secret")
    with TestClient(create_app(settings)) as keyed:
        assert send(keyed, [finding()]).status_code == 401
        assert keyed.delete(f"{API}/rule-findings").status_code == 401


def test_analyze_report_can_be_sent_in_batches(client):
    findings = [finding(first_seen=float(i), last_seen=float(i)) for i in range(1500)]
    totals = send_findings(client, {"capture": "big.pcap", "findings": findings}, api_key=None)
    assert totals == {"created": 1500, "duplicates": 0}


def test_send_reports_api_refusal(client):
    with pytest.raises(httpx.HTTPStatusError):
        send_findings(client, {"capture": "bad/name", "findings": [finding()]}, api_key=None)


def test_older_database_is_upgraded_and_keeps_its_alerts(tmp_path):
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE alerts (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, source TEXT NOT NULL,
            predicted_class TEXT NOT NULL, event_count INTEGER NOT NULL, max_confidence REAL NOT NULL,
            risk_score REAL NOT NULL, severity TEXT NOT NULL, src_ip TEXT, dst_ip TEXT, protocol TEXT,
            status TEXT NOT NULL, status_updated_at TEXT, explanation TEXT NOT NULL
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, occurred_at TEXT NOT NULL, source TEXT NOT NULL,
            predicted_class TEXT NOT NULL, is_attack INTEGER NOT NULL, confidence REAL NOT NULL,
            probabilities TEXT NOT NULL, risk_score REAL NOT NULL, severity TEXT NOT NULL, src_ip TEXT, dst_ip TEXT,
            protocol TEXT, ground_truth TEXT, model_version TEXT NOT NULL, alert_id TEXT REFERENCES alerts (id)
        );
        """
    )
    connection.execute(
        "INSERT INTO alerts VALUES ('a1','t','t','replay','ARP_Spoofing',2,0.9,60,'High',NULL,NULL,NULL,'open',NULL,?)",
        (json.dumps({"summary": "old"}),),
    )
    connection.execute(
        "INSERT INTO events (occurred_at, source, predicted_class, is_attack, confidence, probabilities, risk_score, "
        "severity, model_version, alert_id) VALUES ('t','replay','ARP_Spoofing',1,0.9,'{}',60,'High','v','a1')"
    )
    connection.commit()
    connection.close()

    database = Database(path)
    Database(path)

    with database.session() as store:
        alert = store.get_alert("a1")
        assert alert["detector"] == "model"
        assert alert["max_confidence"] == 0.9
        assert alert["explanation"] == {"summary": "old"}
        assert store.statistics()["events_total"] == 1
        store.create_rule_alert({
            "id": "r1", "created_at": "t", "last_seen_at": "t", "source": "capture", "predicted_class": "ARP_Spoofing",
            "event_count": 1, "risk_score": 55, "severity": "Medium", "status": "open", "explanation": {},
            "detector": "arp_rules",
        })
        assert store.get_alert("r1")["max_confidence"] is None
        store.insert_event({
            "occurred_at": "t", "source": "replay", "predicted_class": "Benign", "is_attack": 0, "confidence": 1.0,
            "probabilities": {}, "risk_score": 0, "severity": "Informational", "src_ip": None, "dst_ip": None,
            "protocol": None, "ground_truth": None, "model_version": "v", "alert_id": "a1",
        })


# Display logic


def rule_alert():
    return {
        "severity": "Medium",
        "predicted_class": "ARP_Spoofing",
        "detector": "arp_rules",
        "risk_score": 55.0,
        "event_count": 7,
        "status": "open",
        "source": "capture",
        "last_seen_at": "2026-01-01T00:00:00.000000+00:00",
        "explanation": {
            "rule": "binding_flapping",
            "capture": "spoof.pcap",
            "detail": "Detail text.",
            "claimed_ip": None,
            "macs": [GATEWAY, ATTACKER],
            "times_observed": 1200,
            "capture_first_seen": 100.0,
            "capture_last_seen": 160.5,
        },
    }


def test_rule_alert_title_says_possible_not_confirmed():
    assert views.alert_title(rule_alert()) == "Possible ARP spoofing: MAC keeps switching"


def test_alert_table_shows_who_detected_it():
    row = views.alerts_table([rule_alert()]).iloc[0]
    assert row["Detected by"] == "Rules"
    assert row["Source"] == "Capture"


def test_evidence_table_shows_observations_and_missing_address():
    values = dict(zip(*[views.rule_evidence_table(rule_alert()["explanation"])[c] for c in ("Field", "Value")]))
    assert values["Claimed IP address"] == "Not specific to one address"
    assert values["MAC addresses"] == f"{GATEWAY}, {ATTACKER}"
    assert values["Times observed"] == "1,200"
    assert values["Time span in capture"] == "60.5 seconds"


def test_arp_rule_table_puts_test_captures_first():
    rows = [
        {"group": "ciciot2023", "capture": "b.pcap", "label": "benign", "flagged": False, "primary_findings": 0, "hours": 1.0, "seconds_to_first_primary_finding": None},
        {"group": "ciciomt_test", "capture": "a.pcap", "label": "arp_spoofing", "flagged": True, "primary_findings": 2, "hours": 0.1, "seconds_to_first_primary_finding": 9.0},
    ]
    table = views.arp_rule_table(rows)
    assert table["Set"].tolist() == ["CICIoMT2024 test", "CICIoT2023"]
    assert table["Label"].tolist() == ["ARP spoofing", "Benign"]
