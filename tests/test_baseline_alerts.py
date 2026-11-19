import json

import dpkt
import pytest
from fastapi.testclient import TestClient

from backend.evaluation_results import _device_baselines, _largest_rise
from backend.main import Settings, create_app
from dashboard import views
from detection.device_baseline import DeviceBaseline
from scripts.device_baselines import MIN_LEARNING_WINDOWS, check, learn, send_findings
from scripts.evaluate_baselines import contested_addresses, device_ranking

API = "/api/v1"
SENSOR = "aa:aa:aa:aa:aa:01"
GATEWAY = "aa:aa:aa:aa:aa:02"
GATEWAY_IP = "10.0.0.1"


def finding(**overrides):
    return {
        "kind": "behaviour_change",
        "device": SENSOR,
        "first_seen": 120.0,
        "last_seen": 240.0,
        "windows": 2,
        "evidence": [{"feature": "arp_replies", "observed": 40.0, "limit": 5.0, "typical": 1.0}],
        "baseline_active_windows": 55,
        "limited_history": False,
        "detail": "A device went above its limits.",
        **overrides,
    }


@pytest.fixture
def client(model_path, tmp_path):
    settings = Settings(model_path=model_path, database_path=tmp_path / "api.db", risk_config_path=None, api_key=None)
    with TestClient(create_app(settings), raise_server_exceptions=False) as test_client:
        yield test_client


def send(client, findings, capture="spoof.pcap"):
    return client.post(f"{API}/baseline-findings", json={"capture": capture, "findings": findings})


def test_finding_becomes_a_low_risk_lead_without_a_confidence_value(client):
    assert send(client, [finding()]).json() == {"created": 1, "duplicates": 0}
    alert = client.get(f"{API}/alerts", params={"source": "capture"}).json()[0]
    assert alert["detector"] == "device_baseline"
    assert alert["severity"] == "Low"
    assert alert["max_confidence"] is None
    assert alert["event_count"] == 2
    assert alert["explanation"]["device"] == SENSOR
    assert "not been verified" in alert["explanation"]["status_note"]


def test_unknown_device_is_lower_risk(client):
    send(client, [finding(kind="unknown_device", evidence=[], baseline_active_windows=0, limited_history=True)])
    alert = client.get(f"{API}/alerts").json()[0]
    assert alert["risk_score"] == 25
    assert alert["severity"] == "Low"


def test_sending_a_capture_twice_adds_no_duplicates(client):
    send(client, [finding()])
    assert send(client, [finding()]).json() == {"created": 0, "duplicates": 1}


def test_baseline_alerts_dont_count_as_events_and_clear_with_capture_alerts(client):
    send(client, [finding()])
    assert client.get(f"{API}/stats").json()["events_total"] == 0
    assert client.delete(f"{API}/rule-findings").status_code == 204
    assert client.get(f"{API}/alerts").json() == []


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "made_up"},
        {"device": "not-a-mac"},
        {"windows": 0},
        {"first_seen": 500.0},
        {"detail": "x" * 501},
        {"evidence": [{"feature": "made_up", "observed": 1, "limit": 1, "typical": 1}]},
        {"evidence": [{"feature": "arp_replies", "observed": -1, "limit": 1, "typical": 1}]},
        {"unexpected": "field"},
    ],
)
def test_invalid_findings_are_refused_and_nothing_is_stored(client, bad):
    assert send(client, [finding(), finding(**bad)]).status_code == 422
    assert client.get(f"{API}/alerts").json() == []


@pytest.mark.parametrize("name", ["../../etc/passwd", r"C:\Windows\system.ini", "a/b.pcap", ""])
def test_capture_names_that_look_like_paths_are_refused(client, name):
    assert send(client, [finding()], capture=name).status_code == 422


def test_baseline_writes_need_the_api_key(model_path, tmp_path):
    settings = Settings(model_path=model_path, database_path=tmp_path / "k.db", risk_config_path=None, api_key="secret")
    with TestClient(create_app(settings)) as keyed:
        assert send(keyed, [finding()]).status_code == 401


# Learning and checking files


def write_capture(path, frames_per_minute, minutes):
    frame = bytes.fromhex(GATEWAY.replace(":", "")) + bytes.fromhex(SENSOR.replace(":", "")) + b"\x08\x00" + bytes(20)
    with open(path, "wb") as handle:
        writer = dpkt.pcap.Writer(handle)
        for minute in range(minutes):
            for i in range(frames_per_minute):
                writer.writepkt(frame, ts=1000.0 + minute * 60 + i * 60 / frames_per_minute)


def test_learn_then_check_finds_a_real_rise_with_capture_times(tmp_path, client):
    normal = tmp_path / "normal.pcap"
    busy = tmp_path / "busy.pcap"
    write_capture(normal, 20, MIN_LEARNING_WINDOWS + 10)
    write_capture(busy, 400, 2)

    baseline = learn([normal])
    assert SENSOR in baseline.devices
    assert check(normal, baseline)["findings"] == []

    report = check(busy, baseline)
    assert report["findings"][0]["device"] == SENSOR
    assert report["findings"][0]["first_seen"] == 1000.0

    # A finding from the script is accepted by the API exactly as written.
    assert send_findings(client, report, api_key=None)["created"] == len(report["findings"])


def test_learning_refuses_too_little_traffic(tmp_path):
    short = tmp_path / "short.pcap"
    write_capture(short, 5, 3)
    with pytest.raises(ValueError):
        learn([short])


def test_a_learned_baseline_survives_a_file_round_trip(tmp_path):
    normal = tmp_path / "normal.pcap"
    write_capture(normal, 20, MIN_LEARNING_WINDOWS + 10)
    baseline = learn([normal])
    path = tmp_path / "baseline.json"
    baseline.save(path)
    assert DeviceBaseline.load(path) == baseline


# Corroboration


def test_a_newcomer_claiming_a_held_address_is_contested():
    usual = {GATEWAY_IP: {GATEWAY: 50}}
    seen = {GATEWAY_IP: {GATEWAY: 5, SENSOR: 20}}
    assert contested_addresses(usual, seen) == [{"ip": GATEWAY_IP, "usual_holders": [GATEWAY], "newcomers": [SENSOR]}]


def test_a_reassigned_address_with_its_old_holder_gone_isnt_contested():
    assert contested_addresses({GATEWAY_IP: {GATEWAY: 50}}, {GATEWAY_IP: {SENSOR: 20}}) == []


def test_a_new_address_isnt_contested():
    assert contested_addresses({}, {GATEWAY_IP: {SENSOR: 5, GATEWAY: 2}}) == []


def test_ranking_puts_the_most_flagged_windows_first():
    findings = [
        {"device": "a", "windows": 1},
        {"device": "b", "windows": 2},
        {"device": "a", "windows": 3},
    ]
    assert device_ranking(findings) == ["a", "b"]


# Saved results and display logic


def test_only_held_out_captures_are_loaded_from_the_latest_run(tmp_path):
    def write(name, findings):
        run = tmp_path / name
        run.mkdir()
        capture = {
            "capture": "test.pcap", "label": "benign", "hours": 1.0, "findings": findings,
            "findings_per_hour": float(findings), "devices_flagged": ["x"], "details": [],
        }
        tuning = {**capture, "capture": "dev.pcap"}
        results = {"network": "ciciomt", "held_out_captures": ["test.pcap"], "captures": [tuning, capture]}
        (run / "results.json").write_text(json.dumps(results), encoding="utf-8")

    write("baseline_final_ciciomt_20260101-000000", 9)
    write("baseline_final_ciciomt_20260102-000000", 3)
    rows = _device_baselines(tmp_path)
    assert [(row["capture"], row["findings"]) for row in rows] == [("test.pcap", 3)]


def test_other_attack_runs_are_shown_with_their_largest_rise(tmp_path):
    run = tmp_path / "baseline_other_attacks_ciciomt_20260101-000000"
    run.mkdir()
    details = [
        {"evidence": [{"observed": 50.0, "limit": 5.0}, {"observed": 90.0, "limit": 60.0}]},
        {"evidence": []},
    ]
    capture = {
        "capture": "scan.pcap", "label": "recon_port_scan", "hours": 0.1, "findings": 2,
        "findings_per_hour": 20.0, "devices_flagged": ["x", "y"], "details": details,
    }
    (run / "results.json").write_text(json.dumps({"network": "ciciomt", "captures": [capture]}), encoding="utf-8")
    rows = _device_baselines(tmp_path)
    assert [(row["capture"], row["largest_rise"]) for row in rows] == [("scan.pcap", 10.0)]


def test_largest_rise_is_empty_when_findings_have_no_evidence():
    assert _largest_rise([{"evidence": []}]) is None


def test_no_saved_baseline_results_means_none(tmp_path):
    assert _device_baselines(tmp_path) is None


def test_damaged_baseline_results_are_not_shown(tmp_path):
    run = tmp_path / "baseline_final_ciciomt_20260101-000000"
    run.mkdir()
    (run / "results.json").write_text("{}", encoding="utf-8")
    assert _device_baselines(tmp_path) is None


def baseline_alert(kind="behaviour_change"):
    return {
        "severity": "Low", "predicted_class": "Device_Behaviour_Change", "detector": "device_baseline",
        "risk_score": 35.0, "event_count": 2, "status": "open", "source": "capture",
        "last_seen_at": "2026-01-01T00:00:00.000000+00:00",
        "explanation": {"kind": kind, "device": SENSOR, "evidence": finding()["evidence"]},
    }


def test_alert_title_names_the_device_without_claiming_an_attack():
    assert views.alert_title(baseline_alert()) == f"Unusual behaviour: {SENSOR}"
    assert views.alert_title(baseline_alert("unknown_device")) == f"Unknown device: {SENSOR}"


def test_alert_table_shows_the_baseline_detected_it():
    assert views.alerts_table([baseline_alert()]).iloc[0]["Detected by"] == "Baseline"


def test_evidence_table_compares_with_the_devices_own_history():
    row = views.baseline_evidence_table(baseline_alert()["explanation"]).iloc[0]
    assert row["Measure"] == "ARP replies"
    assert (row["Observed"], row["Device's limit"], row["Device's typical"]) == (40.0, 5.0, 1.0)


def test_evidence_table_can_be_empty_for_an_unknown_device():
    assert views.baseline_evidence_table({"evidence": []}).empty


def test_results_table_uses_readable_names():
    rows = [{
        "network": "ciciomt", "capture": "a.pcap", "label": "arp_spoofing", "hours": 0.1,
        "findings": 4, "findings_per_hour": 40.0, "devices_flagged": 2, "largest_rise": 12.5,
    }]
    row = views.device_baseline_table(rows).iloc[0]
    assert (row["Network"], row["Label"]) == ("CICIoMT2024", "ARP spoofing")
    assert row["Largest rise"] == 12.5


def test_results_table_names_the_other_attacks():
    rows = [{
        "network": "ciciomt", "capture": "a.pcap", "label": "mqtt_connect_flood", "hours": 0.1,
        "findings": 4, "findings_per_hour": 40.0, "devices_flagged": 2, "largest_rise": None,
    }]
    assert views.device_baseline_table(rows).iloc[0]["Label"] == "MQTT connect flood"
