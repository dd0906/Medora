import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import Settings, create_app
from detection.config import CLASSES, MAX_BATCH_RECORDS, MODEL_FEATURE_COLUMNS
from simulator.traffic_simulator import replay
from tests.conftest import record_for

API = "/api/v1"


def make_client(model_path, tmp_path, **overrides):
    settings = Settings(
        model_path=model_path,
        database_path=tmp_path / "api.db",
        risk_config_path=None,
        api_key=overrides.pop("api_key", None),
    )
    return TestClient(create_app(settings), raise_server_exceptions=False)


@pytest.fixture
def client(model_path, tmp_path):
    with make_client(model_path, tmp_path) as test_client:
        yield test_client


def detect(client, class_name="ARP_Spoofing", **fields):
    return client.post(f"{API}/detect", json={"record": {"features": record_for(class_name), **fields}})


def test_root_and_docs_are_available(client):
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200


def test_health_reports_ok(client):
    assert client.get(f"{API}/health").json() == {
        "status": "ok", "model_loaded": True, "database_available": True,
    }


def test_missing_model_degrades_instead_of_crashing(tmp_path):
    with make_client(tmp_path / "absent.joblib", tmp_path) as test_client:
        assert test_client.get(f"{API}/health").json()["status"] == "degraded"
        assert test_client.get(f"{API}/alerts").status_code == 200
        response = detect(test_client)
        assert response.status_code == 503
        assert "absent" not in response.text


def test_model_info_hides_paths(client, model_path):
    response = client.get(f"{API}/model")
    assert response.status_code == 200
    assert response.json()["feature_columns"] == list(MODEL_FEATURE_COLUMNS)
    assert str(model_path.parent) not in response.text


def test_attack_creates_stored_alert_with_explanation(client):
    body = detect(client).json()
    assert body["predicted_class"] == "ARP_Spoofing"
    assert body["new_alert"] is True

    alert = client.get(f"{API}/alerts/{body['alert_id']}").json()
    assert alert["status"] == "open"
    assert alert["src_ip"] is None
    assert "not been verified" in alert["explanation"]["status_note"]
    assert alert["explanation"]["influential_features"][0]["feature"] == "Rate"


def test_benign_creates_event_but_no_alert(client):
    body = detect(client, "Benign").json()
    assert body["alert_id"] is None
    assert body["severity"] == "Informational"
    assert client.get(f"{API}/alerts").json() == []
    assert client.get(f"{API}/stats").json()["events_benign"] == 1


def test_repeats_join_one_alert_and_raise_risk(client):
    first = detect(client).json()
    second = detect(client).json()
    assert second["alert_id"] == first["alert_id"]
    assert second["new_alert"] is False
    assert second["risk_score"] > first["risk_score"]

    alert = client.get(f"{API}/alerts/{first['alert_id']}").json()
    assert alert["event_count"] == 2
    assert alert["risk_score"] == second["risk_score"]


def test_different_source_address_gets_its_own_alert(client):
    first = detect(client, src_ip="10.0.0.1", dst_ip="10.0.0.2", protocol="TCP").json()
    second = detect(client, src_ip="10.0.0.9").json()
    assert first["alert_id"] != second["alert_id"]
    assert client.get(f"{API}/alerts/{first['alert_id']}").json()["dst_ip"] == "10.0.0.2"


def test_replay_and_api_events_are_never_merged(client):
    records = {"features": record_for("ARP_Spoofing")}
    direct = client.post(f"{API}/detect", json={"record": records}).json()
    replayed = client.post(f"{API}/detect", json={"record": records, "source": "replay"}).json()
    assert direct["alert_id"] != replayed["alert_id"]
    assert client.get(f"{API}/stats", params={"source": "replay"}).json()["events_total"] == 1


@pytest.mark.parametrize(
    "features",
    [
        {"Rate": 1.0},
        {**record_for("Benign"), "extra": 1.0},
    ],
)
def test_wrong_schema_returns_422(client, features):
    response = client.post(f"{API}/detect", json={"record": {"features": features}})
    assert response.status_code == 422


def test_complete_dataset_record_including_iat_is_accepted(client):
    features = {**record_for("ARP_Spoofing"), "IAT": 169402413.0}
    response = client.post(f"{API}/detect", json={"record": {"features": features}})
    assert response.status_code == 200
    assert response.json()["predicted_class"] == "ARP_Spoofing"


def test_single_bad_value_returns_422(client):
    features = {**record_for("Benign"), "Rate": "not-a-number"}
    assert client.post(f"{API}/detect", json={"record": {"features": features}}).status_code == 422


def test_value_too_large_for_the_model_returns_422_not_a_crash(client):
    features = {**record_for("Benign"), "Rate": 1e308}
    response = client.post(f"{API}/detect", json={"record": {"features": features}})
    assert response.status_code == 422
    assert "Rate" in response.json()["detail"]


def test_largest_usable_value_is_still_scored(client):
    features = {**record_for("Benign"), "Rate": 3e38}
    assert client.post(f"{API}/detect", json={"record": {"features": features}}).status_code == 200


@pytest.mark.parametrize("bad", [None, 1e308])
def test_batch_scores_good_rows_and_reports_bad_rows(client, bad):
    records = [
        {"features": record_for("Benign")},
        {"features": {**record_for("Benign"), "Rate": bad}},
        {"features": record_for("Recon_Port_Scan")},
    ]
    body = client.post(f"{API}/detect/batch", json={"records": records}).json()
    assert [r["index"] for r in body["results"]] == [0, 2]
    assert body["rejected"][0]["index"] == 1
    assert "Rate" in body["rejected"][0]["reason"]
    assert client.get(f"{API}/stats").json()["events_total"] == 2


def test_batch_size_limits_are_enforced(client):
    assert client.post(f"{API}/detect/batch", json={"records": []}).status_code == 422
    too_many = [{"features": {"Rate": 1.0}}] * (MAX_BATCH_RECORDS + 1)
    assert client.post(f"{API}/detect/batch", json={"records": too_many}).status_code == 422


@pytest.mark.parametrize(
    "fields",
    [
        {"src_ip": "not-an-ip"},
        {"ground_truth": "Made_Up_Class"},
        {"file_path": "C:/Windows/system.ini"},
    ],
)
def test_invalid_record_fields_are_refused(client, fields):
    assert detect(client, **fields).status_code == 422


def test_alert_lifecycle(client):
    alert_id = detect(client).json()["alert_id"]
    url = f"{API}/alerts/{alert_id}"

    acknowledged = client.patch(url, json={"status": "acknowledged"})
    assert acknowledged.json()["status"] == "acknowledged"
    assert acknowledged.json()["status_updated_at"]

    assert client.patch(url, json={"status": "resolved"}).json()["status"] == "resolved"
    assert client.patch(url, json={"status": "acknowledged"}).status_code == 409


def test_analyst_can_mark_false_positive(client):
    alert_id = detect(client).json()["alert_id"]
    response = client.patch(f"{API}/alerts/{alert_id}", json={"status": "false_positive"})
    assert response.json()["status"] == "false_positive"


def test_detection_after_resolution_opens_a_new_alert(client):
    first = detect(client).json()["alert_id"]
    client.patch(f"{API}/alerts/{first}", json={"status": "resolved"})
    assert detect(client).json()["alert_id"] != first


def test_alert_status_cannot_be_reset_to_open(client):
    alert_id = detect(client).json()["alert_id"]
    assert client.patch(f"{API}/alerts/{alert_id}", json={"status": "open"}).status_code == 422


def test_unknown_alert_returns_404(client):
    assert client.get(f"{API}/alerts/missing").status_code == 404
    assert client.patch(f"{API}/alerts/missing", json={"status": "resolved"}).status_code == 404


def test_alert_filters_and_paging(client):
    detect(client, "ARP_Spoofing")
    detect(client, "Recon_Port_Scan")
    assert len(client.get(f"{API}/alerts").json()) == 2
    only = client.get(f"{API}/alerts", params={"predicted_class": "Recon_Port_Scan"}).json()
    assert [a["predicted_class"] for a in only] == ["Recon_Port_Scan"]
    assert len(client.get(f"{API}/alerts", params={"limit": 1}).json()) == 1
    assert client.get(f"{API}/alerts", params={"severity": "Nonsense"}).status_code == 422
    assert client.get(f"{API}/alerts", params={"limit": 100000}).status_code == 422


def test_events_are_listed_newest_first(client):
    detect(client, "Benign")
    detect(client, "ARP_Spoofing")
    events = client.get(f"{API}/events").json()
    assert [e["predicted_class"] for e in events] == ["ARP_Spoofing", "Benign"]


def test_api_key_protects_writes_only(model_path, tmp_path):
    with make_client(model_path, tmp_path, api_key="secret") as test_client:
        record = {"record": {"features": record_for("Benign")}}
        assert test_client.post(f"{API}/detect", json=record).status_code == 401
        wrong = {"X-API-Key": "wrong"}
        assert test_client.post(f"{API}/detect", json=record, headers=wrong).status_code == 401
        right = {"X-API-Key": "secret"}
        assert test_client.post(f"{API}/detect", json=record, headers=right).status_code == 200
        assert test_client.get(f"{API}/alerts").status_code == 200


def test_replay_runs_the_whole_chain_through_the_api(client):
    """Records flow from the simulator through to stored, labelled alerts."""
    rows = []
    for class_name in CLASSES:
        rows += [{**record_for(class_name), "label": class_name}] * 3
    frame = pd.DataFrame(rows)

    summary = replay(client, frame, batch_size=5, rate=None)

    assert summary["records_scored"] == 12
    assert summary["records_rejected"] == 0
    assert summary["correct"] == 12
    assert summary["new_alerts"] == 3
    assert summary["alerted_events_by_severity"]

    stats = client.get(f"{API}/stats", params={"source": "replay"}).json()
    assert stats["events_total"] == 12
    assert stats["events_malicious"] == 9
    events = client.get(f"{API}/events", params={"source": "replay"}).json()
    assert all(e["ground_truth"] == e["predicted_class"] for e in events)
    assert all(a["source"] == "replay" for a in client.get(f"{API}/alerts").json())


def test_replay_stops_on_a_server_error(tmp_path):
    with make_client(tmp_path / "absent.joblib", tmp_path) as no_model_client:
        frame = pd.DataFrame([{**record_for("Benign"), "label": "Benign"}])
        with pytest.raises(httpx.HTTPStatusError):
            replay(no_model_client, frame, batch_size=1, rate=None)


@pytest.fixture
def replay_client(client):
    """A client whose replay reads small in-memory records instead of the dataset."""

    def loader(split, limit, seed):
        rows = []
        for class_name in CLASSES:
            rows += [{**record_for(class_name), "label": class_name}] * 5
        return pd.DataFrame(rows).head(limit)

    client.app.state.replay._loader = loader
    return client


def test_replay_status_starts_idle(client):
    assert client.get(f"{API}/replay").json()["state"] == "idle"


def test_replay_runs_in_the_background_and_stores_results(replay_client):
    response = replay_client.post(f"{API}/replay", json={"limit": 20, "rate": None})
    assert response.status_code == 202

    replay_client.app.state.replay.wait(30)
    status = replay_client.get(f"{API}/replay").json()

    assert status["state"] == "finished"
    assert status["total"] == status["processed"] == status["correct"] == 20
    assert status["new_alerts"] == 3
    assert "not live" in status["mode"]
    assert replay_client.get(f"{API}/stats", params={"source": "replay"}).json()["events_total"] == 20


def test_second_replay_is_refused_while_one_runs(replay_client):
    assert replay_client.post(f"{API}/replay", json={"limit": 20, "rate": 1}).status_code == 202
    assert replay_client.post(f"{API}/replay", json={"limit": 20}).status_code == 409
    assert replay_client.delete(f"{API}/replay/data").status_code == 409

    replay_client.post(f"{API}/replay/stop")
    replay_client.app.state.replay.wait(30)
    assert replay_client.get(f"{API}/replay").json()["state"] == "stopped"


def test_failed_replay_reports_failure_without_internal_details(client):
    def loader(split, limit, seed):
        raise OSError(r"C:\secret\path missing")

    client.app.state.replay._loader = loader
    client.post(f"{API}/replay", json={"limit": 5})
    client.app.state.replay.wait(30)

    status = client.get(f"{API}/replay")
    assert status.json()["state"] == "failed"
    assert "secret" not in status.text


@pytest.mark.parametrize(
    "body",
    [
        {"split": "C:/Windows/system.ini"},
        {"path": "data.csv"},
        {"limit": 0},
        {"rate": -1},
    ],
)
def test_invalid_replay_requests_are_refused(client, body):
    assert client.post(f"{API}/replay", json=body).status_code == 422


def test_clearing_replay_data_keeps_api_data(replay_client):
    detect(replay_client)
    replay_client.post(f"{API}/replay", json={"limit": 20, "rate": None})
    replay_client.app.state.replay.wait(30)

    assert replay_client.delete(f"{API}/replay/data").status_code == 204

    stats = replay_client.get(f"{API}/stats").json()
    assert stats["events_total"] == 1
    assert [a["source"] for a in replay_client.get(f"{API}/alerts").json()] == ["api"]


def test_replay_writes_need_the_api_key(model_path, tmp_path):
    with make_client(model_path, tmp_path, api_key="secret") as test_client:
        assert test_client.post(f"{API}/replay", json={"limit": 1}).status_code == 401
        assert test_client.post(f"{API}/replay/stop").status_code == 401
        assert test_client.delete(f"{API}/replay/data").status_code == 401
        assert test_client.get(f"{API}/replay").status_code == 200


def test_timeline_and_outcomes_reflect_replayed_labels(replay_client):
    replay_client.post(f"{API}/replay", json={"limit": 20, "rate": None})
    replay_client.app.state.replay.wait(30)

    timeline = replay_client.get(f"{API}/stats/timeline", params={"source": "replay"}).json()
    assert sum(point["count"] for point in timeline) == 20
    assert len(replay_client.get(f"{API}/stats/timeline", params={"bucket": "minute"}).json()) >= 4
    assert replay_client.get(f"{API}/stats/timeline", params={"bucket": "year"}).status_code == 422

    outcomes = replay_client.get(f"{API}/stats/outcomes", params={"source": "replay"}).json()
    assert sum(row["count"] for row in outcomes["by_class"]) == 20
    assert all(row["ground_truth"] == row["predicted_class"] for row in outcomes["by_class"])
    assert sum(row["real_attacks"] for row in outcomes["by_severity"]) == 15
    assert sum(row["actually_benign"] for row in outcomes["by_severity"]) == 0


def test_alert_outcomes_count_events_by_label(replay_client):
    replay_client.post(f"{API}/replay", json={"limit": 20, "rate": None})
    replay_client.app.state.replay.wait(30)

    alert = replay_client.get(f"{API}/alerts", params={"predicted_class": "ARP_Spoofing"}).json()[0]
    assert replay_client.get(f"{API}/alerts/{alert['id']}/outcomes").json() == {"ARP_Spoofing": 5}
    assert replay_client.get(f"{API}/alerts/missing/outcomes").status_code == 404


def test_outcomes_without_labels_are_empty(client):
    detect(client)
    assert client.get(f"{API}/stats/outcomes").json() == {"by_class": [], "by_severity": []}


def test_evaluation_explains_when_model_has_no_run(client, model_path):
    body = client.get(f"{API}/evaluation").json()
    assert body["available"] is False
    assert str(model_path.parent) not in str(body)
