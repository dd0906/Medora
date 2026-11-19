import threading
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest
import uvicorn
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from backend.main import Settings, create_app
from dashboard import charts, views
from dashboard.api_client import ApiError, ApiUnavailable, MedoraClient
from detection.config import CLASSES
from tests.conftest import record_for

APP_FILE = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
PAGES = ("app_pages/overview.py", "app_pages/alerts.py", "app_pages/analytics.py", "app_pages/model_performance.py")


def alert(**overrides):
    return {
        "id": "a1",
        "created_at": "2026-01-01T00:00:00.000000+00:00",
        "last_seen_at": "2026-01-01T00:00:05.123456+00:00",
        "source": "replay",
        "predicted_class": "ARP_Spoofing",
        "event_count": 3,
        "max_confidence": 0.9,
        "risk_score": 80.0,
        "severity": "Critical",
        "src_ip": None,
        "dst_ip": None,
        "protocol": None,
        "status": "open",
        **overrides,
    }


# Display logic


def test_class_labels_are_readable():
    assert views.class_label("MQTT_DDoS_Connect_Flood") == "MQTT DDoS Connect Flood"
    assert views.class_label(None) == "Unknown"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("open", ["acknowledged", "resolved", "false_positive"]),
        ("acknowledged", ["resolved", "false_positive"]),
        ("resolved", []),
        ("false_positive", []),
    ],
)
def test_actions_follow_backend_status_rules(status, expected):
    assert views.available_actions(status) == expected


def test_class_counts_include_classes_with_no_events():
    frame = views.class_counts({"events_by_class": {"Benign": 5}})
    assert frame["events"].tolist() == [5, 0, 0, 0]


def test_severity_counts_are_ordered_and_skip_informational():
    frame = views.severity_counts({"alerts_by_severity": {"Low": 2, "Critical": 1}})
    assert frame["severity"].tolist() == ["Critical", "High", "Medium", "Low"]
    assert frame["alerts"].tolist() == [1, 0, 0, 2]


def test_active_alerts_count_open_and_acknowledged_only():
    assert views.open_alert_count({"alerts_by_status": {"open": 2, "acknowledged": 1, "resolved": 9}}) == 3


def test_alert_table_shows_clean_values():
    row = views.alerts_table([alert()]).iloc[0]
    assert row["Alert"] == "ARP Spoofing"
    assert row["Status"] == "Open"
    assert row["Source"] == "Replay"
    assert row["Last seen"] == "2026-01-01 00:00:05"


def test_empty_tables_keep_their_columns():
    assert "Severity" in views.alerts_table([]).columns
    assert "Real label" in views.events_table([]).columns
    assert views.timeline_frame([]).empty


def test_timeline_fills_missing_classes_with_zero():
    frame = views.timeline_frame(
        [
            {"bucket": "2026-01-01T00:00:00", "predicted_class": "Benign", "count": 3},
            {"bucket": "2026-01-01T00:00:01", "predicted_class": "ARP_Spoofing", "count": 2},
        ]
    )
    assert len(frame) == 8
    assert frame["events"].sum() == 5
    assert set(frame["class"]) == {views.class_label(name) for name in CLASSES}


def test_timeline_fills_quiet_buckets_with_zero():
    frame = views.timeline_frame(
        [
            {"bucket": "2026-01-01T00:00:00", "predicted_class": "Benign", "count": 3},
            {"bucket": "2026-01-01T00:00:05", "predicted_class": "Benign", "count": 2},
        ]
    )
    benign = frame[frame["class"] == "Benign"]
    assert benign["events"].tolist() == [3, 0, 0, 0, 0, 2]


def test_timeline_fills_quiet_minutes():
    frame = views.timeline_frame(
        [
            {"bucket": "2026-01-01T00:00", "predicted_class": "Benign", "count": 3},
            {"bucket": "2026-01-01T00:03", "predicted_class": "Benign", "count": 2},
        ],
        bucket="minute",
    )
    assert frame[frame["class"] == "Benign"]["events"].tolist() == [3, 0, 0, 2]


def test_latest_session_leaves_out_activity_from_another_day():
    points = [
        {"bucket": "2026-01-01T09:00:00", "predicted_class": "Benign", "count": 3},
        {"bucket": "2026-01-01T09:00:05", "predicted_class": "Benign", "count": 1},
        {"bucket": "2026-01-15T18:00:00", "predicted_class": "Benign", "count": 2},
        {"bucket": "2026-01-15T18:00:01", "predicted_class": "ARP_Spoofing", "count": 4},
    ]
    session, earlier = views.latest_session(points)
    assert earlier == 1
    assert [point["bucket"] for point in session] == ["2026-01-15T18:00:00", "2026-01-15T18:00:01"]


def test_latest_session_keeps_activity_with_short_pauses():
    points = [
        {"bucket": "2026-01-01T09:00:00", "predicted_class": "Benign", "count": 3},
        {"bucket": "2026-01-01T09:05:00", "predicted_class": "Benign", "count": 1},
    ]
    session, earlier = views.latest_session(points)
    assert (len(session), earlier) == (2, 0)


def test_timeline_chart_shows_utc_as_its_title_says():
    frame = views.timeline_frame([{"bucket": "2026-01-01T19:35:00", "predicted_class": "Benign", "count": 3}])
    encoding = charts.timeline_area(frame).to_dict()["encoding"]
    assert "UTC" in encoding["x"]["title"]
    assert encoding["x"]["scale"]["type"] == "utc"
    assert encoding["tooltip"][0]["formatType"] == "utc"


def test_older_alerts_show_current_wording():
    old = {"detector": "model", "predicted_class": "Recon_Port_Scan", "explanation": {"summary": "Old wording."}}
    summary, note = views.alert_summary(old)
    assert summary.startswith("Looks like a port scan")
    assert "not been verified" in note


def test_alert_wording_falls_back_to_what_was_stored():
    unknown = {"detector": "model", "predicted_class": "Something_New", "explanation": {"summary": "Stored text."}}
    assert views.alert_summary(unknown)[0] == "Stored text."


def test_latest_session_handles_no_points():
    assert views.latest_session([]) == ([], 0)


def test_outcomes_cover_every_pair_and_compute_accuracy():
    rows = [
        {"ground_truth": "Benign", "predicted_class": "Benign", "count": 3},
        {"ground_truth": "Benign", "predicted_class": "ARP_Spoofing", "count": 1},
    ]
    assert len(views.outcome_matrix(rows)) == 16
    assert views.outcome_accuracy(rows) == 0.75
    assert views.outcome_accuracy([]) is None


def test_severity_outcomes_compute_share_of_real_attacks():
    frame = views.severity_outcomes(
        [
            {"severity": "Medium", "real_attacks": 1, "actually_benign": 3},
            {"severity": "Critical", "real_attacks": 9, "actually_benign": 1},
        ]
    )
    assert frame["Severity"].tolist() == ["Critical", "Medium"]
    assert frame["Share real"].tolist() == [0.9, 0.25]


def test_comparison_lists_active_model_first_even_when_it_scores_lower():
    rows = [
        {"model": "rf_with_iat", "macro_f1": 0.94},
        {"model": "random_forest", "macro_f1": 0.89},
    ]
    table = views.comparison_table(rows, "random_forest")
    assert table["Model"].tolist() == ["random forest", "rf with iat"]
    assert table["In use"].tolist() == [True, False]


def test_every_class_has_a_chart_color_and_charts_build():
    assert set(charts.CLASS_COLORS) == {views.class_label(name) for name in CLASSES}
    assert charts.class_bars(views.class_counts({"events_by_class": {}})).to_dict()
    assert charts.confusion_heatmap(views.outcome_matrix([])).to_dict()


# API client


def test_client_reports_an_unreachable_api():
    transport = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    client = MedoraClient(httpx.Client(transport=transport, base_url="http://test"))
    with pytest.raises(ApiUnavailable):
        client.health()


def test_client_surfaces_api_refusals():
    transport = httpx.MockTransport(lambda request: httpx.Response(409, json={"detail": "A replay is already running."}))
    client = MedoraClient(httpx.Client(transport=transport, base_url="http://test"))
    with pytest.raises(ApiError, match="already running") as error:
        client.start_replay(10, None)
    assert error.value.status_code == 409


def test_client_sends_api_key_and_omits_empty_filters():
    seen = {}

    def handler(request):
        seen.update(key=request.headers.get("X-API-Key"), query=str(request.url.query))
        return httpx.Response(200, json=[])

    client = MedoraClient(httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test"), "secret")
    client.alerts(status=None, severity="High")
    assert seen["key"] == "secret"
    assert "status" not in seen["query"]
    assert "severity=High" in seen["query"]


# Whole pages against a real API


@pytest.fixture(scope="module")
def live_api(model_path, tmp_path_factory):
    """Serve the real app with a small model and some stored data."""
    settings = Settings(
        model_path=model_path,
        database_path=tmp_path_factory.mktemp("dashboard") / "api.db",
        risk_config_path=None,
        api_key=None,
    )
    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8793, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)

    records = []
    for class_name in CLASSES:
        records += [{"features": record_for(class_name), "ground_truth": class_name}] * 3
    httpx.post(
        "http://127.0.0.1:8793/api/v1/detect/batch", json={"records": records, "source": "replay"}, timeout=30
    ).raise_for_status()

    yield "http://127.0.0.1:8793"
    server.should_exit = True
    thread.join(10)


def run_page(page, monkeypatch, url):
    monkeypatch.setenv("MEDORA_API_URL", url)
    test = AppTest.from_file(APP_FILE, default_timeout=30).run()
    if page != PAGES[0]:
        test.switch_page(page).run()
    return test


@pytest.mark.parametrize("page", PAGES)
def test_each_page_renders_without_errors_when_api_is_up(page, live_api, monkeypatch):
    test = run_page(page, monkeypatch, live_api)
    assert not test.exception


@pytest.mark.parametrize("page", PAGES)
def test_each_page_explains_when_api_is_down(page, monkeypatch):
    test = run_page(page, monkeypatch, "http://127.0.0.1:9")
    assert not test.exception
    assert any("uvicorn backend.main:app" in error.value for error in test.error)


def test_every_page_labels_replay_mode(live_api, monkeypatch):
    test = run_page(PAGES[0], monkeypatch, live_api)
    assert any("not a live network" in caption.value for caption in test.sidebar.caption)
    assert any("not confirmed attacks" in caption.value for caption in test.sidebar.caption)


def test_overview_shows_real_counts(live_api, monkeypatch):
    test = run_page(PAGES[0], monkeypatch, live_api)
    values = {metric.label: metric.value for metric in test.metric}
    assert values["Records checked"] == "12"
    assert values["Looked like attacks"] == "9"
    assert values["Open alerts"] == "3"


def test_replay_boxes_start_empty_and_start_waits_for_both(live_api, monkeypatch):
    test = run_page(PAGES[0], monkeypatch, live_api)
    records, rate = test.number_input
    start = next(button for button in test.button if button.label == "Start")
    assert (records.value, rate.value) == (None, None)
    assert start.disabled

    records.set_value(1000).run()
    assert next(button for button in test.button if button.label == "Start").disabled

    test.number_input[1].set_value(200).run()
    assert not next(button for button in test.button if button.label == "Start").disabled


def test_reset_empties_the_replay_boxes(live_api, monkeypatch):
    test = run_page(PAGES[0], monkeypatch, live_api)
    test.number_input[0].set_value(1000)
    test.number_input[1].set_value(200).run()
    next(button for button in test.button if button.label == "Reset").click().run()
    assert [box.value for box in test.number_input] == [None, None]
    assert next(button for button in test.button if button.label == "Start").disabled


def test_sidebar_no_longer_shows_a_connection_badge(live_api, monkeypatch):
    test = run_page(PAGES[0], monkeypatch, live_api)
    sidebar_text = " ".join(str(element.value) for element in test.sidebar.markdown)
    assert "API online" not in sidebar_text


def test_bar_charts_leave_room_for_the_longest_label():
    counts = views.class_counts({"events_by_class": {"Benign": 100, "MQTT_DDoS_Connect_Flood": 6054}})
    domain = charts.class_bars(counts).to_dict()["layer"][0]["encoding"]["x"]["scale"]["domain"]
    assert domain[1] > 6054


def test_overview_can_start_a_replay_request(monkeypatch, model_path, tmp_path):
    settings = Settings(model_path=model_path, database_path=tmp_path / "api.db", risk_config_path=None, api_key=None)
    with TestClient(create_app(settings)) as api:
        assert api.get("/api/v1/replay").json()["state"] == "idle"
        client = MedoraClient(api)
        assert client.replay_status()["state"] == "idle"
        frame = pd.DataFrame([{**record_for("Benign"), "label": "Benign"}])
        api.app.state.replay._loader = lambda split, limit, seed: frame
        assert client.start_replay(1, None)["state"] == "running"
        api.app.state.replay.wait(30)
        assert client.replay_status()["state"] == "finished"
