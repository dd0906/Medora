"""Turns API responses into display tables. Contains no Streamlit calls."""

from typing import Any

import pandas as pd

from backend.service import ALLOWED_TRANSITIONS
from detection.alerts import (
    BASELINE_STATUS_NOTE,
    BASELINE_SUMMARIES,
    CLASS_DESCRIPTIONS,
    MODEL_STATUS_NOTE,
    RULE_STATUS_NOTE,
    RULE_SUMMARIES,
)
from detection.config import CLASSES
from detection.feature_descriptions import feature_meaning, feature_name
from detection.risk_scoring import SEVERITIES

SEVERITY_ICONS = {
    "Critical": ":material/error:",
    "High": ":material/warning:",
    "Medium": ":material/report:",
    "Low": ":material/info:",
    "Informational": ":material/check_circle:",
}
SEVERITY_BADGE_COLORS = {
    "Critical": "red",
    "High": "orange",
    "Medium": "yellow",
    "Low": "blue",
    "Informational": "gray",
}
STATUS_LABELS = {
    "open": "Open",
    "acknowledged": "Acknowledged",
    "resolved": "Resolved",
    "false_positive": "False positive",
}
SOURCE_LABELS = {"replay": "Replay", "api": "API", "capture": "Capture"}
RULE_LABELS = {
    "binding_flapping": "MAC keeps switching",
    "unsolicited_reply_burst": "Burst of unrequested replies",
    "ip_mac_conflict": "Two MACs claim one IP",
    "mac_claims_many_addresses": "One MAC claims many IPs",
}
BASELINE_LABELS = {"behaviour_change": "Unusual behaviour", "unknown_device": "Unknown device"}
CAPTURE_LABELS = {
    "arp_spoofing": "ARP spoofing",
    "benign": "Benign",
    "recon_port_scan": "Port scan",
    "mqtt_connect_flood": "MQTT connect flood",
}
DETECTOR_LABELS = {"model": "Model", "arp_rules": "Rules", "device_baseline": "Baseline"}
FEATURE_LABELS = {
    "frames_sent": "Frames sent",
    "bytes_sent": "Bytes sent",
    "frames_received": "Frames received",
    "peer_macs": "Devices contacted",
    "source_ips": "Source IPs used",
    "destination_ips": "Destination IPs",
    "destination_ports": "Destination ports",
    "tcp_syns": "New TCP connections",
    "arp_requests": "ARP requests",
    "arp_replies": "ARP replies",
}
TIMELINE_FREQUENCIES = {"second": "s", "minute": "min"}
# Activity this far apart is treated as separate sessions, such as replays on different days.
SESSION_GAP = pd.Timedelta(minutes=10)
ACTION_LABELS = {
    "acknowledged": "Acknowledge",
    "resolved": "Resolve",
    "false_positive": "Mark false positive",
}


def class_label(name: str | None) -> str:
    """Make a class name readable, such as MQTT_DDoS_Connect_Flood."""
    return name.replace("_", " ") if name else "Unknown"


def available_actions(status: str) -> list[str]:
    """Statuses an alert can move to next, in a stable order."""
    allowed = ALLOWED_TRANSITIONS.get(status, set())
    return [action for action in ACTION_LABELS if action in allowed]


def class_counts(statistics: dict[str, Any]) -> pd.DataFrame:
    """Event counts per class, including classes with no events yet."""
    counts = statistics["events_by_class"]
    return pd.DataFrame({
        "class": [class_label(name) for name in CLASSES],
        "events": [counts.get(name, 0) for name in CLASSES],
    })


def severity_counts(statistics: dict[str, Any]) -> pd.DataFrame:
    """Alert counts per severity, highest severity first."""
    counts = statistics["alerts_by_severity"]
    ordered = [name for name in reversed(SEVERITIES) if name != "Informational"]
    return pd.DataFrame({"severity": ordered, "alerts": [counts.get(name, 0) for name in ordered]})


def open_alert_count(statistics: dict[str, Any]) -> int:
    statuses = statistics["alerts_by_status"]
    return statuses.get("open", 0) + statuses.get("acknowledged", 0)


def alerts_table(alerts: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Severity": alert["severity"],
                "Alert": alert_title(alert),
                "Detected by": DETECTOR_LABELS.get(alert.get("detector"), "Model"),
                "Risk": alert["risk_score"],
                "Events": alert["event_count"],
                "Status": STATUS_LABELS.get(alert["status"], alert["status"]),
                "Source": SOURCE_LABELS.get(alert["source"], alert["source"]),
                "Last seen": _short_time(alert["last_seen_at"]),
            }
            for alert in alerts
        ],
        columns=["Severity", "Alert", "Detected by", "Risk", "Events", "Status", "Source", "Last seen"],
    )


def alert_title(alert: dict[str, Any]) -> str:
    """Name a model alert by predicted class, and other alerts by what they observed."""
    if alert.get("detector") == "device_baseline":
        explanation = alert["explanation"]
        return f"{BASELINE_LABELS.get(explanation['kind'], explanation['kind'])}: {explanation['device']}"
    if alert.get("detector") == "arp_rules":
        rule = alert["explanation"]["rule"]
        return f"Possible ARP spoofing: {RULE_LABELS.get(rule, rule)}"
    return class_label(alert["predicted_class"])


def alert_summary(alert: dict[str, Any]) -> tuple[str, str]:
    """Current wording for an alert's summary and status note.

    Alerts store their wording when they open, so older alerts would otherwise keep old text.
    """
    explanation = alert["explanation"]
    detector = alert.get("detector")
    if detector == "device_baseline":
        summary, note = BASELINE_SUMMARIES.get(explanation.get("kind")), BASELINE_STATUS_NOTE
    elif detector == "arp_rules":
        summary, note = RULE_SUMMARIES.get(explanation.get("rule")), RULE_STATUS_NOTE
    else:
        summary, note = CLASS_DESCRIPTIONS.get(alert["predicted_class"]), MODEL_STATUS_NOTE
    return summary or explanation.get("summary", ""), note


def reason_sentence(predicted_class: str, features: list[dict[str, Any]], count: int = 3) -> str:
    """One plain sentence naming what pushed the model towards its answer."""
    names = [feature_name(item["feature"]) for item in features[:count]]
    if not names:
        return "The model can't show what drove this prediction."
    listed = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    return f"The model chose {class_label(predicted_class)} mainly because of: {listed}."


def effects_frame(features: list[dict[str, Any]], effect_key: str = "contribution") -> pd.DataFrame:
    """Readable feature effects, biggest first, in percentage points of probability."""
    return pd.DataFrame(
        [
            {
                "Feature": feature_name(item["feature"]),
                "Meaning": feature_meaning(item["feature"]),
                "Push": item[effect_key] * 100,
                "Value": item.get("value"),
                "Column": item["feature"],
            }
            for item in features
        ],
        columns=["Feature", "Meaning", "Push", "Value", "Column"],
    )


def performance_summary(metrics: dict[str, Any]) -> str:
    """A plain sentence on what the model is good and weak at, from its test scores."""
    scores = {name: values["f1"] for name, values in metrics["per_class"].items() if name != "Benign"}
    strong = [class_label(name) for name, f1 in scores.items() if f1 >= 0.95]
    weakest = min(scores, key=scores.get)
    parts = []
    if strong:
        parts.append(f"very good at spotting {' and '.join(strong)}")
    if scores[weakest] < 0.95:
        parts.append(f"weaker at {class_label(weakest)}, which it often confuses with normal traffic")
    return f"In short: {', but '.join(parts)}." if parts else ""


def rule_evidence_table(explanation: dict[str, Any]) -> pd.DataFrame:
    """Show exactly what an ARP rule observed in a capture."""
    span = explanation["capture_last_seen"] - explanation["capture_first_seen"]
    return pd.DataFrame(
        [
            {"Field": "Capture", "Value": explanation["capture"]},
            {"Field": "Observation", "Value": explanation["detail"]},
            {"Field": "Claimed IP address", "Value": explanation["claimed_ip"] or "Not specific to one address"},
            {"Field": "MAC addresses", "Value": ", ".join(explanation["macs"])},
            {"Field": "Times observed", "Value": f"{explanation['times_observed']:,}"},
            {"Field": "Time span in capture", "Value": f"{span:,.1f} seconds"},
        ]
    )


def baseline_evidence_table(explanation: dict[str, Any]) -> pd.DataFrame:
    """Each measure's highest value per window, next to that device's own history."""
    return pd.DataFrame(
        [
            {
                "Measure": FEATURE_LABELS.get(item["feature"], item["feature"]),
                "Observed": item["observed"],
                "Device's limit": item["limit"],
                "Device's typical": item["typical"],
            }
            for item in explanation["evidence"]
        ],
        columns=["Measure", "Observed", "Device's limit", "Device's typical"],
    )


def device_baseline_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Held-out baseline results for each labelled capture."""
    networks = {"ciciomt": "CICIoMT2024", "ciciot2023": "CICIoT2023"}
    return pd.DataFrame(
        [
            {
                "Network": networks.get(row["network"], row["network"]),
                "Capture": row["capture"],
                "Label": CAPTURE_LABELS.get(row["label"], row["label"]),
                "Hours": row["hours"],
                "Findings": row["findings"],
                "Findings per hour": row["findings_per_hour"],
                "Devices flagged": row["devices_flagged"],
                "Largest rise": row["largest_rise"],
            }
            for row in rows
        ]
    )


def arp_rule_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Labelled capture results, with held-out test captures first."""
    groups = {
        "ciciomt_test": "CICIoMT2024 test",
        "ciciomt_train": "CICIoMT2024 train",
        "ciciot2023": "CICIoT2023",
        "ciciot2023_validation": "CICIoT2023 validation",
    }
    order = list(groups)
    ordered = sorted(rows, key=lambda row: (order.index(row["group"]) if row["group"] in groups else len(order), row["capture"]))
    return pd.DataFrame(
        [
            {
                "Set": groups.get(row["group"], row["group"]),
                "Capture": row["capture"],
                "Label": "ARP spoofing" if row["label"] == "arp_spoofing" else "Benign",
                "Flagged": row["flagged"],
                "Strong findings": row["primary_findings"],
                "Hours": row["hours"],
                "First finding after (s)": row["seconds_to_first_primary_finding"],
            }
            for row in ordered
        ]
    )


def events_table(events: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Time": _short_time(event["occurred_at"]),
                "Prediction": class_label(event["predicted_class"]),
                "Confidence": event["confidence"],
                "Risk": event["risk_score"],
                "Severity": event["severity"],
                "Real label": class_label(event["ground_truth"]) if event["ground_truth"] else "None",
                "Source": "Replay" if event["source"] == "replay" else "API",
            }
            for event in events
        ],
        columns=["Time", "Prediction", "Confidence", "Risk", "Severity", "Real label", "Source"],
    )


def timeline_frame(points: list[dict[str, Any]], bucket: str = "second") -> pd.DataFrame:
    """Event counts over time, with one row for each class in each time bucket.

    Quiet buckets are filled with zeros, so the chart doesn't draw traffic across them.
    """
    if not points:
        return pd.DataFrame(columns=["time", "class", "events"])
    frame = pd.DataFrame(points)
    frame["time"] = pd.to_datetime(frame["bucket"], utc=True)
    pivot = frame.pivot_table(index="time", columns="predicted_class", values="count", fill_value=0)
    every_bucket = pd.date_range(pivot.index.min(), pivot.index.max(), freq=TIMELINE_FREQUENCIES[bucket])
    pivot = pivot.reindex(index=every_bucket, columns=list(CLASSES), fill_value=0).rename_axis("time")
    long = pivot.reset_index().melt(id_vars="time", var_name="class", value_name="events")
    long["class"] = long["class"].map(class_label)
    return long.sort_values("time").reset_index(drop=True)


def latest_session(points: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the most recent stretch of activity, and count the earlier ones left out.

    Replays from different days would otherwise share one time axis, mostly empty.
    """
    if not points:
        return [], 0
    times = sorted({pd.Timestamp(point["bucket"]) for point in points})
    starts = [times[0]] + [later for earlier, later in zip(times, times[1:]) if later - earlier > SESSION_GAP]
    latest = starts[-1]
    return [point for point in points if pd.Timestamp(point["bucket"]) >= latest], len(starts) - 1


def outcome_matrix(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Long table of known label against prediction, covering every class pair."""
    counts = {(row["ground_truth"], row["predicted_class"]): row["count"] for row in rows}
    return pd.DataFrame(
        [
            {"Actual": class_label(actual), "Predicted": class_label(predicted), "Records": counts.get((actual, predicted), 0)}
            for actual in CLASSES
            for predicted in CLASSES
        ]
    )


def outcome_accuracy(rows: list[dict[str, Any]]) -> float | None:
    total = sum(row["count"] for row in rows)
    if not total:
        return None
    return sum(row["count"] for row in rows if row["ground_truth"] == row["predicted_class"]) / total


def severity_outcomes(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Share of alerted events that were real attacks, highest severity first."""
    by_severity = {row["severity"]: row for row in rows}
    table = []
    for severity in reversed(SEVERITIES):
        row = by_severity.get(severity)
        if row:
            real, benign = row["real_attacks"], row["actually_benign"]
            table.append({
                "Severity": severity,
                "Real attacks": real,
                "Actually normal": benign,
                "Share real": real / (real + benign) if real + benign else None,
            })
    return pd.DataFrame(table, columns=["Severity", "Real attacks", "Actually normal", "Share real"])


def confusion_matrix_long(matrix: list[list[int]], classes: list[str] | tuple[str, ...] = CLASSES) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Actual": class_label(actual), "Predicted": class_label(predicted), "Records": matrix[i][j]}
            for i, actual in enumerate(classes)
            for j, predicted in enumerate(classes)
        ]
    )


def per_class_table(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Type": class_label(name),
                "Precision": values["precision"],
                "Recall": values["recall"],
                "F1": values["f1"],
                "Records": values["support"],
            }
            for name, values in metrics["per_class"].items()
        ]
    )


def comparison_table(rows: list[dict[str, Any]], selected: str) -> pd.DataFrame:
    """Candidate models, with the active one first and the rest by test macro F1."""
    table = pd.DataFrame(
        [
            {
                "Model": row["model"].replace("_", " "),
                "In use": row["model"] == selected,
                "Macro F1 (training check)": row.get("cv_macro_f1"),
                "Macro F1 (test)": row.get("macro_f1"),
                "Accuracy (test)": row.get("accuracy"),
                "ARP F1 (training check)": row.get("cv_arp_f1"),
                "ARP F1 (test)": row.get("arp_f1"),
                "False alarms": row.get("benign_false_alarm_rate"),
            }
            for row in rows
        ]
    )
    return table.sort_values(["In use", "Macro F1 (test)"], ascending=False).reset_index(drop=True)


def risk_components_table(components: dict[str, Any]) -> pd.DataFrame:
    labels = {
        "base_score": "Starting score for this attack type",
        "confidence": "Model confidence",
        "confidence_factor": "Confidence adjustment",
        "recent_repeats": "Similar detections in the last minute",
        "repeat_bonus": "Extra points for repeats",
    }
    return pd.DataFrame(
        [{"Part": label, "Value": components[key]} for key, label in labels.items() if key in components]
    )


def _short_time(timestamp: str) -> str:
    """Show a UTC timestamp to the second."""
    return timestamp[:19].replace("T", " ")
