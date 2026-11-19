"""The processing chain: validate, classify, assess risk, alert, and store."""

import hashlib
import json
import logging
import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from backend.database import Database, Store
from detection.alerts import (
    BASELINE_RISK_SCORES,
    RULE_RISK_SCORES,
    build_baseline_explanation,
    build_explanation,
    build_rule_explanation,
)
from detection.classifier import Detection, Detector
from detection.risk_scoring import RiskAssessment, RiskConfig, assess_risk, severity_for_score

logger = logging.getLogger("medora.service")

# A detection joins an active alert seen within this time rather than opening
# a new one, so one sustained attack doesn't produce thousands of alerts.
ALERT_GROUP_WINDOW_SECONDS = 300

ALLOWED_TRANSITIONS = {
    "open": {"acknowledged", "resolved", "false_positive"},
    "acknowledged": {"resolved", "false_positive"},
    "resolved": set(),
    "false_positive": set(),
}


class AlertNotFound(LookupError):
    pass


class InvalidTransition(ValueError):
    pass


@dataclass(frozen=True)
class InputRecord:
    features: dict[str, Any]
    src_ip: str | None = None
    dst_ip: str | None = None
    protocol: str | None = None
    ground_truth: str | None = None


class MonitoringService:
    def __init__(self, detector: Detector, database: Database, risk_config: RiskConfig) -> None:
        self.detector = detector
        self.database = database
        self.risk_config = risk_config

    def process(
        self,
        records: Sequence[InputRecord],
        source: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Run records through the whole chain. Stored as one transaction."""
        moment = now or datetime.now(UTC)
        frame = pd.DataFrame([record.features for record in records])
        batch = self.detector.detect(frame)

        results = []
        with self.database.session() as store:
            for detection in batch.detections:
                results.append(self._process_one(store, detection, records[detection.row], frame, source, moment))

        if batch.rejected:
            logger.warning("Rejected %d of %d records", len(batch.rejected), len(records))

        return {
            "results": results,
            "rejected": [{"index": int(row), "reason": reason} for row, reason in batch.rejected.items()],
            "model_version": self.detector.version,
        }

    def _process_one(
        self,
        store: Store,
        detection: Detection,
        record: InputRecord,
        frame: pd.DataFrame,
        source: str,
        moment: datetime,
    ) -> dict[str, Any]:
        timestamp = _iso(moment)
        repeat_since = _iso(moment - timedelta(seconds=self.risk_config.repeat_window_seconds))
        # Repeats beyond the cap cannot change the score, so there's no need to count them.
        repeats = store.count_recent_events(
            source, detection.predicted_class, record.src_ip, repeat_since,
            at_most=math.ceil(self.risk_config.repeat_cap / self.risk_config.repeat_points),
        )
        risk = assess_risk(detection.predicted_class, detection.confidence, repeats, self.risk_config)

        alert_id = None
        new_alert = False
        if risk.should_alert:
            alert_id, new_alert = self._raise_alert(store, detection, risk, record, frame, source, moment)

        event_id = store.insert_event({
            "occurred_at": timestamp,
            "source": source,
            "predicted_class": detection.predicted_class,
            "is_attack": int(detection.is_attack),
            "confidence": detection.confidence,
            "probabilities": detection.probabilities,
            "risk_score": risk.score,
            "severity": risk.severity,
            "src_ip": record.src_ip,
            "dst_ip": record.dst_ip,
            "protocol": record.protocol,
            "ground_truth": record.ground_truth,
            "model_version": self.detector.version,
            "alert_id": alert_id,
        })
        if new_alert:
            logger.info("New alert %s: %s, severity=%s", alert_id, detection.predicted_class, risk.severity)

        return {
            "index": int(detection.row),
            "event_id": event_id,
            "predicted_class": detection.predicted_class,
            "is_attack": detection.is_attack,
            "confidence": detection.confidence,
            "probabilities": detection.probabilities,
            "risk_score": risk.score,
            "severity": risk.severity,
            "risk_components": risk.components,
            "alert_id": alert_id,
            "new_alert": new_alert,
            "occurred_at": timestamp,
        }

    def _raise_alert(
        self,
        store: Store,
        detection: Detection,
        risk: RiskAssessment,
        record: InputRecord,
        frame: pd.DataFrame,
        source: str,
        moment: datetime,
    ) -> tuple[str, bool]:
        """Join a matching active alert, or open a new one. Returns (id, is_new)."""
        timestamp = _iso(moment)
        group_since = _iso(moment - timedelta(seconds=ALERT_GROUP_WINDOW_SECONDS))
        existing = store.find_active_alert(source, detection.predicted_class, record.src_ip, group_since)
        if existing:
            store.extend_alert(existing["id"], timestamp, detection.confidence, risk.score, risk.severity)
            return existing["id"], False

        alert_id = uuid.uuid4().hex
        top_features = self.detector.explain(frame.loc[[detection.row]], detection.predicted_class)
        store.create_alert({
            "id": alert_id,
            "created_at": timestamp,
            "last_seen_at": timestamp,
            "source": source,
            "predicted_class": detection.predicted_class,
            "event_count": 1,
            "max_confidence": detection.confidence,
            "risk_score": risk.score,
            "severity": risk.severity,
            "src_ip": record.src_ip,
            "dst_ip": record.dst_ip,
            "protocol": record.protocol,
            "status": "open",
            "explanation": build_explanation(detection, risk, top_features),
        })
        return alert_id, True

    def record_rule_findings(self, capture: str, findings: Sequence[dict[str, Any]]) -> dict[str, int]:
        """Store ARP rule findings as alerts. Sending a capture again adds no duplicates."""
        timestamp = _iso(datetime.now(UTC))
        created = 0
        with self.database.session() as store:
            for finding in findings:
                score = RULE_RISK_SCORES[finding["rule"]]
                created += store.create_rule_alert({
                    "id": _rule_alert_id(capture, finding),
                    "created_at": timestamp,
                    "last_seen_at": timestamp,
                    "source": "capture",
                    "predicted_class": "ARP_Spoofing",
                    "event_count": finding["count"],
                    "risk_score": score,
                    "severity": severity_for_score(score, self.risk_config),
                    "status": "open",
                    "explanation": build_rule_explanation(finding, capture),
                    "detector": "arp_rules",
                })
        logger.info("Recorded %d new rule alerts from a capture", created)
        return {"created": created, "duplicates": len(findings) - created}

    def record_baseline_findings(self, capture: str, findings: Sequence[dict[str, Any]]) -> dict[str, int]:
        """Store device baseline findings as alerts. Sending a capture again adds no duplicates."""
        timestamp = _iso(datetime.now(UTC))
        created = 0
        with self.database.session() as store:
            for finding in findings:
                score = BASELINE_RISK_SCORES[finding["kind"]]
                created += store.create_rule_alert({
                    "id": _baseline_alert_id(capture, finding),
                    "created_at": timestamp,
                    "last_seen_at": timestamp,
                    "source": "capture",
                    "predicted_class": "Device_Behaviour_Change",
                    "event_count": finding["windows"],
                    "risk_score": score,
                    "severity": severity_for_score(score, self.risk_config),
                    "status": "open",
                    "explanation": build_baseline_explanation(finding, capture),
                    "detector": "device_baseline",
                })
        logger.info("Recorded %d new baseline alerts from a capture", created)
        return {"created": created, "duplicates": len(findings) - created}

    def update_alert_status(self, alert_id: str, status: str) -> dict[str, Any]:
        with self.database.session() as store:
            alert = store.get_alert(alert_id)
            if alert is None:
                raise AlertNotFound(alert_id)
            if status not in ALLOWED_TRANSITIONS[alert["status"]]:
                raise InvalidTransition(f"An alert that is {alert['status']} cannot become {status}.")
            store.update_alert_status(alert_id, status, _iso(datetime.now(UTC)))
            logger.info("Alert %s status changed to %s", alert_id, status)
            return store.get_alert(alert_id)


def _rule_alert_id(capture: str, finding: dict[str, Any]) -> str:
    """Build a stable ID, so the same finding from the same capture is stored once."""
    identity = [capture, finding["rule"], finding["ip"], sorted(finding["macs"]), finding["first_seen"]]
    return hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()[:32]


def _baseline_alert_id(capture: str, finding: dict[str, Any]) -> str:
    identity = ["baseline", capture, finding["kind"], finding["device"], finding["first_seen"]]
    return hashlib.sha256(json.dumps(identity).encode("utf-8")).hexdigest()[:32]


def _iso(moment: datetime) -> str:
    """Fixed-width UTC timestamps, so text comparison in SQL matches time order."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
