"""Transparent risk scoring.

score = base[class] * (floor + (1 - floor) * confidence)
        + min(repeat_cap, repeat_points * recent_repeats)

The base scores and thresholds are documented judgements about impact in a
clinical network, not values learned from data. Confidence only scales the
score: it says how sure the model is, not how harmful the attack would be.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from detection.config import CLASSES

SEVERITIES = ("Informational", "Low", "Medium", "High", "Critical")


@dataclass(frozen=True)
class RiskConfig:
    base_scores: Mapping[str, float] = field(default_factory=lambda: {
        "Benign": 0.0,
        # Can intercept or alter device traffic, affecting patient data integrity.
        "ARP_Spoofing": 70.0,
        # Directly threatens availability of MQTT-connected monitoring.
        "MQTT_DDoS_Connect_Flood": 75.0,
        # Precursor activity: no direct harm yet, but it often precedes attacks.
        "Recon_Port_Scan": 45.0,
    })
    confidence_floor: float = 0.5
    repeat_window_seconds: int = 60
    repeat_points: float = 2.0
    repeat_cap: float = 20.0
    # Lowest score for each severity above Informational, in SEVERITIES order.
    severity_thresholds: tuple[float, float, float, float] = (20.0, 40.0, 60.0, 80.0)
    alert_min_score: float = 20.0

    def __post_init__(self) -> None:
        if set(self.base_scores) != set(CLASSES):
            raise ValueError("base_scores must define exactly the known classes.")
        if not all(0 <= score <= 100 for score in self.base_scores.values()):
            raise ValueError("base_scores must be between 0 and 100.")
        if not 0 <= self.confidence_floor <= 1:
            raise ValueError("confidence_floor must be between 0 and 1.")
        thresholds = list(self.severity_thresholds)
        if len(thresholds) != 4 or thresholds != sorted(set(thresholds)):
            raise ValueError("severity_thresholds must be four increasing values.")


@dataclass(frozen=True)
class RiskAssessment:
    score: float
    severity: str
    should_alert: bool
    components: dict[str, Any]


def severity_for_score(score: float, config: RiskConfig) -> str:
    """Map a score to its severity band."""
    severity = SEVERITIES[0]
    for name, threshold in zip(SEVERITIES[1:], config.severity_thresholds):
        if score >= threshold:
            severity = name
    return severity


def load_risk_config(path: str | Path | None = None) -> RiskConfig:
    """Return defaults, overridden by any keys in a JSON file when one is given."""
    if path is None:
        return RiskConfig()
    overrides = json.loads(Path(path).read_text(encoding="utf-8"))
    if "severity_thresholds" in overrides:
        overrides["severity_thresholds"] = tuple(overrides["severity_thresholds"])
    return RiskConfig(**overrides)


def assess_risk(
    predicted_class: str,
    confidence: float,
    recent_repeats: int,
    config: RiskConfig,
) -> RiskAssessment:
    """Score one detection and report every part of the calculation."""
    if predicted_class not in config.base_scores:
        raise ValueError(f"Unknown class: {predicted_class!r}")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1.")
    if recent_repeats < 0:
        raise ValueError("recent_repeats cannot be negative.")

    base = config.base_scores[predicted_class]
    factor = config.confidence_floor + (1 - config.confidence_floor) * confidence
    # A benign prediction stays at zero however often it repeats.
    repeat_bonus = (
        min(config.repeat_cap, config.repeat_points * recent_repeats) if base > 0 else 0.0
    )
    score = min(100.0, base * factor + repeat_bonus)

    return RiskAssessment(
        score=round(score, 2),
        severity=severity_for_score(score, config),
        should_alert=base > 0 and score >= config.alert_min_score,
        components={
            "base_score": base,
            "confidence": round(confidence, 4),
            "confidence_factor": round(factor, 4),
            "recent_repeats": recent_repeats,
            "repeat_bonus": repeat_bonus,
        },
    )
