"""Human-readable alert explanations built only from what the system measured."""

from typing import Any

from detection.classifier import Detection
from detection.risk_scoring import RiskAssessment

# Documented judgements, capped at Medium: on labelled captures these rules flagged
# all 4 attack captures but also 2 of 6 benign ones (docs/results.md).
RULE_RISK_SCORES = {
    "binding_flapping": 55.0,
    "unsolicited_reply_burst": 45.0,
    "ip_mac_conflict": 30.0,
    "mac_claims_many_addresses": 20.0,
}
RULE_SUMMARIES = {
    "binding_flapping": "The device behind an IP address keeps changing, which ARP spoofing can cause.",
    "unsolicited_reply_burst": "A device sent many ARP replies nobody asked for, which ARP spoofing can cause.",
    "ip_mac_conflict": "Two devices claimed the same IP address at about the same time.",
    "mac_claims_many_addresses": "One device claimed several IP addresses. Routers and some computers do this normally.",
}

# Also judgements, kept at Low: baseline findings also fired on benign traffic and flagged
# many devices during attacks, and one benign rate missed its mark (docs/results.md).
BASELINE_RISK_SCORES = {"behaviour_change": 35.0, "unknown_device": 25.0}
BASELINE_SUMMARIES = {
    "behaviour_change": "A device became much busier than it normally is.",
    "unknown_device": "A device Medora hasn't seen before sent traffic.",
}

MODEL_STATUS_NOTE = "This is the model's prediction. It has not been verified as an attack."
RULE_STATUS_NOTE = (
    "This is a rule-based lead. It has not been verified as an attack, and a duplicate IP address can trigger it too."
)
BASELINE_STATUS_NOTE = (
    "This is a behaviour lead. It has not been verified as an attack, and normal changes in how a device is used "
    "can trigger it too."
)

CLASS_DESCRIPTIONS = {
    "ARP_Spoofing": "Looks like ARP spoofing, where one device pretends to be another on the local network.",
    "MQTT_DDoS_Connect_Flood": "Looks like an MQTT connection flood, which can overload the message server.",
    "Recon_Port_Scan": "Looks like a port scan, which searches for open services to attack.",
}


def build_rule_explanation(finding: dict[str, Any], capture: str) -> dict[str, Any]:
    """Describe an ARP rule finding as a lead, using only what the rule observed."""
    return {
        "summary": RULE_SUMMARIES[finding["rule"]],
        "status_note": RULE_STATUS_NOTE,
        "rule": finding["rule"],
        "detail": finding["detail"],
        "capture": capture,
        "claimed_ip": finding["ip"],
        "macs": finding["macs"],
        "times_observed": finding["count"],
        "capture_first_seen": finding["first_seen"],
        "capture_last_seen": finding["last_seen"],
    }


def build_baseline_explanation(finding: dict[str, Any], capture: str) -> dict[str, Any]:
    """Describe a baseline finding as a lead, using only what was measured."""
    return {
        "summary": BASELINE_SUMMARIES[finding["kind"]],
        "status_note": BASELINE_STATUS_NOTE,
        "kind": finding["kind"],
        "detail": finding["detail"],
        "capture": capture,
        "device": finding["device"],
        "windows": finding["windows"],
        "evidence": finding["evidence"],
        "baseline_active_windows": finding["baseline_active_windows"],
        "limited_history": finding["limited_history"],
        "capture_first_seen": finding["first_seen"],
        "capture_last_seen": finding["last_seen"],
    }


def build_explanation(
    detection: Detection,
    risk: RiskAssessment,
    top_features: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a structured explanation that keeps prediction and fact separate."""
    return {
        "summary": CLASS_DESCRIPTIONS[detection.predicted_class],
        "status_note": MODEL_STATUS_NOTE,
        "confidence_note": (
            f"The model gave this class a probability of {detection.confidence:.1%}. "
            "That measures the model's certainty, not the attack's severity."
        ),
        # The alert's score can rise as events join it; this is its starting point.
        "risk_components_when_opened": risk.components,
        # These describe the record that opened the alert, not every later event.
        "influential_features": top_features,
        "feature_note": (
            "Influential features show what moved this model's output for this record. "
            "They do not establish what caused the traffic."
        ),
    }
