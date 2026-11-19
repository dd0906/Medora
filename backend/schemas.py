"""Pydantic request and response models for the REST API."""

import ipaddress
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from detection.config import CLASSES, MAX_BATCH_RECORDS
from detection.device_baseline import FEATURES

AlertStatus = Literal["open", "acknowledged", "resolved", "false_positive"]
Severity = Literal["Informational", "Low", "Medium", "High", "Critical"]
Source = Literal["api", "replay"]
AlertSource = Literal["api", "replay", "capture"]
Detector = Literal["model", "arp_rules", "device_baseline"]
RuleName = Literal["binding_flapping", "unsolicited_reply_burst", "ip_mac_conflict", "mac_claims_many_addresses"]
MAC_PATTERN = r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}$"


class TrafficRecord(BaseModel):
    """One record of extracted features, with network metadata only if it is known."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, float | str | None] = Field(max_length=100)
    src_ip: str | None = None
    dst_ip: str | None = None
    protocol: str | None = Field(default=None, max_length=32)
    ground_truth: str | None = Field(
        default=None,
        description="Known label when replaying a labelled dataset. Never used by the model.",
    )

    @field_validator("src_ip", "dst_ip")
    @classmethod
    def valid_ip(cls, value: str | None) -> str | None:
        if value is not None:
            ipaddress.ip_address(value)
        return value

    @field_validator("ground_truth")
    @classmethod
    def known_class(cls, value: str | None) -> str | None:
        if value is not None and value not in CLASSES:
            raise ValueError("ground_truth must be a known class.")
        return value


class DetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record: TrafficRecord
    source: Source = "api"


class BatchDetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[TrafficRecord] = Field(min_length=1, max_length=MAX_BATCH_RECORDS)
    source: Source = "api"


class DetectionResult(BaseModel):
    index: int
    event_id: int
    predicted_class: str
    is_attack: bool
    confidence: float
    probabilities: dict[str, float]
    risk_score: float
    severity: Severity
    risk_components: dict[str, Any]
    alert_id: str | None
    new_alert: bool
    occurred_at: str


class RejectedRecord(BaseModel):
    index: int
    reason: str


class BatchDetectResponse(BaseModel):
    results: list[DetectionResult]
    rejected: list[RejectedRecord]
    model_version: str
    note: str = "Results are model predictions, not verified security incidents."


class RuleFinding(BaseModel):
    """One ARP rule finding from a capture read on the client side."""

    model_config = ConfigDict(extra="forbid")

    rule: RuleName
    first_seen: float = Field(ge=0, allow_inf_nan=False)
    last_seen: float = Field(ge=0, allow_inf_nan=False)
    ip: str | None = None
    macs: list[Annotated[str, StringConstraints(pattern=MAC_PATTERN)]] = Field(min_length=1, max_length=32)
    count: int = Field(ge=1)
    detail: str = Field(max_length=500)

    @field_validator("ip")
    @classmethod
    def valid_ip(cls, value: str | None) -> str | None:
        if value is not None:
            ipaddress.ip_address(value)
        return value

    @model_validator(mode="after")
    def ordered_times(self) -> "RuleFinding":
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen can't be before first_seen.")
        return self


class RuleFindingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A display name only. The API never opens files.
    capture: str = Field(min_length=1, max_length=255, pattern=r"^[\w .()\-]+$")
    findings: list[RuleFinding] = Field(max_length=MAX_BATCH_RECORDS)


class RuleFindingsResponse(BaseModel):
    created: int
    duplicates: int


class BaselineEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    observed: float = Field(ge=0, allow_inf_nan=False)
    limit: float = Field(ge=0, allow_inf_nan=False)
    typical: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("feature")
    @classmethod
    def known_feature(cls, value: str) -> str:
        if value not in FEATURES:
            raise ValueError("feature must be a known measure.")
        return value


class BaselineFinding(BaseModel):
    """One device baseline finding from a capture read on the client side."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["behaviour_change", "unknown_device"]
    device: Annotated[str, StringConstraints(pattern=MAC_PATTERN)]
    first_seen: float = Field(ge=0, allow_inf_nan=False)
    last_seen: float = Field(ge=0, allow_inf_nan=False)
    windows: int = Field(ge=1)
    evidence: list[BaselineEvidence] = Field(max_length=len(FEATURES))
    baseline_active_windows: int = Field(ge=0)
    limited_history: bool
    detail: str = Field(max_length=500)

    @model_validator(mode="after")
    def ordered_times(self) -> "BaselineFinding":
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen can't be before first_seen.")
        return self


class BaselineFindingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # A display name only. The API never opens files.
    capture: str = Field(min_length=1, max_length=255, pattern=r"^[\w .()\-]+$")
    findings: list[BaselineFinding] = Field(max_length=MAX_BATCH_RECORDS)


class Alert(BaseModel):
    id: str
    created_at: str
    last_seen_at: str
    source: AlertSource
    detector: Detector
    predicted_class: str
    event_count: int
    max_confidence: float | None
    risk_score: float
    severity: Severity
    src_ip: str | None
    dst_ip: str | None
    protocol: str | None
    status: AlertStatus
    status_updated_at: str | None
    explanation: dict[str, Any]


class AlertUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["acknowledged", "resolved", "false_positive"]


class Event(BaseModel):
    id: int
    occurred_at: str
    source: Source
    predicted_class: str
    is_attack: bool
    confidence: float
    probabilities: dict[str, float]
    risk_score: float
    severity: Severity
    src_ip: str | None
    dst_ip: str | None
    protocol: str | None
    ground_truth: str | None
    model_version: str
    alert_id: str | None


class Statistics(BaseModel):
    events_total: int
    events_benign: int
    events_malicious: int
    events_by_class: dict[str, int]
    alerts_by_severity: dict[str, int]
    alerts_by_status: dict[str, int]
    alerts_by_class: dict[str, int]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    version: str
    algorithm: str
    classes: list[str]
    feature_columns: list[str]
    metadata: dict[str, Any]


class ReplayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: Literal["test", "train"] = "test"
    limit: int | None = Field(default=5000, ge=1, le=1_000_000, description="Records to replay. Null replays them all.")
    rate: float | None = Field(default=500.0, gt=0, le=10_000, description="Records per second. Null means no limit.")


class ReplayStatus(BaseModel):
    mode: str = "Dataset replay: recorded records, not live network monitoring."
    state: Literal["idle", "running", "finished", "stopped", "failed"]
    split: str | None
    total: int
    processed: int
    rejected: int
    correct: int
    new_alerts: int
    rate_limit: float | None
    started_at: str | None
    finished_at: str | None
    error: str | None
    alerted_events_by_severity: dict[str, Any]


class TimelinePoint(BaseModel):
    bucket: str
    predicted_class: str
    count: int


class LabelOutcomes(BaseModel):
    by_class: list[dict[str, Any]]
    by_severity: list[dict[str, Any]]


class Evaluation(BaseModel):
    available: bool
    reason: str | None = None
    run_id: str | None = None
    selected: str | None = None
    selection_rule: str | None = None
    test_split_used_for_selection: bool | None = None
    test_metrics: dict[str, Any] | None = None
    comparison: list[dict[str, Any]] | None = None
    arp_attempt: dict[str, Any] | None = None
    outside_data: dict[str, Any] | None = None
    arp_rules: list[dict[str, Any]] | None = None
    device_baselines: list[dict[str, Any]] | None = None
    feature_effects: dict[str, Any] | None = None


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    database_available: bool
