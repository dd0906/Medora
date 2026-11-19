"""Medora REST API. Run with: uvicorn backend.main:app --host 127.0.0.1 --port 8000"""

import hmac
import logging
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status

from backend.database import Database
from backend.evaluation_results import load_evaluation
from backend.replay import ReplayBusy, ReplayManager
from backend.schemas import (
    Alert,
    AlertSource,
    AlertStatus,
    AlertUpdate,
    BatchDetectRequest,
    BatchDetectResponse,
    DetectionResult,
    DetectRequest,
    Evaluation,
    Event,
    Health,
    LabelOutcomes,
    ModelInfo,
    ReplayRequest,
    ReplayStatus,
    BaselineFindingsRequest,
    RuleFindingsRequest,
    RuleFindingsResponse,
    Severity,
    Source,
    Statistics,
    TimelinePoint,
    TrafficRecord,
)
from backend.service import AlertNotFound, InputRecord, InvalidTransition, MonitoringService
from detection.classifier import Detector, ModelLoadError
from detection.config import API_KEY, ACTIVE_MODEL_PATH, DATABASE_PATH, EVALUATION_DIR, RISK_CONFIG_PATH
from detection.feature_validator import SchemaError
from detection.risk_scoring import load_risk_config

logger = logging.getLogger("medora.api")

TIMELINE_BUCKET_LENGTHS = {"second": 19, "minute": 16}


@dataclass(frozen=True)
class Settings:
    model_path: Path = ACTIVE_MODEL_PATH
    database_path: Path = DATABASE_PATH
    risk_config_path: str | None = RISK_CONFIG_PATH
    api_key: str | None = API_KEY
    evaluation_dir: Path = EVALUATION_DIR


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        app.state.database = Database(settings.database_path)
        try:
            detector = Detector(settings.model_path)
            app.state.service = MonitoringService(
                detector, app.state.database, load_risk_config(settings.risk_config_path)
            )
        except ModelLoadError as error:
            # Alerts stay readable without a model; detection answers 503.
            logger.error("Starting without a model: %s", error)
            app.state.service = None
        app.state.replay = ReplayManager(app.state.service) if app.state.service else None
        yield
        if app.state.replay:
            app.state.replay.stop()

    app = FastAPI(
        title="Medora",
        description="IoMT traffic classification, risk scoring, and alert management. "
        "Results are model predictions, not verified security incidents.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    _register_routes(app)
    return app


def require_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Protect endpoints that change data, when a key has been configured."""
    expected = request.app.state.settings.api_key
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or incorrect API key.")


def get_service(request: Request) -> MonitoringService:
    service = request.app.state.service
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The detection model isn't loaded.")
    return service


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_replay(request: Request) -> ReplayManager:
    replay = request.app.state.replay
    if replay is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "The detection model isn't loaded.")
    return replay


def _to_input(record: TrafficRecord) -> InputRecord:
    return InputRecord(
        features=record.features,
        src_ip=record.src_ip,
        dst_ip=record.dst_ip,
        protocol=record.protocol,
        ground_truth=record.ground_truth,
    )


def _process(service: MonitoringService, records: list[TrafficRecord], source: str) -> dict:
    try:
        return service.process([_to_input(record) for record in records], source)
    except SchemaError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error


def _register_routes(app: FastAPI) -> None:
    prefix = "/api/v1"

    @app.get("/")
    def root() -> dict[str, str]:
        return {"name": "Medora", "docs": "/docs", "api": prefix}

    @app.get(f"{prefix}/health", response_model=Health)
    def health(request: Request, database: Database = Depends(get_database)) -> dict:
        try:
            with database.session() as store:
                store.statistics()
            database_available = True
        except sqlite3.Error:
            logger.exception("Database health check failed")
            database_available = False
        model_loaded = request.app.state.service is not None
        return {
            "status": "ok" if model_loaded and database_available else "degraded",
            "model_loaded": model_loaded,
            "database_available": database_available,
        }

    @app.get(f"{prefix}/model", response_model=ModelInfo)
    def model_info(service: MonitoringService = Depends(get_service)) -> dict:
        return service.detector.info()

    @app.post(f"{prefix}/detect", response_model=DetectionResult, dependencies=[Depends(require_key)])
    def detect(body: DetectRequest, service: MonitoringService = Depends(get_service)) -> dict:
        outcome = _process(service, [body.record], body.source)
        if outcome["rejected"]:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, outcome["rejected"][0]["reason"])
        return outcome["results"][0]

    @app.post(f"{prefix}/detect/batch", response_model=BatchDetectResponse, dependencies=[Depends(require_key)])
    def detect_batch(body: BatchDetectRequest, service: MonitoringService = Depends(get_service)) -> dict:
        return _process(service, body.records, body.source)

    @app.get(f"{prefix}/alerts", response_model=list[Alert])
    def list_alerts(
        alert_status: AlertStatus | None = Query(default=None, alias="status"),
        severity: Severity | None = None,
        predicted_class: str | None = Query(default=None, max_length=64),
        source: AlertSource | None = None,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        database: Database = Depends(get_database),
    ) -> list[dict]:
        with database.session() as store:
            return store.list_alerts(alert_status, severity, predicted_class, source, limit, offset)

    @app.get(f"{prefix}/alerts/{{alert_id}}", response_model=Alert)
    def get_alert(alert_id: str, database: Database = Depends(get_database)) -> dict:
        with database.session() as store:
            alert = store.get_alert(alert_id)
        if alert is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert not found.")
        return alert

    @app.patch(f"{prefix}/alerts/{{alert_id}}", response_model=Alert, dependencies=[Depends(require_key)])
    def update_alert(alert_id: str, body: AlertUpdate, service: MonitoringService = Depends(get_service)) -> dict:
        try:
            return service.update_alert_status(alert_id, body.status)
        except AlertNotFound as error:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert not found.") from error
        except InvalidTransition as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error

    @app.get(f"{prefix}/events", response_model=list[Event])
    def list_events(
        source: Source | None = None,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        database: Database = Depends(get_database),
    ) -> list[dict]:
        with database.session() as store:
            return store.list_events(source, limit, offset)

    @app.get(f"{prefix}/stats", response_model=Statistics)
    def statistics(source: AlertSource | None = None, database: Database = Depends(get_database)) -> dict:
        with database.session() as store:
            return store.statistics(source)

    @app.get(f"{prefix}/stats/timeline", response_model=list[TimelinePoint])
    def timeline(
        source: Source | None = None,
        bucket: Literal["second", "minute"] = "second",
        database: Database = Depends(get_database),
    ) -> list[dict]:
        with database.session() as store:
            return store.timeline(source, TIMELINE_BUCKET_LENGTHS[bucket])

    @app.get(f"{prefix}/stats/outcomes", response_model=LabelOutcomes)
    def label_outcomes(source: Source | None = None, database: Database = Depends(get_database)) -> dict:
        with database.session() as store:
            return store.label_outcomes(source)

    @app.get(f"{prefix}/alerts/{{alert_id}}/outcomes", response_model=dict[str, int])
    def alert_outcomes(alert_id: str, database: Database = Depends(get_database)) -> dict[str, int]:
        with database.session() as store:
            if store.get_alert(alert_id) is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert not found.")
            return store.alert_outcomes(alert_id)

    @app.post(f"{prefix}/rule-findings", response_model=RuleFindingsResponse, dependencies=[Depends(require_key)])
    def record_rule_findings(body: RuleFindingsRequest, service: MonitoringService = Depends(get_service)) -> dict:
        findings = [finding.model_dump() for finding in body.findings]
        return service.record_rule_findings(body.capture, findings)

    @app.post(f"{prefix}/baseline-findings", response_model=RuleFindingsResponse, dependencies=[Depends(require_key)])
    def record_baseline_findings(body: BaselineFindingsRequest, service: MonitoringService = Depends(get_service)) -> dict:
        findings = [finding.model_dump() for finding in body.findings]
        return service.record_baseline_findings(body.capture, findings)

    @app.delete(f"{prefix}/rule-findings", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_key)])
    def clear_rule_findings(database: Database = Depends(get_database)) -> None:
        with database.session() as store:
            store.clear_source("capture")

    @app.get(f"{prefix}/evaluation", response_model=Evaluation)
    def evaluation(request: Request, service: MonitoringService = Depends(get_service)) -> dict:
        run_id = service.detector.metadata.get("evaluation_run")
        return load_evaluation(request.app.state.settings.evaluation_dir, run_id)

    @app.get(f"{prefix}/replay", response_model=ReplayStatus)
    def replay_status(replay: ReplayManager = Depends(get_replay)) -> dict:
        return replay.status()

    @app.post(
        f"{prefix}/replay",
        response_model=ReplayStatus,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_key)],
    )
    def start_replay(body: ReplayRequest, replay: ReplayManager = Depends(get_replay)) -> dict:
        try:
            return replay.start(body.split, body.limit, body.rate)
        except ReplayBusy as error:
            raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from error

    @app.post(f"{prefix}/replay/stop", response_model=ReplayStatus, dependencies=[Depends(require_key)])
    def stop_replay(replay: ReplayManager = Depends(get_replay)) -> dict:
        replay.stop()
        return replay.status()

    @app.delete(f"{prefix}/replay/data", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_key)])
    def clear_replay_data(replay: ReplayManager = Depends(get_replay), database: Database = Depends(get_database)) -> None:
        if replay.running:
            raise HTTPException(status.HTTP_409_CONFLICT, "Stop the replay before clearing its data.")
        with database.session() as store:
            store.clear_source("replay")


app = create_app()
