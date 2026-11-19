"""Runs one dataset replay at a time in the background.

This replays recorded dataset rows. It captures and sends no network traffic.
"""

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from backend.service import InputRecord, MonitoringService
from detection.config import SEED
from simulator.traffic_simulator import build_batches, load_records, summarize

logger = logging.getLogger("medora.replay")

BATCH_SIZE = 200


class ReplayBusy(RuntimeError):
    """A replay is already running."""


class ReplayManager:
    def __init__(
        self,
        service: MonitoringService,
        loader: Callable[[str, int | None, int], pd.DataFrame] = load_records,
    ) -> None:
        self._service = service
        self._loader = loader
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = _empty_status()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def start(self, split: str, limit: int | None, rate: float | None) -> dict[str, Any]:
        with self._lock:
            if self.running:
                raise ReplayBusy("A replay is already running.")
            self._cancel.clear()
            self._status = {
                **_empty_status(),
                "state": "running",
                "split": split,
                "rate_limit": rate,
                "started_at": _now(),
            }
            self._thread = threading.Thread(target=self._run, args=(split, limit, rate), daemon=True)
            self._thread.start()
        return self.status()

    def stop(self) -> None:
        """Ask a running replay to stop after its current batch."""
        self._cancel.set()

    def wait(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _update(self, **changes: Any) -> None:
        with self._lock:
            self._status.update(changes)

    def _run(self, split: str, limit: int | None, rate: float | None) -> None:
        try:
            frame = self._loader(split, limit, SEED)
            self._update(total=len(frame))
            logger.info("Dataset replay started: %d %s records", len(frame), split)

            sent, results = [], []
            handled = rejected = 0
            started = time.perf_counter()
            for batch in build_batches(frame, BATCH_SIZE):
                if self._cancel.is_set():
                    break
                outcome = self._service.process([InputRecord(**record) for record in batch], "replay")

                scored = {result["index"] for result in outcome["results"]}
                sent += [record for index, record in enumerate(batch) if index in scored]
                results += outcome["results"]
                handled += len(batch)
                rejected += len(outcome["rejected"])
                self._update(
                    processed=handled,
                    rejected=rejected,
                    correct=sum(r["predicted_class"] == s["ground_truth"] for r, s in zip(results, sent)),
                    new_alerts=sum(r["new_alert"] for r in results),
                )

                if rate:
                    ahead = handled / rate - (time.perf_counter() - started)
                    if ahead > 0:
                        self._cancel.wait(ahead)

            self._update(
                state="stopped" if self._cancel.is_set() else "finished",
                finished_at=_now(),
                alerted_events_by_severity=summarize(sent, results)["alerted_events_by_severity"],
            )
            logger.info("Dataset replay ended after %d records", handled)
        except Exception:
            logger.exception("Dataset replay failed")
            self._update(state="failed", finished_at=_now(), error="The replay stopped unexpectedly. Check the API log.")


def _empty_status() -> dict[str, Any]:
    return {
        "state": "idle",
        "split": None,
        "total": 0,
        "processed": 0,
        "rejected": 0,
        "correct": 0,
        "new_alerts": 0,
        "rate_limit": None,
        "started_at": None,
        "finished_at": None,
        "error": None,
        "alerted_events_by_severity": {},
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
