"""SQLite storage for detection events and alerts.

Every query is parameterized, and only fixed column names appear in SQL text.
Raw feature vectors are not stored for every event, only the few features an
alert's explanation cites.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    source TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    event_count INTEGER NOT NULL,
    max_confidence REAL,
    risk_score REAL NOT NULL,
    severity TEXT NOT NULL,
    src_ip TEXT,
    dst_ip TEXT,
    protocol TEXT,
    status TEXT NOT NULL,
    status_updated_at TEXT,
    explanation TEXT NOT NULL,
    detector TEXT NOT NULL DEFAULT 'model'
);
CREATE INDEX IF NOT EXISTS idx_alerts_group
    ON alerts (source, predicted_class, status, last_seen_at);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts (severity);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at TEXT NOT NULL,
    source TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    is_attack INTEGER NOT NULL,
    confidence REAL NOT NULL,
    probabilities TEXT NOT NULL,
    risk_score REAL NOT NULL,
    severity TEXT NOT NULL,
    src_ip TEXT,
    dst_ip TEXT,
    protocol TEXT,
    ground_truth TEXT,
    model_version TEXT NOT NULL,
    alert_id TEXT REFERENCES alerts (id)
);
CREATE INDEX IF NOT EXISTS idx_events_recent
    ON events (source, predicted_class, occurred_at);
"""

ACTIVE_STATUSES = ("open", "acknowledged")

# Rebuilds an alerts table from before rule alerts, keeping its rows. Rules have
# no confidence value, so max_confidence must allow NULL, and SQLite can't
# change that on an existing column.
_ALERT_COLUMNS = (
    "id, created_at, last_seen_at, source, predicted_class, event_count, max_confidence, "
    "risk_score, severity, src_ip, dst_ip, protocol, status, status_updated_at, explanation"
)


class Store:
    """Queries bound to one open connection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def count_recent_events(
        self,
        source: str,
        predicted_class: str,
        src_ip: str | None,
        since: str,
        at_most: int = 1_000_000,
    ) -> int:
        """Count matching events, stopping at `at_most` so busy periods stay fast."""
        row = self._connection.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM events WHERE source = ? AND predicted_class = ? "
            "AND src_ip IS ? AND occurred_at >= ? LIMIT ?)",
            (source, predicted_class, src_ip, since, at_most),
        ).fetchone()
        return row[0]

    def find_active_alert(self, source: str, predicted_class: str, src_ip: str | None, since: str) -> dict | None:
        """Find the alert a new detection should join rather than duplicate."""
        row = self._connection.execute(
            "SELECT * FROM alerts WHERE source = ? AND predicted_class = ? AND src_ip IS ? "
            "AND status IN (?, ?) AND last_seen_at >= ? ORDER BY last_seen_at DESC LIMIT 1",
            (source, predicted_class, src_ip, *ACTIVE_STATUSES, since),
        ).fetchone()
        return _alert(row) if row else None

    def create_alert(self, alert: dict[str, Any]) -> None:
        self._connection.execute(
            "INSERT INTO alerts (id, created_at, last_seen_at, source, predicted_class, event_count, "
            "max_confidence, risk_score, severity, src_ip, dst_ip, protocol, status, explanation) "
            "VALUES (:id, :created_at, :last_seen_at, :source, :predicted_class, :event_count, "
            ":max_confidence, :risk_score, :severity, :src_ip, :dst_ip, :protocol, :status, :explanation)",
            {**alert, "explanation": json.dumps(alert["explanation"])},
        )

    def extend_alert(self, alert_id: str, last_seen_at: str, confidence: float, risk_score: float, severity: str) -> None:
        """Add one event to an alert. Its risk only rises while it stays active."""
        self._connection.execute(
            "UPDATE alerts SET event_count = event_count + 1, last_seen_at = ?, "
            "max_confidence = MAX(max_confidence, ?), "
            "severity = CASE WHEN ? > risk_score THEN ? ELSE severity END, "
            "risk_score = MAX(risk_score, ?) WHERE id = ?",
            (last_seen_at, confidence, risk_score, severity, risk_score, alert_id),
        )

    def insert_event(self, event: dict[str, Any]) -> int:
        cursor = self._connection.execute(
            "INSERT INTO events (occurred_at, source, predicted_class, is_attack, confidence, probabilities, "
            "risk_score, severity, src_ip, dst_ip, protocol, ground_truth, model_version, alert_id) "
            "VALUES (:occurred_at, :source, :predicted_class, :is_attack, :confidence, :probabilities, "
            ":risk_score, :severity, :src_ip, :dst_ip, :protocol, :ground_truth, :model_version, :alert_id)",
            {**event, "probabilities": json.dumps(event["probabilities"])},
        )
        return cursor.lastrowid

    def create_rule_alert(self, alert: dict[str, Any]) -> bool:
        """Store a rule alert unless it already exists. Returns True if it was new."""
        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO alerts (id, created_at, last_seen_at, source, predicted_class, event_count, "
            "max_confidence, risk_score, severity, src_ip, dst_ip, protocol, status, explanation, detector) "
            "VALUES (:id, :created_at, :last_seen_at, :source, :predicted_class, :event_count, NULL, "
            ":risk_score, :severity, NULL, NULL, NULL, :status, :explanation, :detector)",
            {**alert, "explanation": json.dumps(alert["explanation"])},
        )
        return cursor.rowcount == 1

    def get_alert(self, alert_id: str) -> dict | None:
        row = self._connection.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return _alert(row) if row else None

    def list_alerts(
        self,
        status: str | None = None,
        severity: str | None = None,
        predicted_class: str | None = None,
        source: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        filters = {"status": status, "severity": severity, "predicted_class": predicted_class, "source": source}
        where, values = _where(filters)
        rows = self._connection.execute(
            f"SELECT * FROM alerts {where} ORDER BY risk_score DESC, last_seen_at DESC LIMIT ? OFFSET ?",
            (*values, limit, offset),
        ).fetchall()
        return [_alert(row) for row in rows]

    def update_alert_status(self, alert_id: str, status: str, updated_at: str) -> None:
        self._connection.execute(
            "UPDATE alerts SET status = ?, status_updated_at = ? WHERE id = ?",
            (status, updated_at, alert_id),
        )

    def list_events(self, source: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
        where, values = _where({"source": source})
        rows = self._connection.execute(
            f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            (*values, limit, offset),
        ).fetchall()
        return [{**dict(row), "probabilities": json.loads(row["probabilities"]), "is_attack": bool(row["is_attack"])} for row in rows]

    def timeline(self, source: str | None, bucket_length: int, limit: int = 5000) -> list[dict]:
        """Count events per class per time bucket, newest buckets first.

        `bucket_length` is how many leading timestamp characters form a bucket.
        Replayed events carry replay time, not capture time.
        """
        where, values = _where({"source": source})
        rows = self._connection.execute(
            f"SELECT substr(occurred_at, 1, ?) AS bucket, predicted_class, COUNT(*) AS count FROM events {where} "
            "GROUP BY bucket, predicted_class ORDER BY bucket DESC LIMIT ?",
            (bucket_length, *values, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def label_outcomes(self, source: str | None = None) -> dict[str, list[dict]]:
        """Compare predictions with known labels, which only replayed data has."""
        where, values = _where({"source": source})
        condition = f"{where} AND" if where else "WHERE"
        by_class = self._connection.execute(
            f"SELECT ground_truth, predicted_class, COUNT(*) AS count FROM events {condition} ground_truth IS NOT NULL "
            "GROUP BY ground_truth, predicted_class",
            values,
        ).fetchall()
        by_severity = self._connection.execute(
            "SELECT severity, SUM(ground_truth != 'Benign') AS real_attacks, SUM(ground_truth = 'Benign') AS actually_benign "
            f"FROM events {condition} ground_truth IS NOT NULL AND alert_id IS NOT NULL GROUP BY severity",
            values,
        ).fetchall()
        return {"by_class": [dict(row) for row in by_class], "by_severity": [dict(row) for row in by_severity]}

    def alert_outcomes(self, alert_id: str) -> dict[str, int]:
        """Count an alert's events by known label, when labels exist."""
        rows = self._connection.execute(
            "SELECT ground_truth, COUNT(*) FROM events WHERE alert_id = ? AND ground_truth IS NOT NULL GROUP BY ground_truth",
            (alert_id,),
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def clear_source(self, source: str) -> None:
        """Delete all events and alerts from one source, such as demo replay data."""
        self._connection.execute("DELETE FROM events WHERE source = ?", (source,))
        self._connection.execute("DELETE FROM alerts WHERE source = ?", (source,))

    def statistics(self, source: str | None = None) -> dict[str, Any]:
        where, values = _where({"source": source})

        def grouped(table: str, column: str) -> dict[str, int]:
            rows = self._connection.execute(
                f"SELECT {column}, COUNT(*) FROM {table} {where} GROUP BY {column}", values
            ).fetchall()
            return {row[0]: row[1] for row in rows}

        by_class = grouped("events", "predicted_class")
        malicious = self._connection.execute(
            f"SELECT COALESCE(SUM(is_attack), 0) FROM events {where}", values
        ).fetchone()[0]
        total = sum(by_class.values())
        return {
            "events_total": total,
            "events_benign": total - malicious,
            "events_malicious": malicious,
            "events_by_class": by_class,
            "alerts_by_severity": grouped("alerts", "severity"),
            "alerts_by_status": grouped("alerts", "status"),
            "alerts_by_class": grouped("alerts", "predicted_class"),
        }


class Database:
    """Opens a short-lived connection for each unit of work."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.session() as store:
            self._upgrade_alerts_table(store._connection)
            store._connection.executescript(SCHEMA)

    @staticmethod
    def _upgrade_alerts_table(connection: sqlite3.Connection) -> None:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(alerts)")]
        if not columns or "detector" in columns:
            return
        connection.execute("PRAGMA foreign_keys=OFF")
        # Otherwise the rename would repoint events.alert_id at the old table.
        connection.execute("PRAGMA legacy_alter_table=ON")
        connection.execute("ALTER TABLE alerts RENAME TO alerts_old")
        connection.execute("PRAGMA legacy_alter_table=OFF")
        connection.executescript(SCHEMA)
        connection.execute(f"INSERT INTO alerts ({_ALERT_COLUMNS}) SELECT {_ALERT_COLUMNS} FROM alerts_old")
        connection.execute("DROP TABLE alerts_old")
        connection.commit()
        connection.execute("PRAGMA foreign_keys=ON")

    @contextmanager
    def session(self) -> Iterator[Store]:
        """Yield a Store whose changes commit together, or not at all on error."""
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield Store(connection)
        finally:
            connection.close()


def _where(filters: dict[str, Any]) -> tuple[str, tuple]:
    """Build a WHERE clause. Keys are fixed column names from this module only."""
    active = {column: value for column, value in filters.items() if value is not None}
    if not active:
        return "", ()
    return "WHERE " + " AND ".join(f"{column} = ?" for column in active), tuple(active.values())


def _alert(row: sqlite3.Row) -> dict[str, Any]:
    return {**dict(row), "explanation": json.loads(row["explanation"])}
