"""SQLite traceability store for product inspection history and NG image archiving."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from steelvision.api.quality import ImageQualityReport
from steelvision.decision.fusion import DecisionResult

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    filename TEXT NOT NULL,
    image_width INTEGER NOT NULL,
    image_height INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    reason TEXT NOT NULL,
    max_det_confidence REAL NOT NULL,
    anomaly_score REAL,
    det_threshold REAL NOT NULL,
    anom_threshold REAL NOT NULL,
    threshold_version TEXT NOT NULL,
    defect_classes_json TEXT NOT NULL,
    detections_json TEXT NOT NULL,
    quality_json TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    saved_image_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_inspections_verdict ON inspections(verdict);
CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp_utc DESC);
"""


class TraceabilityStore:
    """SQLite-backed store for recording and querying product inspection results."""

    def __init__(self, db_path: str | Path = "runs/traceability/inspections.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def record_inspection(
        self,
        *,
        filename: str,
        image_width: int,
        image_height: int,
        decision: DecisionResult,
        quality: ImageQualityReport,
        latency_ms: float,
        threshold_version: str = "v1-val-2026-10-26",
        saved_image_path: str | None = None,
    ) -> dict:
        """Insert a single inspection record and return its hydrated dictionary."""
        ts = datetime.now(timezone.utc).isoformat()
        detections_payload = [
            {
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox": list(d.bbox),
            }
            for d in decision.active_detections
        ]
        defect_classes = decision.defect_classes

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO inspections (
                    timestamp_utc, filename, image_width, image_height,
                    verdict, reason, max_det_confidence, anomaly_score,
                    det_threshold, anom_threshold, threshold_version,
                    defect_classes_json, detections_json, quality_json,
                    latency_ms, saved_image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    filename,
                    int(image_width),
                    int(image_height),
                    decision.verdict,
                    decision.reason,
                    round(float(decision.max_det_confidence), 6),
                    round(float(decision.anomaly_score), 6)
                    if decision.anomaly_score is not None
                    else None,
                    float(decision.thresholds.det_threshold),
                    float(decision.thresholds.anom_threshold),
                    threshold_version,
                    json.dumps(defect_classes),
                    json.dumps(detections_payload),
                    json.dumps(quality.to_dict()),
                    round(float(latency_ms), 2),
                    saved_image_path,
                ),
            )
            row_id = int(cur.lastrowid)

        return self.get_inspection(row_id)  # type: ignore[return-value]

    @staticmethod
    def _hydrate_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["defect_classes"] = json.loads(d.pop("defect_classes_json"))
        d["active_detections"] = json.loads(d.pop("detections_json"))
        d["quality"] = json.loads(d.pop("quality_json"))
        return d

    def get_inspection(self, inspection_id: int) -> dict | None:
        """Retrieve a single inspection record by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM inspections WHERE id = ?", (int(inspection_id),)
            ).fetchone()
        return self._hydrate_row(row) if row is not None else None

    def list_inspections(
        self,
        *,
        limit: int = 100,
        verdict: str | None = None,
        reason: str | None = None,
        defect_class: str | None = None,
    ) -> list[dict]:
        """List recent inspections ordered newest-first with optional filters."""
        clauses: list[str] = []
        params: list[object] = []
        if verdict:
            clauses.append("verdict = ?")
            params.append(verdict.upper())
        if reason:
            clauses.append("reason = ?")
            params.append(reason.lower())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM inspections {where_sql} ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        records = [self._hydrate_row(r) for r in rows]
        if defect_class:
            records = [r for r in records if defect_class in r["defect_classes"]]
        return records

    def get_summary_stats(self) -> dict:
        """Compute aggregate QC statistics across all stored inspections."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM inspections ORDER BY id DESC").fetchall()

        total = len(rows)
        if total == 0:
            return {
                "total_inspected": 0,
                "n_ok": 0,
                "n_ng": 0,
                "ng_rate": 0.0,
                "reasons": {},
                "defect_class_counts": {},
                "mean_latency_ms": 0.0,
            }

        n_ng = sum(1 for r in rows if r["verdict"] == "NG")
        n_ok = total - n_ng
        reasons = Counter(r["reason"] for r in rows)
        class_counts: Counter[str] = Counter()
        for r in rows:
            for cls_name in json.loads(r["defect_classes_json"]):
                class_counts[cls_name] += 1
        mean_lat = sum(float(r["latency_ms"]) for r in rows) / total

        return {
            "total_inspected": total,
            "n_ok": n_ok,
            "n_ng": n_ng,
            "ng_rate": round(n_ng / total, 4),
            "reasons": dict(reasons),
            "defect_class_counts": dict(class_counts.most_common()),
            "mean_latency_ms": round(mean_lat, 2),
        }
