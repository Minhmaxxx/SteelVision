"""FastAPI service for SteelVision metal surface defect inspection.

Start locally:
    uvicorn steelvision.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from steelvision.api.quality import check_image_quality
from steelvision.api.store import TraceabilityStore
from steelvision.decision.fusion import Detection, decide
from steelvision.models.anomaly import OnnxAnomalyDetector
from steelvision.models.detector import OnnxYoloDetector

DEFAULT_YOLO_ONNX = "runs/yolo/neu_det_best.onnx"
DEFAULT_PATCHCORE_ONNX = "runs/patchcore/metal_nut/weights/onnx/patchcore.onnx"
DEFAULT_DB_PATH = "runs/traceability/inspections.db"
DEFAULT_NG_IMAGES_DIR = "runs/traceability/ng_images"
DEFAULT_DET_THRESHOLD = 0.25
DEFAULT_ANOM_THRESHOLD = 0.53
THRESHOLD_VERSION = "v1-val-2026-10-26"


class InspectionService:
    """Stateful inspection engine managing ONNX sessions, SQLite store, and NG image archiving."""

    def __init__(
        self,
        *,
        yolo_onnx_path: str | Path | None = None,
        patchcore_onnx_path: str | Path | None = None,
        db_path: str | Path | None = None,
        ng_images_dir: str | Path | None = None,
        det_threshold: float = DEFAULT_DET_THRESHOLD,
        anom_threshold: float = DEFAULT_ANOM_THRESHOLD,
        threshold_version: str = THRESHOLD_VERSION,
    ) -> None:
        self.yolo_onnx_path = Path(
            yolo_onnx_path or os.environ.get("STEELVISION_YOLO_ONNX", DEFAULT_YOLO_ONNX)
        )
        self.patchcore_onnx_path = Path(
            patchcore_onnx_path
            or os.environ.get("STEELVISION_PATCHCORE_ONNX", DEFAULT_PATCHCORE_ONNX)
        )
        self.db_path = Path(db_path or os.environ.get("STEELVISION_DB_PATH", DEFAULT_DB_PATH))
        self.ng_images_dir = Path(
            ng_images_dir or os.environ.get("STEELVISION_NG_DIR", DEFAULT_NG_IMAGES_DIR)
        )
        self.ng_images_dir.mkdir(parents=True, exist_ok=True)

        self.det_threshold = float(det_threshold)
        self.anom_threshold = float(anom_threshold)
        self.threshold_version = threshold_version

        self.store = TraceabilityStore(self.db_path)
        self.yolo: OnnxYoloDetector | None = (
            OnnxYoloDetector(self.yolo_onnx_path) if self.yolo_onnx_path.exists() else None
        )
        self.patchcore: OnnxAnomalyDetector | None = (
            OnnxAnomalyDetector(self.patchcore_onnx_path)
            if self.patchcore_onnx_path.exists()
            else None
        )

    def inspect_bgr(
        self,
        img_bgr: np.ndarray,
        filename: str = "upload.png",
        *,
        det_threshold: float | None = None,
        anom_threshold: float | None = None,
        early_exit: bool = False,
    ) -> dict:
        """Run full inspection pipeline on a BGR uint8 image and log to SQLite."""
        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]
        tau_det = float(det_threshold if det_threshold is not None else self.det_threshold)
        tau_anom = float(anom_threshold if anom_threshold is not None else self.anom_threshold)

        quality = check_image_quality(img_bgr)

        detections: list[Detection] = []
        if self.yolo is not None:
            detections = self.yolo.predict(img_bgr, conf_threshold=min(tau_det, 0.10))

        det_triggered = any(d.confidence >= tau_det for d in detections)
        anomaly_score: float | None = None
        if self.patchcore is not None and not (early_exit and det_triggered):
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            score, _ = self.patchcore.predict(rgb)
            anomaly_score = round(float(score), 6)

        decision = decide(
            detections=detections,
            anomaly_score=anomaly_score,
            det_threshold=tau_det,
            anom_threshold=tau_anom,
        )

        saved_path: str | None = None
        if decision.is_ng:
            safe_stem = Path(filename).stem[:40] or "part"
            out_name = f"ng_{int(time.time())}_{uuid.uuid4().hex[:6]}_{safe_stem}.png"
            dest = self.ng_images_dir / out_name
            cv2.imwrite(str(dest), img_bgr)
            saved_path = dest.as_posix()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return self.store.record_inspection(
            filename=filename,
            image_width=w,
            image_height=h,
            decision=decision,
            quality=quality,
            latency_ms=latency_ms,
            threshold_version=self.threshold_version,
            saved_image_path=saved_path,
        )


def create_app(service: InspectionService | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SteelVision QC Inspection API",
        version="0.1.0",
        description=(
            "End-to-end metal surface defect inspection combining YOLOv8 object detection "
            "and PatchCore anomaly detection with SQLite traceability."
        ),
    )
    svc = service if service is not None else InspectionService()
    app.state.service = svc

    @app.get("/health")
    def health() -> dict:
        s: InspectionService = app.state.service
        return {
            "status": "ok",
            "yolo_loaded": s.yolo is not None,
            "patchcore_loaded": s.patchcore is not None,
            "det_threshold": s.det_threshold,
            "anom_threshold": s.anom_threshold,
            "threshold_version": s.threshold_version,
            "db_path": s.db_path.as_posix(),
        }

    @app.post("/inspect")
    async def inspect_endpoint(
        file: UploadFile = File(...),  # noqa: B008
        det_threshold: float | None = Query(default=None, ge=0.0, le=1.0),
        anom_threshold: float | None = Query(default=None),
        early_exit: bool = Query(default=False),
    ) -> dict:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image bytes.")

        s: InspectionService = app.state.service
        return s.inspect_bgr(
            img_bgr,
            filename=file.filename or "upload.png",
            det_threshold=det_threshold,
            anom_threshold=anom_threshold,
            early_exit=early_exit,
        )

    @app.get("/inspections")
    def list_inspections_endpoint(
        limit: int = Query(default=50, ge=1, le=500),
        verdict: str | None = Query(default=None),
        reason: str | None = Query(default=None),
        defect_class: str | None = Query(default=None),
    ) -> list[dict]:
        s: InspectionService = app.state.service
        return s.store.list_inspections(
            limit=limit,
            verdict=verdict,
            reason=reason,
            defect_class=defect_class,
        )

    @app.get("/inspections/{inspection_id}")
    def get_inspection_endpoint(inspection_id: int) -> dict:
        s: InspectionService = app.state.service
        rec = s.store.get_inspection(inspection_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Inspection {inspection_id} not found.")
        return rec

    @app.get("/stats")
    def stats_endpoint() -> dict:
        s: InspectionService = app.state.service
        return s.store.get_summary_stats()

    return app


app = create_app()
