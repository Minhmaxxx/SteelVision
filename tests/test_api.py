"""Unit tests for Image Quality Gate, SQLite TraceabilityStore, and FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from steelvision.api.app import InspectionService, create_app
from steelvision.api.quality import check_image_quality
from steelvision.api.store import TraceabilityStore
from steelvision.decision.fusion import Detection, decide


class _StubYoloDetector:
    """Lightweight deterministic detector stub for fast API unit tests."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._detections = detections or []

    def predict(self, image_bgr: np.ndarray, conf_threshold: float = 0.01) -> list[Detection]:
        return [d for d in self._detections if d.confidence >= conf_threshold]


class _StubAnomalyDetector:
    """Lightweight deterministic anomaly stub for fast API unit tests."""

    def __init__(self, score: float = 0.20) -> None:
        self._score = score

    def predict(self, image_rgb: np.ndarray) -> tuple[float, np.ndarray]:
        h, w = image_rgb.shape[:2]
        return self._score, np.full((h, w), self._score, dtype=np.float32)


def _make_textured_bgr(
    mean_val: int = 128,
    amplitude: int = 40,
    size: int = 128,
) -> np.ndarray:
    """Create a synthetic textured image with strong edges and balanced brightness."""
    y, x = np.indices((size, size))
    checker = ((x // 4 + y // 4) % 2) * 2 - 1
    img = np.clip(mean_val + checker * amplitude, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    assert ok
    return buf.tobytes()


def test_quality_gate_passes_normal_image() -> None:
    img = _make_textured_bgr(mean_val=128, amplitude=45)
    report = check_image_quality(img)
    assert report.passed is True
    assert report.warnings == []
    assert 80.0 <= report.mean_brightness <= 180.0
    assert report.contrast_std >= 15.0
    assert report.blur_laplacian_var >= 25.0


def test_quality_gate_flags_dark_bright_flat_and_blurry() -> None:
    dark = np.zeros((64, 64, 3), dtype=np.uint8)
    r_dark = check_image_quality(dark)
    assert not r_dark.passed
    assert any("too_dark" in w for w in r_dark.warnings)
    assert any("low_contrast" in w for w in r_dark.warnings)
    assert any("blurry" in w for w in r_dark.warnings)

    bright = np.full((64, 64, 3), 250, dtype=np.uint8)
    r_bright = check_image_quality(bright)
    assert not r_bright.passed
    assert any("too_bright" in w for w in r_bright.warnings)


def test_traceability_store_crud_and_stats(tmp_path: Path) -> None:
    store = TraceabilityStore(db_path=tmp_path / "inspections.db")
    img = _make_textured_bgr()
    quality = check_image_quality(img)

    ok_decision = decide(
        detections=[],
        anomaly_score=0.18,
        det_threshold=0.25,
        anom_threshold=0.53,
    )
    ok_rec = store.record_inspection(
        filename="part_001.png",
        image_width=128,
        image_height=128,
        decision=ok_decision,
        quality=quality,
        latency_ms=42.0,
    )

    ng_decision = decide(
        detections=[Detection(class_name="scratches", confidence=0.88, bbox=(10, 10, 50, 50))],
        anomaly_score=0.91,
        det_threshold=0.25,
        anom_threshold=0.53,
    )
    ng_rec = store.record_inspection(
        filename="part_002.png",
        image_width=128,
        image_height=128,
        decision=ng_decision,
        quality=quality,
        latency_ms=65.0,
        saved_image_path=(tmp_path / "ng_part_002.png").as_posix(),
    )

    assert ok_rec["verdict"] == "OK"
    assert ok_rec["saved_image_path"] is None
    assert ng_rec["verdict"] == "NG"
    assert ng_rec["reason"] == "both"
    assert ng_rec["defect_classes"] == ["scratches"]

    only_ng = store.list_inspections(verdict="NG")
    assert len(only_ng) == 1
    assert only_ng[0]["id"] == ng_rec["id"]

    by_class = store.list_inspections(defect_class="scratches")
    assert len(by_class) == 1

    stats = store.get_summary_stats()
    assert stats["total_inspected"] == 2
    assert stats["n_ng"] == 1
    assert stats["n_ok"] == 1
    assert stats["ng_rate"] == 0.5
    assert stats["defect_class_counts"]["scratches"] == 1


def test_fastapi_inspect_and_history_endpoints(tmp_path: Path) -> None:
    service = InspectionService(
        yolo_onnx_path=tmp_path / "nonexistent_yolo.onnx",
        patchcore_onnx_path=tmp_path / "nonexistent_patchcore.onnx",
        db_path=tmp_path / "inspections.db",
        ng_images_dir=tmp_path / "ng_images",
        det_threshold=0.25,
        anom_threshold=0.53,
    )
    service.yolo = _StubYoloDetector(  # type: ignore[assignment]
        detections=[
            Detection(class_name="scratches", confidence=0.82, bbox=(12.0, 15.0, 60.0, 70.0))
        ]
    )
    service.patchcore = _StubAnomalyDetector(score=0.75)  # type: ignore[assignment]

    app = create_app(service=service)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["yolo_loaded"] is True
    assert health.json()["patchcore_loaded"] is True

    png_bytes = _encode_png(_make_textured_bgr())
    resp = client.post(
        "/inspect?early_exit=false",
        files={"file": ("sample_ng.png", png_bytes, "image/png")},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["verdict"] == "NG"
    assert payload["reason"] == "both"
    assert len(payload["active_detections"]) == 1
    assert payload["saved_image_path"] is not None
    assert Path(payload["saved_image_path"]).exists()

    # Test early-exit skips anomaly branch when YOLO detects defect
    resp_ee = client.post(
        "/inspect?early_exit=true",
        files={"file": ("sample_ee.png", png_bytes, "image/png")},
    )
    assert resp_ee.status_code == 200
    payload_ee = resp_ee.json()
    assert payload_ee["verdict"] == "NG"
    assert payload_ee["reason"] == "detection"
    assert payload_ee["anomaly_score"] is None

    # Test invalid image payload returns HTTP 400
    bad_resp = client.post(
        "/inspect",
        files={"file": ("corrupt.png", b"not-an-image", "image/png")},
    )
    assert bad_resp.status_code == 400

    history = client.get("/inspections").json()
    assert len(history) == 2

    single = client.get(f"/inspections/{payload['id']}").json()
    assert single["id"] == payload["id"]

    not_found = client.get("/inspections/9999")
    assert not_found.status_code == 404

    stats = client.get("/stats").json()
    assert stats["total_inspected"] == 2
    assert stats["n_ng"] == 2
