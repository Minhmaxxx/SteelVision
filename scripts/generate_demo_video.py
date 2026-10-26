"""Generate a 60-second 720p MP4 showcase video and populate the SQLite traceability DB.

Usage:
    python scripts/generate_demo_video.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from steelvision.api.app import InspectionService

WIDTH, HEIGHT = 1280, 720
FPS = 30


def _draw_header(canvas: np.ndarray, title: str, subtitle: str) -> None:
    cv2.rectangle(canvas, (0, 0), (WIDTH, 88), (22, 28, 38), -1)
    cv2.putText(
        canvas,
        title,
        (36, 42),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        (245, 248, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        subtitle,
        (36, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (160, 195, 235),
        1,
        cv2.LINE_AA,
    )


def _draw_footer(canvas: np.ndarray, step_text: str) -> None:
    cv2.rectangle(canvas, (0, HEIGHT - 40), (WIDTH, HEIGHT), (18, 22, 30), -1)
    cv2.putText(
        canvas,
        "SteelVision | YOLOv8n + PatchCore ONNX | FastAPI + SQLite Traceability",
        (36, HEIGHT - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (150, 165, 185),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        step_text,
        (WIDTH - 280, HEIGHT - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (120, 210, 255),
        1,
        cv2.LINE_AA,
    )


def _write_seconds(writer: cv2.VideoWriter, frame: np.ndarray, seconds: float) -> None:
    n_frames = max(1, int(round(seconds * FPS)))
    for _ in range(n_frames):
        writer.write(frame)


def _make_intro_slide() -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), (14, 18, 24), dtype=np.uint8)
    _draw_header(
        canvas,
        "SteelVision: End-to-End Metal Surface QC Inspection",
        "Dual-Branch Supervised Detection (YOLOv8n) + Unsupervised Anomaly Detection (PatchCore)",
    )

    lines = [
        (
            "1. Pre-Inference Quality Gate",
            "Flags too_dark / too_bright / low_contrast / blurry frames upfront",
        ),
        (
            "2. Branch 1 - YOLOv8n (ONNX)",
            "Detects 6 known NEU-DET steel defects in 58.37 ms CPU (p50)",
        ),
        (
            "3. Branch 2 - PatchCore (ONNX)",
            "Catches unseen anomalies on MVTec metal_nut (99.80% Image AUROC)",
        ),
        (
            "4. OR-Gate Product Fusion",
            "NG if det_conf >= 0.25 OR anom_score >= 0.53 (Escape Rate <= 0.75%)",
        ),
        (
            "5. SQLite Traceability + API",
            "Logs every part verdict, reason, latency, and archives NG images",
        ),
    ]
    y = 155
    for heading, desc in lines:
        cv2.rectangle(canvas, (60, y - 32), (WIDTH - 60, y + 48), (28, 35, 48), -1)
        cv2.rectangle(canvas, (60, y - 32), (WIDTH - 60, y + 48), (65, 95, 135), 1)
        cv2.putText(
            canvas, heading, (85, y), cv2.FONT_HERSHEY_DUPLEX, 0.72, (110, 210, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            desc,
            (85, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 228, 238),
            1,
            cv2.LINE_AA,
        )
        y += 98

    _draw_footer(canvas, "Section 1/4: Architecture")
    return canvas


def _make_neu_slide(img_bgr: np.ndarray, record: dict, idx: int, total: int) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), (14, 18, 24), dtype=np.uint8)
    _draw_header(
        canvas,
        f"Live Inspection - NEU-DET Steel Sample ({idx}/{total})",
        f"File: {record['filename']} | SQLite ID #{record['id']}",
    )

    vis = img_bgr.copy()
    vis = cv2.resize(vis, (480, 480), interpolation=cv2.INTER_LINEAR)
    scale_x = 480.0 / max(1, record["image_width"])
    scale_y = 480.0 / max(1, record["image_height"])

    for det in record["active_detections"]:
        x1, y1, x2, y2 = det["bbox"]
        p1 = (int(x1 * scale_x), int(y1 * scale_y))
        p2 = (int(x2 * scale_x), int(y2 * scale_y))
        cv2.rectangle(vis, p1, p2, (40, 60, 255), 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(
            vis,
            label,
            (p1[0], max(22, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (80, 230, 255),
            2,
            cv2.LINE_AA,
        )

    canvas[130 : 130 + 480, 60 : 60 + 480] = vis
    cv2.rectangle(canvas, (60, 130), (540, 610), (90, 110, 140), 2)

    verdict = record["verdict"]
    v_color = (50, 65, 235) if verdict == "NG" else (60, 185, 85)
    cv2.rectangle(canvas, (600, 130), (1210, 220), v_color, -1)
    cv2.putText(
        canvas,
        f"VERDICT: {verdict}  (Reason: {record['reason'].upper()})",
        (625, 186),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    q_pass = record["quality"]["passed"]
    q_bright = record["quality"]["mean_brightness"]
    max_c = record["max_det_confidence"]
    tau_d = record["det_threshold"]
    details = [
        f"Max YOLO Confidence : {max_c:.3f}  (tau_det = {tau_d:.2f})",
        f"Detected Classes    : {', '.join(record['defect_classes']) or 'None'}",
        f"Active Boxes        : {len(record['active_detections'])}",
        f"Quality Gate Passed : {q_pass} (Brightness={q_bright:.1f})",
        f"E2E Latency (ONNX)  : {record['latency_ms']:.1f} ms",
        f"Traceability DB Row : #{record['id']} ({record['threshold_version']})",
    ]
    y = 280
    for line in details:
        cv2.putText(
            canvas, line, (610, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (225, 232, 242), 1, cv2.LINE_AA
        )
        y += 52

    _draw_footer(canvas, "Section 2/4: NEU-DET Detection")
    return canvas


def _make_mvtec_slide(
    img_bgr: np.ndarray,
    heatmap_norm: np.ndarray,
    record: dict,
    defect_type: str,
    idx: int,
    total: int,
) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), (14, 18, 24), dtype=np.uint8)
    _draw_header(
        canvas,
        f"PatchCore Anomaly Inspection - MVTec metal_nut ({idx}/{total})",
        f"Category: {defect_type} | File: {record['filename']}",
    )

    disp_bgr = cv2.resize(img_bgr, (340, 340), interpolation=cv2.INTER_LINEAR)
    hm_u8 = np.clip(heatmap_norm * 255.0, 0, 255).astype(np.uint8)
    hm_u8 = cv2.resize(hm_u8, (340, 340), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(disp_bgr, 0.55, hm_color, 0.45, 0)

    canvas[150 : 150 + 340, 50 : 50 + 340] = disp_bgr
    canvas[150 : 150 + 340, 420 : 420 + 340] = overlay
    cv2.putText(
        canvas,
        "Input RGB Frame",
        (50, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 215, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "PatchCore Anomaly Heatmap",
        (420, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 215, 235),
        1,
        cv2.LINE_AA,
    )

    verdict = record["verdict"]
    v_color = (50, 65, 235) if verdict == "NG" else (60, 185, 85)
    cv2.rectangle(canvas, (800, 150), (1230, 235), v_color, -1)
    cv2.putText(
        canvas,
        f"VERDICT: {verdict}",
        (825, 204),
        cv2.FONT_HERSHEY_DUPLEX,
        0.92,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    anom_val = record["anomaly_score"] if record["anomaly_score"] is not None else 0.0
    info = [
        f"Ground-Truth Folder : {defect_type}",
        f"Anomaly Score       : {anom_val:.4f}",
        f"Operating Threshold : {record['anom_threshold']:.2f}",
        f"Decision Trigger    : {record['reason'].upper()}",
        f"Quality Gate        : {'PASS' if record['quality']['passed'] else 'WARN'}",
        f"Inference Latency   : {record['latency_ms']:.1f} ms",
    ]
    y = 290
    for line in info:
        cv2.putText(
            canvas, line, (805, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (225, 232, 242), 1, cv2.LINE_AA
        )
        y += 46

    _draw_footer(canvas, "Section 3/4: PatchCore Anomaly")
    return canvas


def _make_summary_slide() -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), (14, 18, 24), dtype=np.uint8)
    _draw_header(
        canvas,
        "Verified QC Results & CPU Optimization Benchmark",
        "Held-out Test Split Metrics + AMD Ryzen 7 5800H CPU Latency (batch=1)",
    )

    rows = [
        (
            "PatchCore (MVTec metal_nut)",
            "Image AUROC: 99.80% | Pixel AUROC: 98.69% | Test Escape: 0.00%",
        ),
        (
            "YOLOv8n (NEU-DET + OK bg)",
            "Val tau*=0.25 -> Test Escape: 0.75% (vs 9.40% at naive 0.50) | F1: 0.9635",
        ),
        (
            "Unseen Defect Simulation",
            "Detector-only escape: 23.7%-26.9% -> Dual-Branch Fusion escape: 0.00%",
        ),
        (
            "CPU Latency (PyTorch E2E)",
            "p50 = 279.60 ms | p95 = 361.33 ms | 3.58 FPS | 1,297 MB RAM",
        ),
        (
            "CPU Latency (ONNX Early-Exit)",
            "p50 = 63.78 ms  | p95 = 100.22 ms | 15.68 FPS | 626 MB RAM (4.38x faster)",
        ),
    ]
    y = 155
    for title, val in rows:
        cv2.rectangle(canvas, (55, y - 30), (WIDTH - 55, y + 48), (26, 34, 46), -1)
        cv2.rectangle(canvas, (55, y - 30), (WIDTH - 55, y + 48), (70, 130, 180), 1)
        cv2.putText(
            canvas, title, (80, y), cv2.FONT_HERSHEY_DUPLEX, 0.68, (115, 220, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            val,
            (80, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 240, 248),
            1,
            cv2.LINE_AA,
        )
        y += 98

    _draw_footer(canvas, "Section 4/4: QC & Speedup Summary")
    return canvas


def main() -> None:
    out_path = Path("runs/demo_steelvision.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    service = InspectionService()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (WIDTH, HEIGHT))

    # 1. Intro slide (8s)
    _write_seconds(writer, _make_intro_slide(), 8.0)

    # 2. NEU-DET samples (5 samples x 4s = 20s)
    neu_dir = Path("data/processed/neu_det/images/test")
    neu_samples: list[Path] = []
    for prefix in ["crazing", "inclusion", "patches", "pitted_surface", "scratches"]:
        matches = sorted(neu_dir.glob(f"*{prefix}_*.jpg"))
        if matches:
            neu_samples.append(matches[0])

    for i, img_path in enumerate(neu_samples, start=1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        rec = service.inspect_bgr(img_bgr, filename=img_path.name, early_exit=True)
        slide = _make_neu_slide(img_bgr, rec, i, len(neu_samples))
        _write_seconds(writer, slide, 4.0)

    # 3. MVTec metal_nut samples (5 samples x 3.6s = 18s)
    mvtec_root = Path("data/raw/mvtec_ad/metal_nut/test")
    mvtec_types = ["good", "scratch", "bent", "color", "flip"]
    mvtec_samples: list[tuple[str, Path]] = []
    for dt in mvtec_types:
        candidates = sorted((mvtec_root / dt).glob("*.png"))
        if candidates:
            mvtec_samples.append((dt, candidates[0]))

    for i, (dt, img_path) in enumerate(mvtec_samples, start=1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        # For MVTec metal_nut demo, set det_threshold=0.99 so PatchCore drives the verdict
        rec = service.inspect_bgr(
            img_bgr,
            filename=f"metal_nut_{dt}_{img_path.name}",
            det_threshold=0.99,
            early_exit=False,
        )
        heatmap = np.zeros(img_bgr.shape[:2], dtype=np.float32)
        if service.patchcore is not None:
            _, heatmap = service.patchcore.predict(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        slide = _make_mvtec_slide(img_bgr, heatmap, rec, dt, i, len(mvtec_samples))
        _write_seconds(writer, slide, 3.6)

    # 4. Summary slide (14s -> total = 60s)
    _write_seconds(writer, _make_summary_slide(), 14.0)

    writer.release()
    stats = service.store.get_summary_stats()
    print(f"Generated demo video: {out_path} ({out_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"Traceability DB stats: {stats}")


if __name__ == "__main__":
    main()
