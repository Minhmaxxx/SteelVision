"""Streamlit QC Traceability Dashboard for SteelVision.

Run locally:
    streamlit run src/steelvision/dashboard/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from steelvision.api.app import InspectionService


def draw_detections_bgr(img_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Render bounding boxes and confidence labels onto a copy of ``img_bgr``."""
    canvas = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 60, 255), 2)
        cv2.putText(
            canvas,
            label,
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 60, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def draw_heatmap_overlay_rgb(img_rgb: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
    """Blend a normalized PatchCore anomaly map onto ``img_rgb``."""
    h, w = img_rgb.shape[:2]
    amap_resized = cv2.resize(
        anomaly_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
    )
    norm = np.clip((amap_resized - amap_resized.min()) / (np.ptp(amap_resized) + 1e-8), 0.0, 1.0)
    heat_bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 0.55, heat_rgb, 0.45, 0)


@st.cache_resource
def get_service() -> InspectionService:
    return InspectionService()


def main() -> None:
    st.set_page_config(page_title="SteelVision QC Dashboard", layout="wide")
    st.title("SteelVision — Metal Surface QC Inspection & Traceability")
    st.caption(
        "Fusing **YOLOv8n** (known defect localization) + **PatchCore** "
        "(one-class anomaly safety net) with SQLite traceability."
    )

    svc = get_service()
    stats = svc.store.get_summary_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Inspected", stats["total_inspected"])
    c2.metric("OK Products", stats["n_ok"])
    c3.metric("NG Products (Rejected)", stats["n_ng"], f"{stats['ng_rate'] * 100:.1f}% NG rate")
    c4.metric("Mean Latency (CPU)", f"{stats['mean_latency_ms']:.1f} ms")

    tab_live, tab_history = st.tabs(["Live Inspection", "Traceability History & Analytics"])

    with tab_live:
        st.subheader("Inspect a Product Image")
        col_cfg, col_up = st.columns([1, 2])
        with col_cfg:
            det_tau = st.slider("Detector Threshold (τ_det)", 0.05, 0.95, svc.det_threshold, 0.05)
            anom_tau = st.slider("Anomaly Threshold (τ_anom)", 0.20, 0.90, svc.anom_threshold, 0.01)
            early_exit = st.checkbox(
                "Early-Exit (skip PatchCore when YOLO already detects known defect)",
                value=False,
            )
        with col_up:
            uploaded = st.file_uploader(
                "Upload metal product image (PNG/JPG/BMP)", type=["png", "jpg", "jpeg", "bmp"]
            )

        if uploaded is not None:
            raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if bgr is not None:
                rec = svc.inspect_bgr(
                    bgr,
                    filename=uploaded.name,
                    det_threshold=det_tau,
                    anom_threshold=anom_tau,
                    early_exit=early_exit,
                )
                verdict_color = "red" if rec["verdict"] == "NG" else "green"
                st.markdown(
                    f"### Verdict: :{verdict_color}[**{rec['verdict']}**] — "
                    f"Reason: `{rec['reason']}` | Latency: `{rec['latency_ms']:.1f} ms` | "
                    f"Max Box Conf: `{rec['max_det_confidence']:.3f}` | "
                    f"Anomaly Score: `{rec['anomaly_score']}`"
                )
                if not rec["quality"]["passed"]:
                    st.warning(f"Image Quality Gate Warnings: {rec['quality']['warnings']}")

                img_cols = st.columns(3)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img_cols[0].image(rgb, caption="Original Input", use_container_width=True)

                boxed_bgr = draw_detections_bgr(bgr, rec["active_detections"])
                n_boxes = len(rec["active_detections"])
                img_cols[1].image(
                    cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB),
                    caption=f"YOLOv8 Detections ({n_boxes} boxes >= {det_tau:.2f})",
                    use_container_width=True,
                )

                if svc.patchcore is not None:
                    _, amap = svc.patchcore.predict(rgb)
                    if amap is not None:
                        overlay = draw_heatmap_overlay_rgb(rgb, amap)
                        img_cols[2].image(
                            overlay,
                            caption="PatchCore Anomaly Heatmap",
                            use_container_width=True,
                        )
                st.json(rec, expanded=False)

    with tab_history:
        st.subheader("SQLite Inspection Log")
        f1, f2, f3 = st.columns(3)
        verdict_filter = f1.selectbox("Filter Verdict", ["ALL", "NG", "OK"])
        reason_filter = f2.selectbox("Filter Reason", ["ALL", "detection", "anomaly", "both", "ok"])
        limit_val = f3.number_input("Max Records", min_value=10, max_value=500, value=100, step=10)

        records = svc.store.list_inspections(
            limit=int(limit_val),
            verdict=None if verdict_filter == "ALL" else verdict_filter,
            reason=None if reason_filter == "ALL" else reason_filter,
        )

        if records:
            df = pd.DataFrame(
                [
                    {
                        "ID": r["id"],
                        "Timestamp (UTC)": r["timestamp_utc"],
                        "Filename": r["filename"],
                        "Verdict": r["verdict"],
                        "Reason": r["reason"],
                        "Max Det Conf": r["max_det_confidence"],
                        "Anomaly Score": r["anomaly_score"],
                        "Defects": ", ".join(r["defect_classes"]),
                        "Quality Passed": r["quality"]["passed"],
                        "Latency (ms)": r["latency_ms"],
                    }
                    for r in records
                ]
            )
            st.dataframe(df, use_container_width=True)

            if stats["defect_class_counts"]:
                st.subheader("Defect Class Breakdown")
                st.bar_chart(pd.Series(stats["defect_class_counts"], name="Count"))

            ng_records = [r for r in records if r["verdict"] == "NG" and r.get("saved_image_path")]
            if ng_records:
                st.subheader("Archived NG Images Gallery")
                selected_id = st.selectbox(
                    "Inspect Archived NG Record",
                    options=[r["id"] for r in ng_records],
                    format_func=lambda rid: (
                        f"ID #{rid} — {next(r['filename'] for r in ng_records if r['id'] == rid)}"
                    ),
                )
                sel = next(r for r in ng_records if r["id"] == selected_id)
                img_path = Path(sel["saved_image_path"])
                if img_path.exists():
                    ng_bgr = cv2.imread(str(img_path))
                    ng_boxed = draw_detections_bgr(ng_bgr, sel["active_detections"])
                    cap = (
                        f"Archived NG #{sel['id']} ({sel['filename']}) — "
                        f"Defects: {sel['defect_classes']}"
                    )
                    st.image(
                        cv2.cvtColor(ng_boxed, cv2.COLOR_BGR2RGB),
                        caption=cap,
                        width=400,
                    )
        else:
            st.info("No inspection records match the current filters.")


if __name__ == "__main__":
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    main()
