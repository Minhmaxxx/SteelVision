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

SAMPLES_DIR = Path("assets/samples")

SAMPLE_CATALOG: list[dict[str, object]] = [
    {
        "file": "metal_nut_good.png",
        "label": "🟢 [MVTec] Đai ốc đạt chuẩn (OK — Không lỗi)",
        "domain": "mvtec",
        "desc": "Mẫu đai ốc kim loại hoàn toàn bình thường (Ground Truth: OK).",
    },
    {
        "file": "metal_nut_scratch.png",
        "label": "🔴 [MVTec] Đai ốc bị trầy xước bề mặt (Scratch — NG)",
        "domain": "mvtec",
        "desc": "Đai ốc có vết xước sáng trên bề mặt kim loại (Ground Truth: NG).",
    },
    {
        "file": "metal_nut_bent.png",
        "label": "🔴 [MVTec] Đai ốc bị biến dạng / cong cạnh (Bent — NG)",
        "domain": "mvtec",
        "desc": "Đai ốc bị móp/biến dạng ở góc cạnh (Ground Truth: NG).",
    },
    {
        "file": "metal_nut_color.png",
        "label": "🔴 [MVTec] Đai ốc bị vết bẩn màu (Color — Lỗi lạ — NG)",
        "domain": "mvtec",
        "desc": "Lỗi đổi màu cục bộ chưa từng có trong nhãn YOLO, được PatchCore bắt gọn.",
    },
    {
        "file": "metal_nut_flip.png",
        "label": "🔴 [MVTec] Đai ốc bị dập ngược mặt (Flip — NG)",
        "domain": "mvtec",
        "desc": "Linh kiện bị lật ngược so với chiều chuẩn (Ground Truth: NG).",
    },
    {
        "file": "neu_scratches.jpg",
        "label": "🟠 [NEU-DET] Thép cán nóng: Vết xước dài (Scratches — NG)",
        "domain": "neu",
        "desc": "Bề mặt thép có các vệt xước dọc/ngang rõ nét.",
    },
    {
        "file": "neu_pitted_surface.jpg",
        "label": "🟠 [NEU-DET] Thép cán nóng: Rỗ bề mặt (Pitted Surface — NG)",
        "domain": "neu",
        "desc": "Bề mặt thép có các cụm lỗ rỗ li ti mật độ cao.",
    },
    {
        "file": "neu_patches.jpg",
        "label": "🟠 [NEU-DET] Thép cán nóng: Đốm mảng tối (Patches — NG)",
        "domain": "neu",
        "desc": "Vùng đốm loang diện tích lớn trên bề mặt tấm thép.",
    },
    {
        "file": "neu_inclusion.jpg",
        "label": "🟠 [NEU-DET] Thép cán nóng: Lẫn tạp chất (Inclusion — NG)",
        "domain": "neu",
        "desc": "Các vệt tạp chất phi kim loại kéo dài theo chiều cán.",
    },
    {
        "file": "neu_crazing.jpg",
        "label": "🟠 [NEU-DET] Thép cán nóng: Nứt chân chim (Crazing — NG)",
        "domain": "neu",
        "desc": "Mạng lưới vết nứt mảnh độ tương phản thấp.",
    },
    {
        "file": "neu_rolled-in_scale.jpg",
        "label": "🟠 [NEU-DET] Thép cán nóng: Vảy oxit cán ép (Rolled-in Scale — NG)",
        "domain": "neu",
        "desc": "Vảy oxit sắt bị cán lún vào bề mặt thép.",
    },
]

REASON_LABELS_VI = {
    "ok": "Đạt chuẩn (Cả 2 nhánh đều dưới ngưỡng lỗi)",
    "detection": "Nhánh YOLOv8n phát hiện lỗi bề mặt đã biết",
    "anomaly": "Nhánh PatchCore phát hiện điểm bất thường vượt ngưỡng",
    "both": "Cả 2 nhánh YOLOv8n và PatchCore cùng phát hiện lỗi",
}


def inject_custom_css() -> None:
    """Inject modern dark/light-compatible CSS for industrial QC look and feel."""
    st.markdown(
        """
        <style>
        .hero-box {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #1e3a5f 100%);
            padding: 1.4rem 1.8rem;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            color: #f8fafc;
            margin-bottom: 1.2rem;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18);
        }
        .hero-title {
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0 0 0.35rem 0;
            color: #ffffff;
        }
        .hero-sub {
            font-size: 0.95rem;
            color: #cbd5e1;
            margin: 0;
        }
        .badge-row {
            margin-top: 0.75rem;
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
        }
        .status-badge {
            background: rgba(56, 189, 248, 0.14);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: #7dd3fc;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .verdict-card-ng {
            background: linear-gradient(135deg, rgba(220, 38, 38, 0.16), rgba(153, 27, 27, 0.08));
            border-left: 6px solid #ef4444;
            border-radius: 10px;
            padding: 1rem 1.3rem;
            margin: 0.8rem 0 1.1rem 0;
        }
        .verdict-card-ok {
            background: linear-gradient(135deg, rgba(22, 163, 74, 0.16), rgba(21, 128, 61, 0.08));
            border-left: 6px solid #22c55e;
            border-radius: 10px;
            padding: 1rem 1.3rem;
            margin: 0.8rem 0 1.1rem 0;
        }
        .verdict-title {
            font-size: 1.35rem;
            font-weight: 800;
            margin: 0 0 0.25rem 0;
        }
        .guide-card {
            background: rgba(30, 41, 59, 0.45);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 10px;
            padding: 0.85rem 1.1rem;
            margin-bottom: 0.75rem;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def draw_detections_bgr(img_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Render bounding boxes and confidence labels onto a copy of ``img_bgr``."""
    canvas = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        label = f"{det['class_name']} ({det['confidence']:.2f})"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 50, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        ty = max(th + 6, y1)
        cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 6, ty), (30, 50, 255), -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, ty - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
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


def render_sidebar_controls(svc: InspectionService) -> tuple[float, float, bool, str]:
    """Render sidebar controls and detailed Vietnamese explanations for sliders."""
    with st.sidebar:
        st.header("⚙️ Cấu hình Ngưỡng QC")
        preset = st.radio(
            "Chế độ vận hành (Preset):",
            options=[
                "🔄 Tự động theo mẫu ảnh (Khuyên dùng)",
                "🔩 Chuyên biệt Thép cán (NEU-DET)",
                "⚙️ Chuyên biệt Đai ốc (MVTec PatchCore)",
                "🎛️ Tự chỉnh thủ công (Custom)",
            ],
            index=0,
            help=(
                "Tự động đặt ngưỡng phù hợp với miền dữ liệu (Thép phẳng NEU-DET hoặc "
                "Đai ốc MVTec) hoặc cho phép bạn tự kéo thanh trượt."
            ),
        )

        default_det = svc.det_threshold
        default_anom = svc.anom_threshold
        default_ee = False

        if preset == "🔩 Chuyên biệt Thép cán (NEU-DET)":
            default_det, default_anom, default_ee = 0.25, 0.90, True
        elif preset == "⚙️ Chuyên biệt Đai ốc (MVTec PatchCore)":
            default_det, default_anom, default_ee = 0.99, 0.53, False

        det_tau = st.slider(
            "1️⃣ Ngưỡng phát hiện lỗi YOLOv8 (τ_det)",
            min_value=0.05,
            max_value=0.99,
            value=float(default_det),
            step=0.01,
            help=(
                "Độ tự tin tối thiểu để YOLOv8 khoanh vùng lỗi (xước, rỗ, nứt...) và báo NG. "
                "Kéo thấp = bắt lỗi gắt hơn; Kéo cao = giảm báo nhầm."
            ),
        )

        anom_tau = st.slider(
            "2️⃣ Ngưỡng điểm bất thường PatchCore (τ_anom)",
            min_value=0.20,
            max_value=0.95,
            value=float(default_anom),
            step=0.01,
            help=(
                "Điểm khác biệt tối đa so với mẫu chuẩn không lỗi. Nếu điểm bất thường >= τ_anom "
                "thì sản phẩm bị đánh NG (dùng để chặn lỗi mới chưa từng học)."
            ),
        )

        early_exit = st.checkbox(
            "⚡ Bật Early-Exit (Ngắt sớm tăng tốc CPU 4.38×)",
            value=default_ee,
            help=(
                "Nếu YOLOv8 đã tìm thấy lỗi >= τ_det thì kết luận NG ngay lập tức, "
                "bỏ qua bước chạy PatchCore để giảm thời gian xử lý xuống ~60 ms."
            ),
        )

        st.markdown("---")
        with st.expander("📖 Giải thích các thanh trượt (Ấn để xem)", expanded=True):
            st.markdown(
                "- **Ngưỡng YOLOv8 (`τ_det = 0.25`):** Tìm 6 loại lỗi thép đã biết. "
                "Nếu có khung bao đạt độ tự tin $\\ge \\tau_{\\text{det}}$ $\\rightarrow$ **NG**.\n"
                "- **Ngưỡng PatchCore (`τ_anom = 0.53`):** Đo độ lệch so với ảnh chuẩn `OK`. "
                "Nếu điểm lệch $\\ge \\tau_{\\text{anom}}$ $\\rightarrow$ **NG**.\n"
                "- **Luật hợp nhất (OR-Gate):** Chỉ cần **1 trong 2** nhánh vượt ngưỡng "
                "là sản phẩm bị loại (**NG**) để chống lọt lỗi tối đa."
            )
    return det_tau, anom_tau, early_exit, preset


def main() -> None:
    st.set_page_config(
        page_title="SteelVision — Trạm Kiểm Định Chất Lượng Bề Mặt Kim Loại",
        page_icon="🛡️",
        layout="wide",
    )
    inject_custom_css()

    svc = get_service()
    stats = svc.store.get_summary_stats()

    yolo_badge = "🟢 YOLOv8n ONNX: Sẵn sàng" if svc.yolo else "⚪ YOLOv8n: Chưa nạp"
    pc_badge = "🟢 PatchCore ONNX: Sẵn sàng" if svc.patchcore else "⚪ PatchCore: Chưa nạp"

    st.markdown(
        f"""
        <div class="hero-box">
            <p class="hero-title">🛡️ SteelVision — Hệ Thống Kiểm Tra Lỗi Bề Mặt Kim Loại (QC)</p>
            <p class="hero-sub">
                Kết hợp <b>YOLOv8n</b> (định vị lỗi quen thuộc) + <b>PatchCore</b>
                (lưới an toàn bắt lỗi lạ) + <b>Image Quality Gate</b> & lưu vết <b>SQLite</b>.
            </p>
            <div class="badge-row">
                <span class="status-badge">{yolo_badge}</span>
                <span class="status-badge">{pc_badge}</span>
                <span class="status-badge">🎯 Chuẩn QC: Escape Rate &le; 2.0%</span>
                <span class="status-badge">🗄️ Phiên bản ngưỡng: {svc.threshold_version}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    det_tau, anom_tau, early_exit, preset = render_sidebar_controls(svc)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Tổng số đã kiểm tra", f"{stats['total_inspected']} mẫu")
    c2.metric("✅ Sản phẩm Đạt (OK)", f"{stats['n_ok']} mẫu")
    c3.metric(
        "❌ Sản phẩm Lỗi (NG)",
        f"{stats['n_ng']} mẫu",
        f"Tỷ lệ NG: {stats['ng_rate'] * 100:.1f}%",
        delta_color="inverse",
    )
    c4.metric("⚡ Độ trễ TB (CPU)", f"{stats['mean_latency_ms']:.1f} ms")

    tab_live, tab_history, tab_guide = st.tabs(
        [
            "🔍 Kiểm Tra Sản Phẩm (Có Sẵn 11 Ảnh Mẫu)",
            "📊 Lịch Sử Truy Vết SQLite & Kho Ảnh NG",
            "💡 Hướng Dẫn & Ý Nghĩa Chỉ Số QC",
        ]
    )

    with tab_live:
        col_src, col_preview = st.columns([1.15, 1.85])
        selected_bgr: np.ndarray | None = None
        selected_name: str = "upload.png"
        sample_domain: str = "custom"
        sample_desc: str = ""

        with col_src:
            st.subheader("1. Chọn nguồn ảnh kiểm tra")
            input_mode = st.radio(
                "Nguồn ảnh đầu vào:",
                options=[
                    "📂 Dùng bộ 11 ảnh test mẫu tích hợp sẵn (Không cần tải ảnh)",
                    "⬆️ Tải ảnh từ máy tính của bạn (PNG / JPG / BMP)",
                ],
                index=0,
            )

            if input_mode.startswith("📂"):
                available_samples = [
                    s for s in SAMPLE_CATALOG if (SAMPLES_DIR / str(s["file"])).exists()
                ]
                if available_samples:
                    chosen = st.selectbox(
                        "Chọn mẫu sản phẩm để test thử ngay:",
                        options=available_samples,
                        format_func=lambda item: str(item["label"]),
                    )
                    sample_path = SAMPLES_DIR / str(chosen["file"])
                    selected_bgr = cv2.imread(str(sample_path))
                    selected_name = str(chosen["file"])
                    sample_domain = str(chosen["domain"])
                    sample_desc = str(chosen["desc"])
                    st.info(f"**Mô tả mẫu:** {sample_desc}")
                else:
                    st.warning(
                        "Không tìm thấy thư mục `assets/samples/`. Hãy tải ảnh lên ở mục dưới."
                    )
            else:
                uploaded = st.file_uploader(
                    "Chọn file ảnh bề mặt kim loại (PNG, JPG, BMP):",
                    type=["png", "jpg", "jpeg", "bmp"],
                )
                if uploaded is not None:
                    raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
                    selected_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    selected_name = uploaded.name

            # Apply automatic domain preset if selected
            eff_det_tau, eff_anom_tau, eff_ee = det_tau, anom_tau, early_exit
            if preset.startswith("🔄") and sample_domain == "mvtec":
                eff_det_tau, eff_anom_tau, eff_ee = 0.99, 0.53, False
                st.caption(
                    "✨ *Chế độ Tự động:* Ảnh thuộc nhóm **Đai ốc MVTec** → ưu tiên nhánh "
                    "**PatchCore** (`τ_anom = 0.53`) để phân biệt đai ốc `OK` và `NG`."
                )
            elif preset.startswith("🔄") and sample_domain == "neu":
                eff_det_tau, eff_anom_tau, eff_ee = 0.25, 0.90, True
                st.caption(
                    "✨ *Chế độ Tự động:* Ảnh thuộc nhóm **Thép cán NEU-DET** → ưu tiên nhánh "
                    "**YOLOv8n** (`τ_det = 0.25`) + bật **Early-Exit** tăng tốc CPU."
                )

            run_btn = st.button(
                "🚀 Chạy Kiểm Tra & Lưu Vào Sổ Truy Vết SQLite",
                type="primary",
                use_container_width=True,
                disabled=(selected_bgr is None),
            )

        with col_preview:
            if selected_bgr is not None:
                # Automatically run inspection on preview or when button is clicked
                if (
                    run_btn
                    or "last_inspected_name" not in st.session_state
                    or (
                        st.session_state.get("last_inspected_name") != selected_name
                        or st.session_state.get("last_taus") != (eff_det_tau, eff_anom_tau, eff_ee)
                    )
                ):
                    rec = svc.inspect_bgr(
                        selected_bgr,
                        filename=selected_name,
                        det_threshold=eff_det_tau,
                        anom_threshold=eff_anom_tau,
                        early_exit=eff_ee,
                    )
                    st.session_state["last_rec"] = rec
                    st.session_state["last_inspected_name"] = selected_name
                    st.session_state["last_taus"] = (eff_det_tau, eff_anom_tau, eff_ee)
                else:
                    rec = st.session_state["last_rec"]

                is_ng = rec["verdict"] == "NG"
                card_cls = "verdict-card-ng" if is_ng else "verdict-card-ok"
                verdict_icon = (
                    "❌ KẾT LUẬN: NG (LOẠI SẢN PHẨM)" if is_ng else "✅ KẾT LUẬN: OK (ĐẠT CHUẨN)"
                )
                reason_vi = REASON_LABELS_VI.get(rec["reason"], rec["reason"])

                lat_str = f"{rec['latency_ms']:.1f} ms"
                st.markdown(
                    f"""
                    <div class="{card_cls}">
                        <p class="verdict-title">{verdict_icon}</p>
                        <div><b>Lý do:</b> {reason_vi} &nbsp;|&nbsp;
                        <b>Thời gian xử lý:</b> <code>{lat_str}</code> &nbsp;|&nbsp;
                        <b>Mã lưu vết SQLite:</b> <code>#{rec["id"]}</code></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Score bars comparing actual model output vs slider thresholds
                sc1, sc2, sc3 = st.columns(3)
                max_c = float(rec["max_det_confidence"])
                anom_s = rec["anomaly_score"]
                sc1.metric(
                    "Điểm YOLOv8 cao nhất",
                    f"{max_c:.3f}",
                    f"Ngưỡng τ_det = {eff_det_tau:.2f}",
                    delta_color="off",
                )
                sc2.metric(
                    "Điểm bất thường PatchCore",
                    f"{anom_s:.3f}" if anom_s is not None else "Đã bỏ qua (Early-Exit)",
                    f"Ngưỡng τ_anom = {eff_anom_tau:.2f}",
                    delta_color="off",
                )
                q_ok = rec["quality"]["passed"]
                sc3.metric(
                    "Cổng chất lượng ảnh (Quality Gate)",
                    "ĐẠT (Tốt)" if q_ok else "CẢNH BÁO",
                    f"Độ sáng TB: {rec['quality']['mean_brightness']:.1f}",
                    delta_color="off",
                )

                if not q_ok:
                    st.warning(f"⚠️ Cảnh báo chất lượng ảnh camera: {rec['quality']['warnings']}")

                img_cols = st.columns(3)
                rgb = cv2.cvtColor(selected_bgr, cv2.COLOR_BGR2RGB)
                img_cols[0].image(
                    rgb, caption=f"1. Ảnh gốc ({selected_name})", use_container_width=True
                )

                boxed_bgr = draw_detections_bgr(selected_bgr, rec["active_detections"])
                n_boxes = len(rec["active_detections"])
                img_cols[1].image(
                    cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2RGB),
                    caption=f"2. Nhánh YOLOv8n ({n_boxes} vùng lỗi >= {eff_det_tau:.2f})",
                    use_container_width=True,
                )

                if svc.patchcore is not None and not (eff_ee and n_boxes > 0):
                    _, amap = svc.patchcore.predict(rgb)
                    if amap is not None:
                        overlay = draw_heatmap_overlay_rgb(rgb, amap)
                        img_cols[2].image(
                            overlay,
                            caption="3. Bản đồ nhiệt PatchCore (Vùng đỏ = Bất thường)",
                            use_container_width=True,
                        )
                else:
                    img_cols[2].info(
                        "⚡ Nhánh PatchCore được bỏ qua nhờ **Early-Exit** "
                        "vì YOLOv8n đã bắt được lỗi rõ ràng."
                    )

                with st.expander("🔎 Xem chi tiết bản ghi JSON lưu trong SQLite"):
                    st.json(rec)

    with tab_history:
        st.subheader("🗄️ Sổ Truy Vết Kiểm Định (SQLite Traceability Log)")
        f1, f2, f3 = st.columns(3)
        verdict_filter = f1.selectbox("Lọc theo Phán quyết", ["ALL", "NG", "OK"])
        reason_filter = f2.selectbox(
            "Lọc theo Nguyên nhân", ["ALL", "detection", "anomaly", "both", "ok"]
        )
        limit_val = f3.number_input(
            "Số bản ghi tối đa", min_value=10, max_value=500, value=50, step=10
        )

        records = svc.store.list_inspections(
            limit=int(limit_val),
            verdict=None if verdict_filter == "ALL" else verdict_filter,
            reason=None if reason_filter == "ALL" else reason_filter,
        )

        if records:
            df = pd.DataFrame(
                [
                    {
                        "Mã ID": r["id"],
                        "Thời gian (UTC)": r["timestamp_utc"][:19].replace("T", " "),
                        "Tên ảnh": r["filename"],
                        "Kết luận": r["verdict"],
                        "Lý do": r["reason"],
                        "Điểm YOLO": round(r["max_det_confidence"], 3),
                        "Điểm PatchCore": (
                            round(r["anomaly_score"], 3) if r["anomaly_score"] is not None else None
                        ),
                        "Loại lỗi phát hiện": ", ".join(r["defect_classes"]) or "—",
                        "Ảnh đạt chuẩn": "✅" if r["quality"]["passed"] else "⚠️",
                        "Độ trễ (ms)": r["latency_ms"],
                    }
                    for r in records
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

            col_chart, col_gallery = st.columns([1, 1.2])
            with col_chart:
                if stats["defect_class_counts"]:
                    st.subheader("📈 Thống kê tần suất các loại lỗi")
                    st.bar_chart(pd.Series(stats["defect_class_counts"], name="Số lần xuất hiện"))

            with col_gallery:
                ng_records = [
                    r for r in records if r["verdict"] == "NG" and r.get("saved_image_path")
                ]
                if ng_records:
                    st.subheader("🖼️ Kho ảnh sản phẩm lỗi (NG) đã lưu trữ")
                    id_to_fname = {r["id"]: r["filename"] for r in ng_records}
                    selected_id = st.selectbox(
                        "Chọn bản ghi NG để xem lại ảnh bằng chứng:",
                        options=[r["id"] for r in ng_records],
                        format_func=lambda rid: f"Mã #{rid} — {id_to_fname[rid]}",
                    )
                    sel = next(r for r in ng_records if r["id"] == selected_id)
                    img_path = Path(sel["saved_image_path"])
                    if img_path.exists():
                        ng_bgr = cv2.imread(str(img_path))
                        ng_boxed = draw_detections_bgr(ng_bgr, sel["active_detections"])
                        cap = (
                            f"Ảnh lưu trữ NG #{sel['id']} ({sel['filename']}) — "
                            f"Lý do: {sel['reason']} | Lỗi: {sel['defect_classes'] or 'Anomaly'}"
                        )
                        st.image(
                            cv2.cvtColor(ng_boxed, cv2.COLOR_BGR2RGB),
                            caption=cap,
                            width=360,
                        )
        else:
            st.info("Chưa có bản ghi nào khớp với bộ lọc hiện tại.")

    with tab_guide:
        st.subheader("💡 Giải Thích Các Thanh Trượt & Cơ Chế Hoạt Động")
        st.markdown(
            '<div class="guide-card">'
            "<h4>1. Thanh trượt <code>Ngưỡng phát hiện lỗi YOLOv8 (τ_det)</code> là gì?</h4>"
            "<p>Nhánh thứ nhất dùng mô hình <b>YOLOv8n</b> đã được huấn luyện trên 6 loại lỗi "
            "bề mặt thép (<i>vết xước, rỗ bề mặt, đốm mảng, tạp chất, nứt chân chim, vảy cán</i>). "
            "Mỗi khi tìm thấy một vùng nghi ngờ, YOLOv8 gắn cho vùng đó một "
            "<b>điểm tự tin (Confidence) từ 0.00 đến 1.00</b>.</p>"
            "<ul>"
            "<li>Nếu điểm tự tin <b>&ge; τ_det</b> (mặc định <code>0.25</code>): Hệ thống vẽ "
            "khung đỏ và kết luận sản phẩm bị lỗi (<b>NG</b>).</li>"
            "<li>Nếu kéo <b>τ_det thấp xuống</b> (ví dụ <code>0.15</code>): Máy bắt lỗi cực nhạy, "
            "hầu như không để lọt hàng hỏng (giảm <i>Escape Rate</i>), nhưng dễ báo nhầm vết bụi "
            "thành lỗi (tăng <i>Overkill Rate</i>).</li>"
            "<li>Nếu kéo <b>τ_det cao lên</b> (ví dụ <code>0.60</code>): Máy chỉ báo lỗi khi "
            "vết lỗi thật sự rõ ràng.</li>"
            "</ul></div>"
            '<div class="guide-card">'
            "<h4>2. Thanh trượt <code>Ngưỡng điểm bất thường PatchCore (τ_anom)</code> là gì?</h4>"
            "<p>Trong nhà máy thực tế, luôn có nguy cơ xuất hiện <b>loại lỗi mới chưa từng có "
            "trong dữ liệu huấn luyện</b> (ví dụ: linh kiện bị lem sơn màu xanh hoặc dập ngược). "
            "Vì YOLOv8 chỉ biết 6 lỗi cũ nên sẽ bỏ sót.</p>"
            "<p>Nhánh thứ hai dùng <b>PatchCore</b> — mô hình chỉ học từ 220 bức ảnh sản phẩm "
            "<b>Đạt chuẩn (OK)</b> để ghi nhớ <i>bề mặt bình thường trông như thế nào</i>. "
            "Khi có ảnh mới, PatchCore tính <b>Điểm bất thường (Anomaly Score)</b>:</p>"
            "<ul>"
            "<li>Nếu điểm bất thường <b>&ge; τ_anom</b> (mặc định <code>0.53</code>): Bề mặt có "
            "dị tật lạ, lập tức chặn lại và báo <b>NG</b> kèm bản đồ nhiệt (vùng đỏ = lỗi).</li>"
            "<li>Nếu điểm bất thường <b>&lt; τ_anom</b> (ví dụ mẫu <code>metal_nut_good.png</code> "
            "có điểm ~<code>0.44</code>): Sản phẩm được xác nhận đạt chuẩn <b>OK</b>.</li>"
            "</ul></div>"
            '<div class="guide-card">'
            "<h4>3. Nút <code>Bật Early-Exit (Ngắt sớm tăng tốc CPU)</code> dùng để làm gì?</h4>"
            "<p>Vì quy tắc phán quyết cuối cùng là phép <b>HOẶC (OR-Gate)</b>: chỉ cần YOLOv8 "
            "<i>hoặc</i> PatchCore phát hiện lỗi là sản phẩm bị loại (<b>NG</b>). Nhánh YOLOv8 "
            "chạy rất nhanh (<code>~58 ms</code>), trong khi PatchCore nặng hơn "
            "(<code>~114 ms</code>). Khi bật <b>Early-Exit</b>, nếu YOLOv8 đã bắt được lỗi "
            "rõ ràng thì hệ thống kết luận <b>NG</b> ngay lập tức mà không cần chạy thêm "
            "PatchCore, giúp tốc độ kiểm tra trên CPU tăng vọt gấp <b>4.38 lần</b>.</p></div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    main()
