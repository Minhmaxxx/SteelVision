# Nhật ký dự án SteelVision (Tiếng Việt)

> Dòng thời gian làm việc: đã làm gì, quyết định gì, vấp gì, học được gì.
> Số liệu chi tiết nằm ở `docs/reports/experiments.vi.md`; thay đổi code xem
> git history; quy ước dữ liệu nằm ở `docs/dataset-card.md`.

## 2026-10-24 — Khởi động dự án (buổi 1)

**Đã làm**
- Chốt kế hoạch 8 phase với các quyết định: không deadline, huấn luyện trên
  Kaggle/Colab, dữ liệu tự chụp để sau (soạn sẵn quy trình), repo song ngữ,
  tracking bằng W&B.
- Dựng repo: package `src/steelvision`, Makefile, CI (GitHub Actions +
  ruff + pytest), pre-commit, README. 9/9 test pass.
- Pipeline dữ liệu NEU-DET + GC10-DET: tải Kaggle → convert VOC XML→YOLO →
  khử trùng lặp pHash → chia train/val/test theo nhóm. Xác minh NEU-DET có
  nhãn box (rủi ro #1 của tài liệu được gỡ).
- Viết quy trình tự chụp + hướng dẫn gán nhãn (tiếng Việt) để Phase 5 không
  bị chặn khi có thiết bị.

**Vấp phải (2 bài học đắt giá)**
1. `pip install` treo 20+ phút vì backtrack dependency tree (torch +
   anomalib + streamlit) → chuyển sang **uv**, cài xong trong vài phút.
2. **Bug chỉ số nghiêm trọng trong `materialize`**: dùng chỉ số NHÓM để
   truy mảng MẪU → nhãn dán nhầm lên ảnh khác cho mọi mẫu sau cặp ảnh
   gần-trùng nhau, im lặng không báo lỗi. Bắt được nhờ lệch 1 ảnh trong
   đếm (1263 vs 1264). Sửa + thêm assert lúc ghi + test hồi quy end-to-end.
   → Bài học: pipeline dữ liệu phải tự đối chiếu số lượng ở mọi bước.

**Kết quả dữ liệu:** NEU-DET 1.800 ảnh / 4.189 box, chia 1264/270/266;
GC10 2.294 ảnh / 3.563 box (mirror Kaggle dùng tên lớp pinyin Trung Quốc →
bảng rename về tên chuẩn theo paper GC10, bỏ 1 nhãn rác `d`).

## 2026-10-25 — Baseline + ablation (buổi 2)

**Việc phụ:** dọn sạch Docker cũ của dự án khác (~2,9 GB disk + 1,5 GB RAM).

**Chuỗi sự cố kernel Kaggle (6 lần fail, mỗi fail một nguyên nhân)**
1. v1: CLI Kaggle 2.x **bỏ qua** `"enable_gpu"` cũ → train CPU âm thầm 10,7h.
2. v2: đoán trường `"accelerator"` — vẫn sai (đúng là `machine_shape`).
3. v3: GPU xác nhận (thêm cell assert CUDA fail-nhanh) — nhưng `git clone`
   repo **private** treo chờ username → killed sau 2h (exit 137).
4. v4: CLI chỉ upload `code_file`, zip kèm theo bị bỏ qua → nhúng source
   base64 vào notebook.
5. v5: `!python -m` sinh process mới **mất sys.path** → gọi pipeline
   in-process.
6. v6: batch session **cấm kagglehub lazy-attach dataset** → khai báo
   `dataset_sources` (dataset mount tại `/kaggle/input/`).

**Baseline (v7) — thành công**
- YOLOv8n @640, 100 epochs, 17 phút GPU: **test mAP50 0.766** (trên dải
  tham chiếu 0.55–0.65), crazing 0.432 là điểm nghẽn.
- Fingerprint chia tập trong kernel = `b66d053e19f2` = khớp tuyệt đối máy
  local → pipeline tái lập được xuyên OS.
- Weights + log + metrics về `runs/kaggle/neu-det-baseline-final/`.

**Ablation (mỗi lần đổi 1 yếu tố)**
- #1 imgsz 1024: **PHỦ NHẬN** — mAP50 −1.6, crazing −5.8. Ảnh gốc 200×200,
  phóng to chỉ khuyếch đại nhiễu. SAHI loại khỏi NEU.
- #2 mosaic=0.0: **XÁC NHẬN** — mAP50 0.784 (+1.7), crazing +4.4. Cấu hình
  chuẩn mới.
- #3 yolov8s (nền mosaic0): đang chạy.

**Bài học lớn nhất trong ngày:** augmentation mặc định của Ultralytics sinh
ra cho COCO — dataset công nghiệp nhỏ cần thử lại lưới augmentation từ đầu;
và pipeline phải fail-nhanh + fail-ầm-ầm (cell assert, assert đếm) thay vì
treo âm thầm.

## 2026-10-26 — Đóng Phase 2, sẵn sàng Phase 3 (buổi 3)

**Đã làm**
- Ablation #2 (mosaic=0.0): **DƯƠNG** — mAP50 0.766→0.784, crazing 0.432→0.476
  → chốt cấu hình chuẩn: yolov8n + imgsz 640 + mosaic=0.0.
- Ablation #3 (yolov8s): **ÂM** — 0.776, overfit dataset nhỏ. Bảng tổng hợp
  ablation hoàn chỉnh, điều kiện Phase 2 (≥3 thay đổi) đạt.
- Run GC10-DET (v11) với cấu hình chuẩn: mAP50 0.679 — khó hơn NEU, hai lớp
  nghèo dữ liệu (rolled_pit 32 box train, inclusion 141) rơi đúng chỗ thấp.
  Pipeline chứng minh tổng quát cho dataset thứ 2.
- Dọn template notebook: bỏ cell prep thừa còn sót, khôi phục cell assert GPU
  (phát hiện khi thêm tham số `--dataset` cho build_kernel).
- Nhật ký này + báo cáo cập nhật sau mỗi kết quả.

**Vấp phải (nhỏ)**
- Fingerprint GC10 kernel khác local dù thành phần chia tập khớp tuyệt đối —
  do `sample_id` nhạy layout thư mục (kagglehub cache vs `/kaggle/input`).
  Việc nhỏ: đổi sample_id sang tên XML. NEU thì khớp 100% xuyên OS.

**Bài học**
- Ablation cho kết quả âm cũng là kết quả: 2/3 giả thuyết bị số liệu phủ
  nhận (imgsz, model size) — tránh được 2 hướng đi tốn công vô ích ở Phase sau.

**Trạng thái:** Phase 2 ĐÓNG. Tiếp: Phase 3 — PatchCore trên MVTec metal_nut
(tải MVTec AD về local trước).

## 2026-10-26 — Phase 3: PatchCore Anomaly Detection + Decision Fusion (buổi 4)

**Đã làm**
- Xây dựng module hợp nhất quyết định `src/steelvision/decision/fusion.py`:
  phán định `NG` nếu `max(box.conf) >= τ_det` **hoặc** `anomaly_score >= τ_anom`,
  phân loại rõ `reason` (`ok` / `detection` / `anomaly` / `both`), hỗ trợ cả chế độ
  chạy độc lập từng nhánh và quét ngưỡng nhanh. Viết 8 unit test trong `tests/test_fusion.py`.
- Xây dựng pipeline PatchCore `src/steelvision/models/anomaly.py` + wrapper suy luận
  CPU `OnnxAnomalyDetector` + notebook `notebooks/02_train_patchcore.ipynb` + unit test
  `tests/test_anomaly.py`.
- Huấn luyện và đánh giá PatchCore (`wide_resnet50_2`, `coreset_sampling_ratio=0.01`,
  `256×256`) trên `MVTec AD (metal_nut)` (220 ảnh `train/good`, 115 ảnh `test`):
  - **image_AUROC = 0.9980** (99.80%), **image_F1Score = 0.9840**
  - **pixel_AUROC = 0.9869** (98.69%), **pixel_F1Score = 0.8445**
  - Tại $\tau_{\text{anom}} = 0.50$, bắt trúng **93/93 (100%)** sản phẩm lỗi ở cả 4 nhóm
    (`bent`, `color`, `flip`, `scratch` — escape rate = 0%).
- Xuất mô hình sang ONNX (`runs/patchcore/metal_nut/weights/onnx/patchcore.onnx`) và
  kiểm chứng suy luận bằng ONNX Runtime CPU (`190.5 ms/ảnh`, nhanh hơn PyTorch ~32%).

**Vấp phải (2 lỗi khi xuất ONNX trên Windows với PyTorch 2.x + Anomalib 2.7)**
1. Bộ xuất ONNX dynamo của PyTorch 2.x in ký tự `✅` (`\u2705`) ra stdout khi
   capture graph xong → gãy `UnicodeEncodeError` trên console `cp1252` của Windows.
   → Sửa: ép `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` trong pipeline.
2. Mặc định `anomalib 2.7` sinh dummy input có `batch_size = 2` khi dùng `dynamic_axes`
   và hạ opset từ 18 về 14 → `PatchcoreModel.compute_anomaly_score` bị cố định `batch=2`
   trong đồ thị ONNX.
   → Sửa: truyền `onnx_kwargs={"opset_version": 18, "dynamic_axes": {}, "dynamic_shapes": None}`
   để xuất đồ thị tĩnh chuẩn `batch=1` (`1×3×256×256`).

**Trạng thái:** Phase 3 ĐÓNG (`24/24` pytest pass, ruff sạch). Tiếp: Phase 4 —
Đánh giá QC (`escape rate`, `overkill rate`, đường đánh đổi ngưỡng) + kiểm thử độ bền
và bài test "lỗi chưa từng gặp".

## 2026-10-26 — Phase 4: Đánh giá QC, Lỗi chưa từng gặp & Độ bền (buổi 5)

**Đã làm**
- Xây dựng [`src/steelvision/eval/qc_metrics.py`](file:///e:/Vision/src/steelvision/eval/qc_metrics.py):
  tính `escape_rate`, `overkill_rate` cấp sản phẩm, quét lưới ngưỡng `sweep_thresholds`
  và tự động chọn điểm vận hành `select_operating_point` trên tập `val` theo mục tiêu
  `escape_rate <= 2.0%`.
- Xây dựng [`src/steelvision/eval/robustness.py`](file:///e:/Vision/src/steelvision/eval/robustness.py):
  6 phép biến đổi môi trường (`brightness_up`, `brightness_down`, `contrast_low`,
  `gaussian_blur`, `gaussian_noise`, `rotate_5deg`) + bộ tính VOC/COCO 101-point
  interpolated `AP50` phục vụ đánh giá độ bền và khác miền.
- Xây dựng [`src/steelvision/eval/run_qc.py`](file:///e:/Vision/src/steelvision/eval/run_qc.py)
  và cấu hình [`configs/eval.yaml`](file:///e:/Vision/configs/eval.yaml) (`make eval`):
  - **PatchCore (`MVTec metal_nut`)**: chọn 	au_anom = 0.53 từ `val` →
    trên tập `test` đạt **escape rate = `0.00%`**, **overkill rate = `0.00%`** (`4.55%`
    trên toàn tập 115 ảnh, `F1 = 0.9947`).
  - **YOLOv8n (`NEU-DET` + 80 mẫu nền thép OK mỗi split)**: chọn 	au_det = 0.25
    từ `val` (`escape 1.85%`) → trên tập `test` đạt **escape rate = `0.75%` (`2/266`)**,
    `overkill = 22.50%`, `F1 = 0.9635` (giảm 12.5× tỷ lệ lọt lỗi so với ngưỡng cảm tính
    `0.50` vốn để lọt `9.40%`).
  - **Test "lỗi chưa từng gặp" (Leave-One-Class-Out):** khi ẩn 1 lớp lỗi bất kỳ,
    detector đơn lẻ để lọt `14.7%–17.3%` trên NEU-DET và `23.7%–26.9%` trên `metal_nut`;
    khi hợp nhất với PatchCore (`fusion.py`), tỷ lệ lọt lỗi về **`0.00%`** ở cả 4 nhóm lỗi.
  - **Kiểm thử độ bền & khác miền (`NEU → GC10`):** chỉ ra 2 phát hiện kỹ thuật quan trọng:
    (1) YOLOv8n nhạy với mờ/nhiễu hạt còn PatchCore giữ AUROC `0.979–0.999` nhưng bị dịch
    thang điểm gây tăng overkill khi tối/nhiễu → cần Image Quality Gate ở bước tiền xử lý;
    (2) Cross-domain `NEU → GC10` sụt AP50 từ `0.749` xuống `0.060` do khác trường nhìn
    (`200×200` vs `2048×1000`) và độ bóng bề mặt → khẳng định vai trò của dữ liệu tự chụp Phase 5.
- Viết báo cáo chi tiết tại [`docs/reports/qc_robustness.vi.md`](file:///e:/Vision/docs/reports/qc_robustness.vi.md)
  và unit test (`tests/test_qc_metrics.py`, `tests/test_robustness.py` — `30/30` pass).

**Trạng thái:** Phase 4 ĐÓNG (`30/30` pytest pass, ruff sạch). Tiếp: Phase 6 (ONNX CPU
benchmark & tối ưu hiệu năng) hoặc Phase 5 (khi có dữ liệu tự chụp).

## 2026-10-26 — Phase 5 Tooling & Phase 6: Tối ưu ONNX CPU & Benchmark Latency/RAM (buổi 6)

**Đã làm**
- **Dọn dung lượng ổ đĩa:** Kiểm toán dung lượng toàn dự án và dọn cache/dữ liệu thừa.
- **Công cụ dữ liệu tự chụp (Phase 5 — `src/steelvision/data/self_captured.py`):**
  - Kiểm tra `manifest.csv` theo đúng `docs/capture-protocol.vi.md` (tỷ lệ OK ≥ 40%,
    cân bằng sáng A/B/C, điều kiện dừng ≥ 300 ảnh QC-passed).
  - Gom nhóm theo `(panel, position)` để 3 điều kiện sáng A/B/C của cùng vị trí không
    rò rỉ giữa `train/val/test`.
  - Đo độ nhất quán nhãn khi gán lại 10% (`compute_label_consistency`: mean box IoU ≥ 0.75,
    image F1 ≥ 0.90).
- **Tối ưu & Benchmark CPU (Phase 6 — `src/steelvision/models/detector.py`, `src/steelvision/eval/benchmark.py`):**
  - Xuất `YOLOv8n` (`runs/yolo/neu_det_best.pt`) sang `runs/yolo/neu_det_best.onnx`
    (`opset 18`, tĩnh `batch=1`) và viết `OnnxYoloDetector` thuần `onnxruntime` + NumPy NMS
    (độ lệch `max_conf` so với PyTorch chỉ `1e-6`, đồng thuận phán định `100%`).
  - Đo hiệu năng CPU (`AMD Ryzen 7 5800H`, `batch=1`, 10 warm-up + 50 timed runs):
    - **YOLOv8n (`640×640`):** `p50` giảm từ `66.65 ms` (PyTorch) xuống **`58.37 ms`** (ONNX Runtime, `17.13 FPS`), RSS RAM giảm từ `1,166 MB` xuống **`241 MB`** (`4.8×` nhẹ hơn).
    - **PatchCore (`256×256`):** `p50` giảm từ `210.89 ms` (PyTorch) xuống **`114.47 ms`** (ONNX Runtime, nhanh gấp **`1.84×`**, `p95` giảm `-47.9%` từ `237.22 ms` xuống `123.53 ms`), RSS RAM giảm từ `1,295 MB` xuống **`571 MB`**.
    - **Pipeline End-to-End (`YOLO + PatchCore + Fusion`):**
      - Chế độ Full Dual-Branch ONNX: `p50` giảm từ `279.60 ms` xuống **`236.23 ms`** (`p95` giảm từ `361.33 ms` xuống `282.66 ms`), RSS RAM giảm **51.7%** (`1,297 MB → 626 MB`).
      - Chế độ Early-Exit ONNX (tận dụng tính chất `OR` của luật quyết định khi YOLO đã bắt được lỗi quen thuộc): `p50` giảm từ **`279.60 ms` xuống `63.78 ms` (nhanh gấp `4.38×`, `15.68 FPS`)** mà không đổi bất kỳ phán định OK/NG nào.
- Ghi báo cáo chi tiết vào [`docs/reports/performance.vi.md`](file:///e:/Vision/docs/reports/performance.vi.md) và thêm `make benchmark` vào `Makefile`. `35/35` unit test pass, ruff sạch.

**Trạng thái:** Phase 6 ĐÓNG. Tiếp: Phase 7 — Dịch vụ FastAPI (`/inspect` + SQLite lưu vết) + Dashboard Streamlit + Docker Compose + Hoàn thiện bàn giao.

## 2026-10-26 — Phase 7: Dịch vụ FastAPI, SQLite Traceability, Streamlit Dashboard, Docker & Bàn giao (buổi 7)

**Đã làm**
- Xây dựng cổng kiểm tra chất lượng ảnh trước suy luận [`src/steelvision/api/quality.py`](file:///e:/Vision/src/steelvision/api/quality.py) (`check_image_quality`): đo độ sáng trung bình, độ tương phản và phương sai độ nét Laplacian để gắn cờ `too_dark`, `too_bright`, `low_contrast`, `blurry` (giải quyết trực tiếp hiện tượng tăng overkill khi sụt nguồn sáng phát hiện ở Phase 4).
- Xây dựng kho lưu vết SQLite [`src/steelvision/api/store.py`](file:///e:/Vision/src/steelvision/api/store.py) (`TraceabilityStore`): lưu mọi lần kiểm tra vào bảng `inspections` (thời gian UTC, tên file, kích thước, phán quyết `OK/NG`, lý do `ok/detection/anomaly/both`, điểm tin cậy, phiên bản ngưỡng `v1-val-2026-10-26`, danh sách lỗi, kết quả quality gate, latency và đường dẫn ảnh NG lưu trữ).
- Xây dựng dịch vụ FastAPI [`src/steelvision/api/app.py`](file:///e:/Vision/src/steelvision/api/app.py) (`GET /health`, `POST /inspect`, `GET /inspections`, `GET /inspections/{id}`, `GET /stats`) và giao diện quản trị QC [`src/steelvision/dashboard/app.py`](file:///e:/Vision/src/steelvision/dashboard/app.py) (Streamlit dashboard: KPI tổng quan, kiểm tra trực tiếp hiển thị bbox + bản đồ nhiệt PatchCore, bảng lịch sử lọc theo `OK/NG`/lớp lỗi và thư viện ảnh NG).
- Đóng gói [`Dockerfile`](file:///e:/Vision/Dockerfile), [`docker-compose.yml`](file:///e:/Vision/docker-compose.yml), [`.dockerignore`](file:///e:/Vision/.dockerignore) và viết bộ test tự động [`tests/test_api.py`](file:///e:/Vision/tests/test_api.py) (`39/39` unit test toàn repo pass).
- Viết script tạo video demo 60 giây [`scripts/generate_demo_video.py`](file:///e:/Vision/scripts/generate_demo_video.py) → xuất `runs/demo_steelvision.mp4` (`1280×720`, `30 FPS`, `15.68 MB`) và hoàn thiện toàn bộ tài liệu bàn giao: [`docs/model-card.md`](file:///e:/Vision/docs/model-card.md), [`docs/lessons-learned.vi.md`](file:///e:/Vision/docs/lessons-learned.vi.md), [`README.md`](file:///e:/Vision/README.md), [`SteelVision_Project_Overview.md`](file:///e:/Vision/SteelVision_Project_Overview.md).

**Trạng thái:** Phase 7 ĐÓNG — Dự án hoàn tất toàn bộ 7 Phase.

## Quy ước duy trì nhật ký

- Sau mỗi buổi làm việc: bổ sung mục theo ngày (đã làm / vấp phải / bài học /
  kết quả), giữ ngắn gọn — số chi tiết đẩy về báo cáo thí nghiệm.
- Quyết định thay đổi hướng đi của user ghi lại đúng ngày, kèm lý do.
