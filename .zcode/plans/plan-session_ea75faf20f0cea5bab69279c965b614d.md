# Kế hoạch thực hiện SteelVision

## Bối cảnh đã xác minh
- **NEU-DET** (1.800 ảnh, 6 lớp) và **GC10-DET** (3.570 ảnh, 10 lớp) đều có nhãn bounding box **PASCAL VOC XML** → chỉ cần 1 script chuyển XML→YOLO dùng chung. Nguồn: Kaggle (neu-surface-defect-database, gc10det), Dataset Ninja. Rủi ro số 1 trong tài liệu đã được gỡ.
- **Anomalib v2.2** (PatchCore) cần **Python 3.10–3.12** → chọn **Python 3.11**.
- Huấn luyện trên **Kaggle/Colab** (YOLOv8n trên NEU-DET ~30–60 phút/GPU session, nằm trong quota Kaggle). Theo dõi thí nghiệm bằng **W&B** (chạy mượt trên Colab/Kaggle, cần tài khoản miễn phí) — nếu bạn muốn MLflow local thì đổi sau được.

## Cấu trúc repo (git init tại E:\Vision)
```
E:\Vision\
├── data/                  # gitignored: raw/ interim/ processed/
├── src/steelvision/       # package: data/ models/ eval/ decision/ api/ dashboard/
├── notebooks/             # notebook Kaggle/Colab (huấn luyện)
├── configs/               # yaml: dataset, model, ngưỡng quyết định
├── tests/                 # pytest
├── docs/                  # labeling-guide (VI), báo cáo thí nghiệm/QC (VI), model card (EN)
├── docker/                # Dockerfile + compose
├── Makefile, pyproject.toml, README.md (EN), .github/workflows/ci.yml, .pre-commit-config.yaml
```
Quy ước: **local Windows (CPU)** — chuẩn bị dữ liệu, đánh giá, benchmark, API, dashboard; **Kaggle/Colab** — huấn luyện; notebook commit vào `notebooks/`, weights/metrics tải về `runs/` (gitignored, chỉ bảng tổng hợp vào git).

## Các giai đoạn (mỗi phase có điều kiện hoàn thành rõ, không deadline)

**Phase 0 — Nền tảng repo (bắt đầu ngay)**
1. Git init + khung repo + `pyproject.toml` + Makefile (`make data/train/eval/serve` dạng stub) + pre-commit (ruff) + CI GitHub Actions (lint + pytest).
2. Tạo venv Python 3.11, cài core deps (ultralytics, anomalib, albumentations, opencv, onnxruntime, fastapi, streamlit, pytest, wandb, imagehash), smoke test import.
✅ Xong khi: repo push lên GitHub, CI xanh.

**Phase 1 — Dữ liệu công khai**
1. Script tải NEU-DET, GC10-DET, MVTec AD (metal_nut) vào `data/raw/` (kagglehub).
2. `convert_voc_to_yolo.py` dùng chung (class map, kiểm tra box hợp lệ, kích thước tối thiểu).
3. Notebook EDA: phân bố lớp, thống kê kích thước box (lỗi nhỏ → quyết định imgsz và nhu cầu SAHI).
4. Chia train/val/test **theo ảnh** + khử trùng lặp bằng perceptual hash (imagehash).
5. Dataset card: nguồn, giấy phép, cách tái tạo.
✅ Xong khi: `data/processed/` chuẩn YOLO, EDA committed.

**Phase 2 — Baseline detection (Kaggle)**
1. `01_train_yolo_baseline.ipynb`: YOLOv8n trên NEU-DET (imgsz 640) → mAP50/mAP50-95 baseline, log W&B.
2. Chạy GC10 riêng, rồi ablation **mỗi lần đổi 1 yếu tố**: imgsz 640→1024, augmentation, mất cân bằng lớp, n→s.
3. Notebook phân tích lỗi: AP theo lớp, theo kích thước box, confusion matrix, gallery ảnh fail.
4. Ghi baseline cũ (classification + Grad-CAM++) vào bảng so sánh.
✅ Xong khi: baseline + ≥3 ablation có bảng kết quả trong `docs/reports/`.

**Phase 3 — Anomaly detection (PatchCore)**
1. Chạy PatchCore trên **MVTec metal_nut** trước (có mask anomaly chuẩn) → sanity check với AUROC gần mức paper (~99%).
2. Báo cáo image-level + pixel-level AUROC; xuất OpenVINO/ONNX.
3. Module `decision/fusion.py`: NG nếu (box conf ≥ τ_det) **hoặc** (anomaly score ≥ τ_anom) + test unit test.
✅ Xong khi: AUROC khớp kỳ vọng, fusion module có test.

**Phase 4 — Đánh giá QC + độ bền (trên dữ liệu công khai)**
1. `eval/qc_metrics.py`: quy về OK/NG cấp ảnh → **escape rate, overkill rate**, quét ngưỡng → đường đánh đổi; chọn điểm vận hành trên val, báo test.
2. `eval/robustness.py`: sáng/tương phản, nhiễu, mờ, xoay nhẹ → bảng sụt giảm; cross-domain: train NEU test GC10 (và ngược lại).
3. Test "lỗi chưa từng gặp": để dành 1 lớp lỗi khỏi training, kiểm chứng PatchCore bắt được.
✅ Xong khi: báo cáo QC (VI) có đường escape/overkill + bảng độ bền.

**Phase 5 — Dữ liệu tự chụp (khởi động khi có thiết bị, không chờ)**
1. Soạn ngay **quy trình chụp (VI)**: 300–500 ảnh, ≥40% OK, 2–3 điều kiện sáng, khoảng cách/góc cố định, manifest CSV.
2. Hướng dẫn gán nhãn (CVAT/Label Studio) + xuất YOLO; gán lại 10% để đo nhất quán.
3. Fine-tune từ weights Phase 2 → chạy lại toàn bộ QC eval trên test tự chụp (đây là số điền vào CV).
✅ Xong khi: dataset tự chụp có tài liệu + mô hình fine-tuned + số QC trên dữ liệu thật.

**Phase 6 — Tối ưu hiệu năng (CPU, chạy local)**
1. Export ONNX → benchmark ONNX Runtime CPU batch=1 (warm-up): **latency p50/p95, RAM, FPS** trước/sau.
2. Tuỳ chọn nếu còn điều kiện: OpenVINO INT8, SAHI cho lỗi nhỏ.
✅ Xong khi: bảng hiệu năng trước/sau trong docs.

**Phase 7 — Dịch vụ + bàn giao**
1. FastAPI `/inspect` (ảnh → tiền xử lý → detector+anomaly → quyết định → JSON) + **SQLite lưu vết** (thời gian, verdict, điểm số, phiên bản ngưỡng) + lưu ảnh NG.
2. Dashboard Streamlit: duyệt lịch sử, lọc NG, xem box/heatmap, thống kê.
3. Docker (CPU) + `docker compose up` api+dashboard; pytest (API, fusion rule, converter); CI chạy test.
4. README (EN, có mục Hạn chế), model card, video demo 60–90s, bài học rút ra (VI), mẫu CV điền số thật.
✅ Xong khi: `docker compose up` chạy end-to-end, đủ 8 sản phẩm bàn giao của tài liệu.

## Việc đầu tiên ngay khi duyệt
1. Phase 0 trọn vẹn: git init, khung repo, venv 3.11, CI, Makefile.
2. Phase 1 bước 1–2: script tải NEU-DET + GC10-DET, converter XML→YOLO, chạy trên máy thật.
3. Notebook baseline YOLOv8n sẵn sàng để bạn mở Kaggle chạy buổi tiếp theo.

## Quyết định mặc định (đổi được sau)
- W&B thay MLflow cho tracking (phù hợp Kaggle/Colab); MLflow thêm sau nếu muốn local.
- MVTec AD dùng nhóm **metal_nut** làm chính, screw tuỳ chọn.
- Repo GitHub remote sẽ cần bạn tạo (tôi chuẩn bị sẵn lệnh push).