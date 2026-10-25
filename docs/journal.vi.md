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

## Quy ước duy trì nhật ký

- Sau mỗi buổi làm việc: bổ sung mục theo ngày (đã làm / vấp phải / bài học /
  kết quả), giữ ngắn gọn — số chi tiết đẩy về báo cáo thí nghiệm.
- Quyết định thay đổi hướng đi của user ghi lại đúng ngày, kèm lý do.
