# Báo cáo thí nghiệm — SteelVision (Tiếng Việt)

> Mọi con số trong README/CV đều truy nguồn về báo cáo này. Cập nhật mỗi khi có
> run mới; run gốc log trên Kaggle kernel `lngnhtminh/steelvision-01-yolo-baseline`.

## 1. Baseline detection: YOLOv8n trên NEU-DET (run 2026-10-25, kernel v7)

**Cấu hình:** YOLOv8n pretrained COCO · imgsz 640 · batch 16 · 100 epochs · seed 42 ·
data pipeline = pipeline repo (convert VOC→YOLO, pHash dedupe, split theo nhóm).

**Môi trường:** Kaggle GPU Tesla T4 (15 GB), Python 3.13, torch 2.11+cu128,
Ultralytics 8.4.188. Thời gian train: **0.282 giờ (~17 phút)** cho 100 epochs.

**Tái lập:** fingerprint chia tập trong kernel = `b66d053e19f2` — **khớp tuyệt đối
với máy local (Windows)**: 1.800 ảnh, 4.189 box, 0 box lỗi, chia 1.264/270/266.

### Kết quả trên tập test (266 ảnh, chưa từng thấy trong train/val)

| Chỉ số | Giá trị |
|---|---|
| **mAP50** | **0.766** |
| mAP50-95 | 0.450 |
| Precision | 0.732 |
| Recall | 0.709 |

Tham chiếu công khai cho YOLO-family trên NEU-DET thường ở mAP50 ~0.55–0.65 →
baseline này nằm **trên** dải tham chiếu (khác biệt có thể do: chia tập có khử
trùng lặp, convert sạch, không rò rỉ val/test).

### AP50 theo lớp

| Lớp | AP50 | Nhận xét |
|---|---|---|
| scratches | 0.909 | Tốt nhất — vết trầy tuyến tính, đặc trưng rõ |
| patches | 0.878 | |
| inclusion | 0.868 | |
| pitted_surface | 0.836 | |
| rolled_in_scale | 0.674 | |
| **crazing** | **0.432** | **Điểm nghẽn** — nứt mạng nhện dạng texture mịn, box nhỏ, biên mơ hồ |

### Đọc hiểu

- 5/6 lớp ≥ 0.67 — mô hình nắm tốt các lỗi có hình thái rõ.
- **crazing kéo mAP tổng xuống**: đây là lỗi dạng kết cấu (texture) với box rất
  nhỏ và biên không rõ ràng — vấn đề phân giải đầu vào nhiều hơn là vấn đề mô hình.
- Lưu ý dữ liệu: NEU-DET gốc có vài box trùng lặp trong cùng ảnh
  (Ultralytics tự loại 1 box duplicate ở `crazing_120`) — cân nhắc khử trùng
  box trong converter (việc nhỏ, làm khi tiện).

### Hệ quả cho ablation (mỗi lần đổi 1 yếu tố)

1. ~~imgsz 640 → 1024~~ → **đã chạy, kết quả âm — xem mục 2**.
2. Augmentation: tắt mosaic (mixup mặc định đã tắt) — NEU-DET nhỏ, mosaic có thể
   phá ngữ cảnh texture.
3. Model size n → s: thêm ~2.5× tham số, kiểm tra tăng thực.
4. GC10-DET: dataset thứ 2 + kiểm thử khác miền (Phase 4).

## 2. Ablation #1: imgsz 640 → 1024 (run 2026-10-25, kernel v8)

**Giả thuyết:** crazing là lỗi texture mịn → tăng độ phân giải đầu vào sẽ giúp.

### So sánh trên cùng tập test

| Chỉ số | 640 (baseline) | 1024 | Δ |
|---|---|---|---|
| mAP50 | **0.766** | 0.750 | −0.016 |
| mAP50-95 | **0.450** | 0.415 | −0.035 |
| Precision | **0.732** | 0.637 | −0.095 |
| Recall | 0.709 | **0.762** | +0.053 |
| crazing AP50 | **0.432** | 0.374 | −0.058 |

### Kết luận

- **Giả thuyết bị phủ nhận**: mAP50 giảm 1.6 điểm, crazing còn giảm 5.8 điểm.
- Giải thích hợp lý: ảnh NEU gốc chỉ 200×200 — phóng lên 1024 **không tạo thêm
  thông tin**, chỉ khuyếch đại nhiễu nén/quét khiến nền trông giống lỗi →
  precision rơi mạnh (−9.5) dù recall tăng (+5.3): mô hình "báo động giả" nhiều hơn.
- Hệ quả: (a) vấn đề của crazing **không nằm ở độ phân giải** mà ở tính chất
  texture + biên nhãn mơ hồ; (b) SAHI/tiling **không áp dụng cho NEU** (ảnh nhỏ
  sẵn) — chỉ cân nhắc cho dữ liệu tự chụp độ phân giải cao ở Phase 5.
- imgsz 640 giữ nguyên làm cấu hình chuẩn.

## 2. Nhật ký hạ tầng huấn luyện (bài học Kaggle)

Xem `memory`/phiên 2026-10-25: 4 lỗi batch-session đã sửa (GPU flag `machine_shape`,
CLI chỉ upload code_file → nhúng source base64, `!python -m` mất sys.path → gọi
in-process, batch session cấm kagglehub lazy-attach → `dataset_sources`).
Flow chuẩn hiện nay: `make kaggle-push`.
