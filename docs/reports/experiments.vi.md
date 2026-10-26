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

## 3. Ablation #2: tắt mosaic (mosaic=0.0) (run 2026-10-25, kernel v9)

**Giả thuyết:** NEU-DET nhỏ (1.264 ảnh train), mosaic ghép 4 ảnh phá ngữ cảnh
texture → tắt mosaic sẽ giúp các lớp texture.

### So sánh với baseline (cùng imgsz 640)

| Chỉ số | baseline | mosaic=0.0 | Δ |
|---|---|---|---|
| **mAP50** | 0.766 | **0.784** | **+1.7** |
| mAP50-95 | 0.450 | 0.456 | +0.5 |
| Precision | 0.732 | 0.742 | +1.0 |
| Recall | 0.709 | 0.735 | +2.6 |
| **crazing AP50** | 0.432 | **0.476** | **+4.4** |
| scratches AP50 | 0.909 | 0.942 | +3.3 |
| patches AP50 | 0.878 | 0.905 | +2.7 |
| inclusion AP50 | 0.868 | 0.855 | −1.3 |

### Kết luận

- **Giả thuyết được xác nhận**: tắt mosaic cải thiện tổng thể, mạnh nhất ở
  crazing (+4.4) và scratches (+3.3) — đúng nhóm lớp nhạy texture. Chỉ inclusion
  giảm nhẹ (−1.3).
- **Cấu hình chuẩn mới: imgsz 640 + mosaic=0.0** (mAP50 0.784). Các ablation
  sau xếp chồng trên cấu hình này.
- Bài học chung: augmentation mặc định của Ultralytics sinh ra cho COCO — với
  dataset công nghiệp nhỏ, lưới augmentation cần thử lại từ đầu.

## 4. Ablation #3: model size n → s (yolov8s, nền mosaic=0.0) (run 2026-10-25, kernel v10)

| Chỉ số | yolov8n+mosaic0 | yolov8s+mosaic0 | Δ |
|---|---|---|---|
| mAP50 | **0.784** | 0.776 | −0.8 |
| mAP50-95 | **0.456** | 0.437 | −1.8 |
| crazing AP50 | **0.476** | 0.445 | −3.0 |
| scratches AP50 | **0.942** | 0.910 | −3.2 |
| patches AP50 | 0.905 | **0.919** | +1.4 |

**Kết luận:** giả thuyết bị phủ nhận — 1.264 ảnh train không nuôi được mô hình
~2.5× tham số (dấu hiệu overfit). **Yolov8n giữ nguyên.**

## 5. Bảng tổng hợp ablation (điều kiện Phase 2: ≥3 thay đổi — ĐẠT)

| Run | Thay đổi | mAP50 | mAP50-95 | crazing | Kết luận |
|---|---|---|---|---|---|
| v7 baseline | — (yolov8n, 640) | 0.766 | 0.450 | 0.432 | mốc so sánh |
| v8 | imgsz → 1024 | 0.750 | 0.415 | 0.374 | ❌ khuyếch đại nhiễu |
| v9 | mosaic → 0.0 | **0.784** | **0.456** | **0.476** | ✅ **chốt làm chuẩn** |
| v10 | model → yolov8s | 0.776 | 0.437 | 0.445 | ❌ overfit dataset nhỏ |

**Cấu hình detector cuối Phase 2: YOLOv8n + imgsz 640 + mosaic=0.0 → mAP50 0.784.**

## 6. GC10-DET với cấu hình chuẩn (run 2026-10-26, kernel v11)

**Cấu hình:** YOLOv8n @640 + mosaic=0.0, 100 epochs, seed 42 — đúng công thức
đã chốt ở §5, chỉ đổi dataset. Kernel chạy 43 phút (chuẩn bị dữ liệu + train).

### Kết quả trên tập test GC10 (349 ảnh, 10 lớp)

| Chỉ số | Giá trị |
|---|---|
| mAP50 | 0.679 |
| mAP50-95 | 0.317 |
| Precision / Recall | 0.658 / 0.701 |

AP50 theo lớp: punching **0.986** · crescent_gap 0.931 · weld_line 0.876 ·
water_spot 0.809 · silk_spot 0.678 · oil_spot 0.690 · waist_folding 0.657 ·
crease 0.595 · **inclusion 0.321** · **rolled_pit 0.242**.

### Đọc hiểu

- Khó hơn NEU (0.679 vs 0.784) — hợp lý: 10 lớp, mất cân bằng nặng
  (silk_spot 514 box train vs rolled_pit 32).
- **Hai lớp nghèo dữ liệu rơi đúng chỗ nghèo hiệu năng**: rolled_pit (chỉ 32
  box train → AP 0.242), inclusion (141 box → 0.321). Đây là bằng chứng rõ
  cho việc xử lý mất cân bằng lớp — hướng tiếp theo nếu cần nâng GC10.
- Mục đích chính của run: chứng minh pipeline tổng quát cho dataset thứ 2 ✓,
  và làm nền cho kiểm thử cross-domain (Phase 4: train NEU test GC10 & ngược lại).

### Ghi chú tái lập (trung thực)

Fingerprint GC10 trong kernel (`05e356eb691d`) khác local (`732df69e4388`)
NHƯNG thành phần chia tập khớp tuyệt đối từng lớp từng split (so trực tiếp
2 bản log). Nguyên nhân: `sample_id` dùng đường dẫn tương đối — thư mục
kagglehub cache (local) và mount `/kaggle/input` (kernel) khác cấp thư mục
gốc → khác chuỗi id, cùng một phép chia. Việc nhỏ để làm sau: đổi
`sample_id` sang dùng tên XML (bất biến với layout) để fingerprint so được
xuyên môi trường.

## 7. Trạng thái Phase 2: ĐÓNG

Điều kiện hoàn thành Phase 2 theo kế hoạch: baseline ✓ (v7), ≥3 ablation ✓
(imgsz1024 ❌, mosaic0 ✅, yolov8s ❌), bảng kết quả trong docs ✓, dataset
thứ 2 ✓ (GC10), phân tích lỗi theo lớp ✓. → Chuyển **Phase 3: PatchCore
anomaly detection (MVTec metal_nut)**. với cấu hình chuẩn

## 2. Nhật ký hạ tầng huấn luyện (bài học Kaggle)

Xem `memory`/phiên 2026-10-25: 4 lỗi batch-session đã sửa (GPU flag `machine_shape`,
CLI chỉ upload code_file → nhúng source base64, `!python -m` mất sys.path → gọi
in-process, batch session cấm kagglehub lazy-attach → `dataset_sources`).
Flow chuẩn hiện nay: `make kaggle-push`.
