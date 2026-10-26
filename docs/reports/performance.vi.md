# Báo cáo Tối ưu Hiệu năng CPU (Phase 6) — SteelVision

> Toàn bộ số liệu trong báo cáo này được đo thật trên máy local CPU (`batch = 1`,
> `10` lượt warm-up + `50` lượt đo bấm giờ `time.perf_counter()`, ảnh nạp sẵn trong
> RAM để loại nhiễu đọc đĩa) và tái lập hoàn toàn bằng lệnh:
> ```bash
> make benchmark   # hoặc: python -m steelvision.eval.benchmark --config configs/benchmark.yaml
> ```
> Kết quả JSON gốc lưu tại `runs/eval/benchmark_report.json`.

---

## 1. Cấu hình phần cứng & Phương pháp đo

- **CPU:** AMD Ryzen 7 5800H with Radeon Graphics (`8` nhân vật lý / `16` luồng logic, Windows 11 64-bit).
- **Môi trường phần mềm:** Python `3.11.9`, PyTorch `2.14.1+cpu`, Ultralytics `8.4.187`, Anomalib `2.7.0`, ONNX `1.23.2` (`opset 18`), ONNX Runtime `1.31.0` (`CPUExecutionProvider`, `GraphOptimizationLevel.ORT_ENABLE_ALL`).
- **Quy trình đo công bằng:**
  1. Đo **ONNX Runtime thuần** trước khi `import torch / ultralytics / anomalib` để ghi nhận chính xác dung lượng **RSS RAM (`WorkingSetSize`)** thực tế khi triển khai dịch vụ nhẹ (không gánh DLL của PyTorch).
  2. Mỗi cấu hình chạy `10` lượt warm-up (không tính giờ) rồi chạy `50` lượt bấm giờ liên tiếp với `batch = 1` (đúng kịch bản kiểm tra từng sản phẩm trên băng chuyền).
  3. Đối chiếu độ lệch số học (`numerical parity`) giữa PyTorch và ONNX Runtime trên tập test.

---

## 2. Bảng tổng hợp hiệu năng Trước và Sau tối ưu (`batch = 1`)

### 2.1. Từng mô hình độc lập (YOLOv8n @ `640×640` & PatchCore @ `256×256`)

| Mô hình / Cấu hình | Mean ± Std (ms) | **Latency p50 (ms)** | **Latency p95 (ms)** | Min / Max (ms) | **Throughput (FPS)** | **RSS RAM (MB)** | Kích thước file (MB) | Tăng tốc p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **YOLOv8n — PyTorch CPU** | 65.87 ± 5.13 | **66.65** | **72.39** | 55.11 / 80.02 | 15.00 | 1,166.22 | 6.21 | 1.00× (mốc) |
| **YOLOv8n — ONNX Runtime (default threads)** | 59.44 ± 5.05 | **58.37** | **67.60** | 52.72 / 85.31 | **17.13** | **241.38** | 12.16 | **1.14× (−12.4%)** |
| YOLOv8n — ONNX Runtime (4 threads) | 68.16 ± 9.16 | 67.44 | 83.06 | 53.60 / 83.43 | 14.83 | 290.21 | 12.16 | 0.99× |
| **PatchCore — PyTorch CPU** | 211.14 ± 16.88 | **210.89** | **237.22** | 178.78 / 268.44 | 4.74 | 1,295.18 | 108.30 | 1.00× (mốc) |
| **PatchCore — ONNX Runtime (default threads)** | 114.58 ± 5.52 | **114.47** | **123.53** | 104.44 / 136.50 | **8.74** | **570.81** | 108.27 | **1.84× (−45.7%)** |
| PatchCore — ONNX Runtime (4 threads) | 144.69 ± 13.07 | 143.45 | 160.72 | 118.47 / 178.39 | 6.97 | 625.30 | 108.27 | 1.47× |

### 2.2. Toàn bộ Pipeline kiểm tra End-to-End (`YOLOv8n + PatchCore + Luật hợp nhất`)

| Cấu hình Pipeline End-to-End | Mean ± Std (ms) | **Latency p50 (ms)** | **Latency p95 (ms)** | **Throughput (FPS)** | **RSS RAM (MB)** | Tăng tốc p50 |
|---|---:|---:|---:|---:|---:|---:|
| **Trước tối ưu: PyTorch CPU E2E** (`YOLO PT + PatchCore PT`) | 290.08 ± 38.89 | **279.60** | **361.33** | 3.58 | 1,297.18 | 1.00× (mốc) |
| **Sau tối ưu (Chế độ 1): ONNX Runtime Full Dual-Branch** (luôn chạy cả 2 nhánh) | 233.62 ± 29.96 | **236.23** | **282.66** | 4.23 | **626.36** | **1.18× (p95 −21.8%, RAM −51.7%)** |
| **Sau tối ưu (Chế độ 2): ONNX Runtime + Early-Exit** (dừng sớm khi YOLO đã bắt được lỗi đã biết) | 70.71 ± 29.97 | **63.78** | **100.22** | **15.68** | **626.37** | **4.38× (p50 −77.2%, p95 −72.3%)** |

---

## 3. Kiểm chứng độ tương đương số học (Numerical Parity)

Trên `20` ảnh test của `NEU-DET`, đối chiếu trực tiếp đầu ra của `YOLOv8n (PyTorch)` và [`OnnxYoloDetector`](file:///e:/Vision/src/steelvision/models/detector.py) (tự cài đặt `letterbox`, chuẩn hóa `NCHW`, giải mã anchor `1×10×8400` và `NMS` thuần NumPy):

- **Tỷ lệ đồng thuận phán định (`verdict_agreement_rate` tại $\tau_{\text{det}}^* = 0.25$):** **`100.0%` (`20/20` ảnh)**
- **Sai số tuyệt đối trung bình của `max_confidence`:** **`1.0 × 10⁻⁶`**
- **Sai số tuyệt đối lớn nhất của `max_confidence`:** **`2.0 × 10⁻⁶`** (chỉ lệch ở chữ số thập phân thứ 6 do sai số dấu phẩy động FP32 giữa ATen và ONNX Runtime).

---

## 4. Phân tích kỹ thuật & Quyết định kiến trúc cho Dịch vụ (Phase 7)

1. **Giảm gần một nửa độ trễ của PatchCore (`210.89 ms → 114.47 ms`, nhanh gấp `1.84×`) và giảm 3× phương sai (`std 16.88 ms → 5.52 ms`):**
   - Trong PatchCore, phần nặng nhất là trích đặc trưng `WideResNet50-2` (`layer2`, `layer3`) và nhân ma trận khoảng cách Euclid với memory bank (`2.252 × 1.536`). Khi xuất sang đồ thị tĩnh ONNX (`batch=1`, `opset 18`) và bật `ORT_ENABLE_ALL`, ONNX Runtime hợp nhất (fuse) các lớp `Conv + BatchNorm + ReLU` và tối ưu phép tính khoảng cách bằng tập lệnh AVX2 trên toàn bộ 8 nhân vật lý của Ryzen 7 5800H.
2. **Tiết kiệm 52%–79% bộ nhớ RAM (`1,297 MB → 626 MB` cho cả 2 model; `241 MB` nếu chỉ chạy YOLO):**
   - Loại bỏ hoàn toàn runtime PyTorch + Lightning + Anomalib ở tầng suy luận giúp container Docker ở Phase 7 nhẹ hơn đáng kể và khởi động nhanh.
3. **Tối ưu kiến trúc "Short-Circuit / Early-Exit" cho Luật hợp nhất `OR`:**
   - Vì luật quyết định là phép tuyển logic $\text{NG} \iff (\text{det\_conf} \ge \tau_{\text{det}}) \lor (\text{anom\_score} \ge \tau_{\text{anom}})$, khi nhánh `OnnxYoloDetector` (`58.37 ms`) đã phát hiện lỗi quen thuộc vượt ngưỡng $\tau_{\text{det}}^* = 0.25$ (có sẵn tên lớp lỗi và tọa độ bounding box), kết quả hợp nhất **chắc chắn là `NG`**.
   - Trong chế độ `early_exit=True` (có thể bật/tắt qua tham số API ở Phase 7):
     - Với sản phẩm có lỗi quen thuộc: trả kết quả ngay sau **`~63.8 ms` (`15.7 FPS`, nhanh gấp `4.38×` so với PyTorch E2E `279.6 ms`)** mà **không làm thay đổi bất kỳ phán định OK/NG nào**.
     - Với sản phẩm không có lỗi quen thuộc (ứng viên `OK` hoặc nghi ngờ lỗi mới chưa từng gặp): tự động chạy tiếp nhánh `OnnxAnomalyDetector` (`~114.5 ms`) để kiểm tra an toàn trước khi cho qua.
     - Nếu người vận hành cần xem đồng thời cả bounding box lẫn bản đồ nhiệt anomaly (`anomaly_map`) cho mọi ảnh, chỉ cần đặt `always_run_anomaly=True` (`p50 = 236.23 ms`).
