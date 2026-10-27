# SteelVision — Hệ Thống Kiểm Tra Lỗi Bề Mặt Kim Loại (End-to-End QC)

**SteelVision** là hệ thống kiểm định chất lượng (Quality Control — QC) bề mặt kim loại hoàn chỉnh từ đầu đến cuối (end-to-end). Khi nhận một bức ảnh chụp bề mặt tấm thép hoặc linh kiện kim loại, hệ thống thực hiện kiểm tra chất lượng ảnh đầu vào, chạy song song hai nhánh mô hình (phát hiện lỗi có giám sát + phát hiện bất thường không giám sát), đưa ra phán quyết **OK (Đạt) / NG (Loại)** kèm vị trí lỗi và bản đồ nhiệt, đồng thời lưu vết toàn bộ lịch sử kiểm tra vào cơ sở dữ liệu SQLite.

---

## 1. Tổng Quan Dự Án và Bài Toán Thực Tế

Các mô hình học thuật trên tập dữ liệu lỗi bề mặt (như `NEU-DET`) thường dừng lại ở bài toán phân loại ảnh (Image Classification) và chỉ báo cáo độ chính xác (Accuracy). Khi áp dụng vào dây chuyền sản xuất thực tế, cách tiếp cận đó thất bại vì 5 lý do cốt lõi:

1. **Cần định vị chính xác vị trí và kích thước lỗi:** Kỹ thuật viên QC và cơ cấu phân loại cần biết vết lỗi nằm ở đâu và rộng bao nhiêu, không chỉ một nhãn tên lỗi cho cả bức ảnh.
2. **Mô hình có giám sát không định nghĩa được "OK là gì" và bỏ sót lỗi mới:** Một bộ phát hiện đối tượng (như YOLO) chỉ biết $K$ lớp lỗi đã có trong tập huấn luyện. Khi xuất hiện một loại dị tật mới chưa từng gặp (như lem hóa chất màu lạ hoặc dập ngược mặt linh kiện), mô hình cho điểm tự tin thấp và phán quyết nhầm sản phẩm hỏng là **OK**.
3. **Chi phí sai số bất đối xứng (Escape Rate so với Overkill Rate):** Để lọt một sản phẩm lỗi tới tay khách hàng (**Escape Rate** — tỷ lệ lọt lỗi) gây thiệt hại lớn hơn rất nhiều so với việc báo nhầm sản phẩm tốt cần kiểm tra lại (**Overkill Rate** — tỷ lệ báo giả). Ngưỡng quyết định phải được chọn trên tập `validation` theo mục tiêu QC thay vì dùng ngưỡng cảm tính `0.50`.
4. **Nhạy cảm với biến đổi quang học và khác miền dữ liệu:** Sự suy giảm cường độ đèn chiếu sáng, rung mờ camera, nhiễu cảm biến hoặc thay đổi khoảng cách chụp có thể làm trượt thang điểm của mô hình nếu không có cổng kiểm tra chất lượng ảnh chặn trước.
5. **Ràng buộc phần cứng công nghiệp (CPU-only):** Phần lớn trạm kiểm tra tại nhà máy chạy trên máy tính công nghiệp không có GPU rời, đòi hỏi tối ưu độ trễ suy luận đơn ảnh (`batch=1`) trên CPU và đóng gói thành dịch vụ độc lập có lưu vết.

SteelVision giải quyết trực tiếp cả 5 bài toán trên trong một kho mã nguồn tái lập hoàn chỉnh.

---

## 2. Kiến Trúc Hệ Thống

```text
                  Ảnh đầu vào (Camera công nghiệp / File ảnh)
                                     |
                                     v
           +---------------------------------------------------+
           |        Cổng kiểm tra chất lượng ảnh đầu vào       |
           |  (Độ sáng trung bình, Độ tương phản, Độ nét ảnh)  |
           +-------------------------+-------------------------+
                                     |
                      +--------------+--------------+
                      |                             |
                      v                             v
        +---------------------------+ +---------------------------+
        |   Nhánh 1: YOLOv8n ONNX   | |  Nhánh 2: PatchCore ONNX  |
        |  Phát hiện lỗi đã biết    | | Phát hiện bất thường (OK) |
        |  (Loại lỗi + Khung bao)   | | (Điểm lệch + Bản đồ nhiệt)|
        +-------------+-------------+ +-------------+-------------+
                      |                             |
                      +--------------+--------------+
                      |                             |
                      v
           +---------------------------------------------------+
           |            Luật hợp nhất quyết định OR            |
           |    NG nếu max_det_conf >= 0.25 HOẶC anom >= 0.53  |
           +-------------------------+-------------------------+
                                     |
                      +--------------+--------------+
                      |                             |
                      v                             v
        +---------------------------+ +---------------------------+
        |    Dịch vụ REST FastAPI   | |   Sổ lưu vết SQLite DB    |
        |  (/inspect, /inspections) | |  + Kho lưu trữ ảnh lỗi NG |
        +-------------+-------------+ +-------------+-------------+
                      |                             |
                      +--------------+--------------+
                                     |
                                     v
                    Bảng điều khiển QC (Streamlit UI)
```

### Luật Quyết Định Sản Phẩm (OR-Gate Decision Rule)

Gọi `max_det_conf` là độ tự tin cao nhất trong các khung bao lỗi do YOLOv8n dự đoán và `anom_score` là điểm bất thường cấp ảnh do PatchCore tính toán. Luật phán quyết sản phẩm là phép tuyển logic (`OR`):

```text
Verdict = NG  khi và chỉ khi  (max_det_conf >= tau_det) HOẶC (anom_score >= tau_anom)
```

Trong đó cặp ngưỡng vận hành chuẩn (`v1-val-2026-10-26`) được hiệu chuẩn trên tập `validation` là `tau_det = 0.25` và `tau_anom = 0.53`:

- **Định vị lỗi quen thuộc:** Khi `max_det_conf >= 0.25`, nhánh YOLOv8n trả về tên lớp lỗi và tọa độ khung bao (`bbox`).
- **Lưới an toàn bắt lỗi lạ:** Khi xuất hiện loại lỗi mới mà YOLOv8n chưa từng được học, nhánh PatchCore phát hiện độ lệch đặc trưng so với tập mẫu chuẩn không lỗi (`anom_score >= 0.53`) và chặn sản phẩm lại (`NG`).
- **Tối ưu ngắt sớm trên CPU (Early-Exit):** Vì luật hợp nhất là phép `HOẶC`, khi bật `early_exit=True` và nhánh YOLOv8n (`58.37 ms` trên CPU) đã phát hiện lỗi vượt ngưỡng `tau_det`, hệ thống kết luận ngay `NG (reason="detection")` và bỏ qua nhánh PatchCore (`114.47 ms`), giúp tăng tốc toàn tuyến gấp `4.38 lần` mà không làm thay đổi bất kỳ phán quyết `OK/NG` nào.

---

## 3. Dữ Liệu và Quy Trình Chống Rò Rỉ (Data Pipeline)

Toàn bộ các bộ dữ liệu công khai được xử lý qua pipeline tất định (`python -m steelvision.data.prepare --all`) thực hiện kiểm tra tọa độ nhãn, khử ảnh trùng lặp bằng **perceptual hash (`pHash`, khoảng cách Hamming `<= 4`)** trước khi chia tập để chống rò rỉ dữ liệu giữa `train` và `test`, sau đó chia phân tầng đa nhãn theo tỷ lệ `70 / 15 / 15` (`seed = 42`):

| Bộ dữ liệu | Vai trò | Số ảnh gốc | Số cặp trùng đã loại | Số ảnh sạch | Phân chia (`train / val / test`) | Số lớp |
|---|---|---:|---:|---:|---:|---|
| **NEU-DET** | Huấn luyện và đánh giá phát hiện lỗi thép cán nóng | 1,800 | 1 | 1,799 | `1,259 / 270 / 270` | 6 lớp (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`) |
| **GC10-DET** | Đánh giá đa lớp và kiểm thử chéo miền (cross-domain) | 2,294 | 8 (loại 35 nhãn lỗi tọa độ) | 2,286 | `1,600 / 343 / 343` | 10 lớp (`crease`, `crescent_gap`, `inclusion`, `oil_spot`, `punching_hole`, `rolled_pit`, `silk_spot`, `waist_folding`, `water_spot`, `welding_line`) |
| **MVTec AD (`metal_nut`)** | Huấn luyện và đánh giá phát hiện bất thường (OK/NG) | 335 | 0 | 335 | `220` (`train/good`) / `115` (`test`: 22 OK + 93 NG) | `good` + 4 nhóm lỗi (`bent`, `color`, `flip`, `scratch`) |

Ngoài ra, dự án cung cấp quy trình thu thập dữ liệu tự chụp dưới 3 điều kiện ánh sáng `A/B/C` ([`docs/capture-protocol.vi.md`](docs/capture-protocol.vi.md)), hướng dẫn gán nhãn ([`docs/labeling-guide.vi.md`](docs/labeling-guide.vi.md)) và công cụ kiểm định tự động [`src/steelvision/data/self_captured.py`](src/steelvision/data/self_captured.py) (kiểm tra tỷ lệ ảnh `OK >= 40%`, gom nhóm theo vị trí vật lý và đo độ nhất quán khi gán nhãn lại 10% với yêu cầu `mean IoU >= 0.75`, `image F1 >= 0.90`).

---

## 4. Kết Quả Thực Nghiệm

Chi tiết toàn bộ bảng số liệu và phân tích lỗi được lưu tại [`docs/reports/experiments.vi.md`](docs/reports/experiments.vi.md), [`docs/reports/qc_robustness.vi.md`](docs/reports/qc_robustness.vi.md) và [`docs/reports/performance.vi.md`](docs/reports/performance.vi.md).

### 4.1 Phát hiện lỗi có giám sát (YOLOv8) và Nghiên cứu Ablation (Tập Test)

Huấn luyện 80 epochs (`seed = 42`) trên GPU NVIDIA Tesla T4, mỗi lần chỉ thay đổi một yếu tố:

| Thí nghiệm | Bộ dữ liệu | Mô hình | Kích thước ảnh | Tăng cường Mosaic | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | NEU-DET | YOLOv8n | 640 | `1.0` | 0.6778 | 0.7280 | 0.7641 | 0.4272 |
| **Ablation 2 (Được chọn)** | **NEU-DET** | **YOLOv8n** | **640** | **`0.0` (Tắt)** | **0.6967** | **0.7429** | **0.7836** | **0.4404** |
| Ablation 1 (Tăng kích cỡ mạng) | NEU-DET | YOLOv8s | 640 | `1.0` | 0.7082 | 0.7007 | 0.7662 | 0.4314 |
| Ablation 3 (Tăng độ phân giải) | NEU-DET | YOLOv8n | 800 | `0.0` | 0.7001 | 0.7002 | 0.7510 | 0.4193 |
| Baseline 10 lớp | GC10-DET | YOLOv8n | 640 | `1.0` | 0.6808 | 0.6352 | **0.6793** | **0.3512** |

**Kết luận từ Ablation:** Việc tắt phép ghép ảnh `Mosaic` (`mosaic=0.0`) trên bề mặt thép `NEU-DET` giúp tăng `mAP@50` trên tập `test` từ `0.7641` lên **`0.7836` (`+1.95%`)** và kéo lớp lỗi khó nhất là nứt chân chim (`crazing`) từ `0.388` lên **`0.491` (`+10.3%`)**, do các đường biên cắt ghép nhân tạo của Mosaic dễ gây nhầm lẫn với vết nứt/xước trên nền kim loại.

### 4.2 Phát hiện bất thường không giám sát (PatchCore trên MVTec AD `metal_nut`)

Chỉ huấn luyện trên `220` ảnh đạt chuẩn không lỗi (`train/good`), sử dụng mạng khung `wide_resnet50_2` (`layer2`, `layer3`), tỷ lệ lấy mẫu coreset `1%`, kích thước đầu vào `256x256`:

| Chỉ số đánh giá | Kết quả trên tập Test |
|---|---:|
| **Image-level AUROC** | **0.9980 (99.80%)** |
| **Image-level F1 Score** | **0.9840 (98.40%)** |
| **Pixel-level AUROC** | **0.9869 (98.69%)** |
| **Pixel-level F1 Score** | **0.8445 (84.45%)** |

### 4.3 Đánh giá cấp sản phẩm theo chỉ số QC (Mục tiêu `Escape Rate <= 2.0%`)

Ngưỡng vận hành được chọn trên tập `val` để thỏa mãn ràng buộc `escape_rate <= 2.0%`, sau đó cố định và đo một lần trên tập `test`:

| Nhánh / Bộ dữ liệu | Ngưỡng chọn từ `val` | Test Escape Rate (Lọt lỗi) | Test Overkill Rate (Báo giả) | Test Precision | Test Recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| **PatchCore (`MVTec metal_nut`)** | `tau_anom = 0.53` | **0.00%** (`0/47`) | **0.00%** (`0/11` test, `4.55%` toàn tập) | **100.00%** | **100.00%** | **1.0000** |
| **YOLOv8n (`NEU-DET` + nền OK)** | `tau_det = 0.25` | **0.75%** (`2/266`) | **22.50%** (`18/80`) | **93.62%** | **99.25%** | **0.9635** |
| *YOLOv8n ngưỡng mặc định* | `tau_det = 0.50` | *9.40%* (`25/266`) | *2.50%* (`2/80`) | *99.18%* | *90.60%* | *0.9469* |
| **Hợp nhất 2 nhánh (Test lỗi mới)** | `tau_det = 0.25, tau_anom = 0.53` | **0.00%** (trên cả 4 lớp lỗi bị giấu) | **4.55%** | **98.94%** | **100.00%** | **0.9947** |

**Thí nghiệm giả lập lỗi mới chưa từng gặp (Leave-One-Class-Out):** Khi lần lượt ẩn từng lớp lỗi để giả lập loại dị tật mới phát sinh trên dây chuyền, nếu chỉ dùng riêng bộ phát hiện có giám sát thì tỷ lệ lọt lỗi vọt lên `14.66%–17.29%` trên `NEU-DET` và `23.66%–26.88%` trên `metal_nut`. Khi kết hợp thêm nhánh PatchCore theo luật `OR`, tỷ lệ lọt lỗi trên `metal_nut` giảm về **`0.00%`** ở cả 4 kịch bản.

### 4.4 Hiệu năng suy luận và bộ nhớ trên CPU (`AMD Ryzen 7 5800H`, `batch=1`)

Đo với 10 lượt khởi động (warm-up) và 50 lượt chạy tính giờ liên tiếp:

| Thành phần / Chế độ | Backend | Latency p50 (ms) | Latency p95 (ms) | Trung bình (ms) | FPS | RSS RAM (MB) | Tăng tốc so với PyTorch |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n (`640x640`) | PyTorch CPU | 66.65 | 72.39 | 66.22 | 15.00 | 1,166.22 | `1.00x` |
| YOLOv8n (`640x640`) | ONNX Runtime CPU | **58.37** | **67.60** | **58.65** | **17.13** | **241.38** | **`1.14x`** |
| PatchCore (`256x256`) | PyTorch CPU | 210.89 | 237.22 | 213.71 | 4.74 | 1,295.18 | `1.00x` |
| PatchCore (`256x256`) | ONNX Runtime CPU | **114.47** | **123.53** | **114.33** | **8.74** | **570.81** | **`1.84x`** |
| Toàn tuyến (Full Dual-Branch) | PyTorch CPU | 279.60 | 361.33 | 290.89 | 3.58 | 1,297.18 | `1.00x` |
| Toàn tuyến (Full Dual-Branch) | ONNX Runtime CPU | 236.23 | 282.66 | 240.31 | 4.23 | 626.36 | `1.18x` |
| **Toàn tuyến (Early-Exit)** | **ONNX Runtime CPU** | **63.78** | **100.22** | **73.32** | **15.68** | **626.37** | **`4.38x`** |

---

## 5. Cấu Trúc Thư Mục

```text
SteelVision/
├── assets/samples/                  # 11 ảnh mẫu chuẩn (NEU-DET + MVTec) tích hợp sẵn cho giao diện
├── configs/
│   ├── datasets.yaml                # Cấu hình đường dẫn và lớp lỗi của các bộ dữ liệu
│   ├── eval.yaml                    # Cấu hình đánh giá QC và kiểm thử độ bền (Phase 4)
│   └── benchmark.yaml               # Cấu hình đo độ trễ và RAM trên CPU (Phase 6)
├── docs/
│   ├── dataset-card.md              # Nguồn gốc, giấy phép, khử trùng lặp pHash và thống kê chia tập
│   ├── model-card.md                # Kiến trúc mô hình, chỉ số thực nghiệm, ngưỡng vận hành và hạn chế
│   ├── lessons-learned.vi.md        # Bài viết đúc kết 6 bài học kỹ thuật trong dự án
│   ├── capture-protocol.vi.md       # Giao thức thu thập bộ dữ liệu tự chụp trong 3 điều kiện sáng
│   ├── labeling-guide.vi.md         # Quy chuẩn gán nhãn khung bao và kiểm định chất lượng nhãn
│   ├── journal.vi.md                # Nhật ký kỹ thuật theo từng giai đoạn
│   └── reports/                     # Báo cáo chi tiết thực nghiệm, QC/độ bền và tối ưu hiệu năng CPU
├── notebooks/
│   ├── 00_eda_public_datasets.ipynb # Khảo sát dữ liệu (EDA) trên NEU-DET và GC10-DET
│   ├── 01_train_yolo_baseline.ipynb # Notebook huấn luyện YOLOv8 và chạy Ablation trên GPU
│   └── 02_train_patchcore.ipynb     # Notebook huấn luyện PatchCore và xuất ONNX
├── scripts/
│   ├── build_kernel.py              # Đóng gói mã nguồn thành notebook độc lập đẩy lên Kaggle GPU
│   └── generate_demo_video.py       # Tạo video demo 60 giây (runs/demo_steelvision.mp4)
├── src/steelvision/
│   ├── api/                         # Dịch vụ FastAPI (app.py), Quality Gate (quality.py), SQLite (store.py)
│   ├── dashboard/                   # Giao diện trạm kiểm định QC bằng Streamlit (app.py)
│   ├── data/                        # Tải dữ liệu, chuyển đổi VOC sang YOLO, lọc trùng pHash, QC tự chụp
│   ├── decision/                    # Luật hợp nhất quyết định OR-Gate (fusion.py)
│   ├── eval/                        # Chỉ số QC (escape/overkill), nhiễu quang học, chéo miền và benchmark
│   └── models/                      # Bộ suy luận ONNX Runtime cho YOLOv8 (detector.py) và PatchCore (anomaly.py)
├── tests/                           # 39 bài kiểm thử đơn vị (unit tests) tự động
├── Dockerfile                       # Ảnh container Python 3.11 chạy suy luận CPU
├── docker-compose.yml               # Khởi chạy đồng thời FastAPI (:8000) và Streamlit (:8501)
├── Makefile                         # Các lệnh tự động hóa chuẩn
└── pyproject.toml                   # Khai báo gói, phụ thuộc, cấu hình ruff và pytest
```

---

## 6. Hướng Dẫn Cài Đặt và Chạy Hệ Thống

### Cài đặt trên máy cá nhân (Local)

```bash
python -m venv .venv
.venv\Scripts\activate          # Trên Windows (dùng `source .venv/bin/activate` trên Linux/macOS)
pip install -e ".[dev]"
```

### Các lệnh vận hành chuẩn

```bash
make data        # Tải, khử trùng lặp pHash và chia tập NEU-DET, GC10-DET, MVTec AD
make patchcore   # Huấn luyện PatchCore trên MVTec metal_nut và xuất mô hình ONNX
make eval        # Chạy đánh giá QC (escape/overkill), giả lập lỗi mới và kiểm thử độ bền
make benchmark   # Đo độ trễ CPU (p50/p95), FPS và dung lượng RAM (PyTorch vs ONNX)
make serve       # Khởi chạy dịch vụ FastAPI tại http://localhost:8000 (Swagger UI tại /docs)
make dashboard   # Khởi chạy giao diện Streamlit QC tại http://localhost:8501 (có sẵn 11 ảnh mẫu)
make demo        # Tạo video demo MP4 dài 60 giây tại runs/demo_steelvision.mp4
make test        # Chạy bộ 39 unit tests bằng pytest
make lint        # Kiểm tra chất lượng mã nguồn bằng ruff
```

### Triển khai bằng Docker Compose

```bash
docker compose up --build
```

- **FastAPI Swagger UI:** `http://localhost:8000/docs` (`POST /inspect`, `GET /inspections`, `GET /inspections/{id}`, `GET /stats`, `GET /health`)
- **Bảng điều khiển Streamlit QC:** `http://localhost:8501`

---

## 7. Hạn Chế Kỹ Thuật (Limitations)

1. **Nhầm lẫn giữa vết nứt siêu mảnh và vân bề mặt thép (`crazing`):** Ở độ phân giải gốc `200x200` của `NEU-DET`, lớp vết nứt chân chim (`crazing`) đạt `mAP@50 = 0.491` do độ tương phản cục bộ rất thấp so với vân thép cán nóng. Để bắt các vết nứt dưới milimet cần ống kính độ phóng đại cao hơn hoặc suy luận cắt ô (SAHI).
2. **Trượt ngưỡng khi nguồn sáng suy giảm mạnh (`brightness_down`):** Mặc dù PatchCore giữ nguyên khả năng xếp hạng (`image_AUROC >= 0.9785`) dưới cả 6 phép nhiễu quang học, việc giảm 40% độ sáng làm dịch chuyển khoảng cách đặc trưng và đẩy tỷ lệ báo giả (`overkill_rate`) lên cao nếu giữ ngưỡng cố định. Cổng kiểm tra chất lượng ảnh (`check_image_quality`) đã khắc phục bằng cách phát hiện và cảnh báo sớm các khung hình quá tối (`too_dark`), quá chói (`too_bright`), mất tương phản (`low_contrast`) hoặc bị mờ (`blurry`) trước khi suy luận, tuy nhiên việc duy trì nguồn sáng đèn LED ổn định vẫn rất quan trọng.
3. **Chênh lệch tỷ lệ trường nhìn khi chuyển miền (`NEU-DET -> GC10-DET`):** Khi mang mô hình huấn luyện trên ảnh cận cảnh `200x200` (`NEU-DET`) sang chạy trực tiếp trên ảnh chụp toàn khổ `2048x1000` (`GC10-DET`), chỉ số `AP@50` của lớp `inclusion` giảm xuống `0.0686` do chênh lệch kích thước vật lý gấp 10 lần. Khi lắp đặt sang trạm camera mới, cần chuẩn hóa tỷ lệ `mm/pixel` hoặc tinh chỉnh lại mô hình trên mẫu ảnh tại dây chuyền.

---

## 8. Danh Mục Tài Liệu

- [`SteelVision_Project_Overview.md`](SteelVision_Project_Overview.md) — Báo cáo tổng quan toàn diện dự án và bản tóm tắt CV
- [`docs/model-card.md`](docs/model-card.md) — Hồ sơ mô hình (Model Card)
- [`docs/dataset-card.md`](docs/dataset-card.md) — Hồ sơ dữ liệu (Dataset Card)
- [`docs/lessons-learned.vi.md`](docs/lessons-learned.vi.md) — 6 bài học kỹ thuật cốt lõi rút ra từ dự án
- [`docs/reports/experiments.vi.md`](docs/reports/experiments.vi.md) — Báo cáo thực nghiệm YOLOv8, Ablation và PatchCore
- [`docs/reports/qc_robustness.vi.md`](docs/reports/qc_robustness.vi.md) — Báo cáo đánh giá QC (`escape`/`overkill`), lỗi mới và độ bền
- [`docs/reports/performance.vi.md`](docs/reports/performance.vi.md) — Báo cáo đo kiểm độ trễ CPU (`p50`/`p95`), `FPS` và `RAM`
- [`docs/capture-protocol.vi.md`](docs/capture-protocol.vi.md) và [`docs/labeling-guide.vi.md`](docs/labeling-guide.vi.md) — Giao thức chụp ảnh và hướng dẫn gán nhãn dữ liệu tự thu thập
- [`docs/journal.vi.md`](docs/journal.vi.md) — Nhật ký quá trình phát triển theo từng giai đoạn

## Giấy Phép (License)

Mã nguồn dự án được phát hành theo giấy phép MIT. Các bộ dữ liệu công khai (`NEU-DET`, `GC10-DET`, `MVTec AD`) giữ nguyên giấy phép gốc của tác giả được ghi chi tiết trong [`docs/dataset-card.md`](docs/dataset-card.md).
