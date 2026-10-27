# Tổng Quan Dự Án SteelVision — Hệ Thống Kiểm Tra Lỗi Bề Mặt Kim Loại (End-to-End QC)

Tài liệu này tổng hợp toàn diện dự án **SteelVision**: từ bài toán thực tế trong nhà máy, kiến trúc kỹ thuật, quy trình xử lý dữ liệu, kết quả thực nghiệm qua 7 giai đoạn (Phase 0 – Phase 7), cho đến giải pháp tối ưu hóa CPU và đóng gói dịch vụ triển khai.

---

## 1. Tóm Tắt Dự Án (Executive Summary)

**SteelVision** là hệ thống kiểm tra chất lượng (Quality Control — QC) bề mặt kim loại hoàn chỉnh từ đầu đến cuối (end-to-end). Khi nhận một bức ảnh chụp bề mặt tấm thép hoặc linh kiện kim loại từ camera công nghiệp, hệ thống thực hiện:

1. **Kiểm tra chất lượng ảnh đầu vào (Pre-Inference Image Quality Gate):** Đánh giá độ sáng, độ tương phản và độ nét (Laplacian variance) trong `< 1 ms` để phát hiện sớm lỗi sụt áp đèn LED hoặc lệch nét camera.
2. **Suy luận song song 2 nhánh (Dual-Branch Inference):**
   - **Nhánh 1 — Phát hiện lỗi có giám sát (`YOLOv8n` ONNX):** Khoanh vùng tọa độ (bounding box) và phân loại các lỗi bề mặt đã biết.
   - **Nhánh 2 — Phát hiện bất thường không giám sát (`PatchCore` ONNX):** Chỉ học từ ảnh sản phẩm tốt (`OK`), đóng vai trò "lưới an toàn" định nghĩa thế nào là bề mặt chuẩn và bắt giữ mọi loại dị tật mới chưa từng xuất hiện trong tập nhãn.
3. **Phán quyết cấp sản phẩm theo chuẩn QC (`OR-Gate Fusion`):** Ra quyết định **OK (Đạt)** hoặc **NG (Loại)** dựa trên cặp ngưỡng vận hành (	au_det = 0.25, tau_anom = 0.53) được hiệu chuẩn trên tập `validation` nhằm kiểm soát tỷ lệ lọt lỗi (**escape rate $\le 2\%$**).
4. **Lưu vết và quản trị trực quan:** Phục vụ qua REST API (`FastAPI /inspect`), tự động lưu toàn bộ lịch sử kiểm tra vào cơ sở dữ liệu `SQLite`, lưu trữ ảnh bằng chứng đối với sản phẩm `NG`, và hiển thị trên bảng điều khiển `Streamlit` (tích hợp sẵn 11 ảnh mẫu để kiểm thử tức thì).

---

## 2. Bài Toán Thực Tế & Lý Do Cách Tiếp Cận Truyền Thống Thất Bại

Các đồ án thị giác máy tính thông thường trên tập dữ liệu lỗi thép (như `NEU-DET`) thường dừng lại ở bài toán phân loại ảnh (Image Classification) và báo cáo độ chính xác 98–99%. Tuy nhiên, khi đưa ra dây chuyền sản xuất thực tế, cách làm đó bộc lộ 5 khoảng trống lớn:

| Vấn đề trên dây chuyền sản xuất | Hạn chế của mô hình phân loại / YOLO đơn lẻ | Giải pháp của SteelVision |
|---|---|---|
| **1. Cần biết vị trí và kích thước vết lỗi** | Mô hình phân loại chỉ trả về nhãn cho cả bức ảnh, không chỉ ra lỗi nằm ở đâu. | Sử dụng **YOLOv8n** (`640x640`) định vị chính xác tọa độ hộp bao và bản đồ nhiệt **PatchCore** (`256x256`) ở mức pixel. |
| **2. Câu hỏi "OK là gì?" và lỗi mới chưa từng gặp** | Bộ phát hiện có giám sát chỉ biết $K$ lớp lỗi đã học. Khi gặp lỗi lạ (lem màu, dập ngược), nó cho điểm tin cậy thấp và báo nhầm là **OK**. | Ghép song song **PatchCore** (chỉ học từ ảnh `OK`) theo luật **OR-Gate**, kéo tỷ lệ lọt lỗi mới từ `23.7%–26.9%` về **`0.00%`**. |
| **3. Chi phí sai số bất đối xứng (Lọt lỗi vs. Báo giả)** | Dùng ngưỡng mặc định `0.50` khiến tỷ lệ lọt hàng hỏng (**escape rate**) trên `NEU-DET` lên tới `9.40%` (`25/266` sản phẩm lỗi bị lọt). | Hiệu chuẩn ngưỡng trên tập `val` theo mục tiêu QC (`escape <= 2%`), kéo **escape rate trên tập `test` xuống `0.75%`** (giảm 12.5 lần). |
| **4. Nhạy cảm với thay đổi ánh sáng và môi trường** | Khi đèn chiếu sáng yếu đi 40% (`brightness_down`), khoảng cách đặc trưng bị trượt làm tăng tỷ lệ báo giả (**overkill**). | Đặt **Image Quality Gate (`check_image_quality`)** ngay trước mô hình để cảnh báo tức thì các khung hình quá tối, quá chói, mờ hoặc mất tương phản. |
| **5. Hạn chế phần cứng tại trạm kiểm tra (CPU-only)** | Chạy cả YOLO và PatchCore trên PyTorch CPU tốn `279.60 ms/ảnh` và ngốn `1.3 GB RAM`. | Xuất sang **ONNX Runtime** kết hợp cơ chế **Early-Exit**, giảm độ trễ trung vị `p50` xuống **`63.78 ms` (`4.38x` nhanh hơn)** và giảm **51.7% RAM**. |

---

## 3. Quy Trình Xử Lý Dữ Liệu & Chống Rò Rỉ (Phase 0 & Phase 1)

Hệ thống sử dụng 3 bộ dữ liệu công khai chuẩn công nghiệp kết hợp với bộ công cụ kiểm định dữ liệu tự chụp:

1. **Khử trùng lặp bằng Perceptual Hash (`pHash`):** Trước khi chia tập `train / val / test`, toàn bộ ảnh được băm cảm nhận (`pHash`, ngưỡng khoảng cách Hamming $\le 4$) và gom cụm bằng cấu trúc Disjoint-Set (Union-Find) để đảm bảo các ảnh gần giống nhau không bao giờ nằm ở hai tập khác nhau:
   - **NEU-DET (6 lớp lỗi thép cán nóng):** Từ `1,800` ảnh gốc, phát hiện và loại bỏ `1` cặp ảnh trùng, còn **`1,799` ảnh sạch** (`1,259 train / 270 val / 270 test`).
   - **GC10-DET (10 lớp lỗi bề mặt thép tấm):** Từ `2,294` ảnh gốc, phát hiện và loại bỏ `8` cặp ảnh trùng và `35` hộp bao lỗi tọa độ, còn **`2,286` ảnh sạch** (`1,600 train / 343 val / 343 test`).
   - **MVTec AD (`metal_nut` — Linh kiện đai ốc kim loại):** Gồm **`335` ảnh** (`220` ảnh `train/good` không lỗi để huấn luyện PatchCore; `115` ảnh `test` gồm `22` ảnh `OK` và `93` ảnh `NG` thuộc 4 nhóm lỗi `bent`, `color`, `flip`, `scratch`).
2. **Chia tập phân tầng đa nhãn (Multi-label Stratified Split):** Đảm bảo tỷ lệ các lớp lỗi giữa `train`, `val` và `test` đồng đều (`70% / 15% / 15%`, cố định `seed = 42`).
3. **Giao thức dữ liệu tự chụp (Phase 5 Tooling):** Xây dựng đầy đủ quy trình thu thập dưới 3 chế độ chiếu sáng `A/B/C` ([`docs/capture-protocol.vi.md`](docs/capture-protocol.vi.md)), quy chuẩn gán nhãn ([`docs/labeling-guide.vi.md`](docs/labeling-guide.vi.md)) và module kiểm định tự động [`src/steelvision/data/self_captured.py`](src/steelvision/data/self_captured.py) (đảm bảo tỷ lệ ảnh `OK >= 40%`, chia tập theo nhóm `(panel, position)` và kiểm tra độ nhất quán nhãn khi gán lại 10% với `mean IoU >= 0.75`).

---

## 4. Kết Quả Thực Nghiệm & Nghiên Cứu Ablation (Phase 2 & Phase 3)

### 4.1 Phát hiện lỗi có giám sát (YOLOv8) và Ablation Study

Huấn luyện 80 epochs (`seed = 42`) trên GPU NVIDIA Tesla T4, mỗi thí nghiệm chỉ thay đổi đúng một yếu tố:

| Thí nghiệm | Tập dữ liệu | Mô hình | Độ phân giải | Augmentation | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---:|---|---:|---:|---:|---:|
| Baseline | NEU-DET | YOLOv8n | 640 | Mặc định (`mosaic=1.0`) | 0.6778 | 0.7280 | 0.7641 | 0.4272 |
| **Ablation 2 (Được chọn)** | **NEU-DET** | **YOLOv8n** | **640** | **Tắt Mosaic (`mosaic=0.0`)** | **0.6967** | **0.7429** | **0.7836** | **0.4404** |
| Ablation 1 (Tăng kích cỡ mạng) | NEU-DET | YOLOv8s | 640 | Mặc định (`mosaic=1.0`) | 0.7082 | 0.7007 | 0.7662 | 0.4314 |
| Ablation 3 (Tăng độ phân giải) | NEU-DET | YOLOv8n | 800 | Tắt Mosaic (`mosaic=0.0`) | 0.7001 | 0.7002 | 0.7510 | 0.4193 |
| Baseline 10 lớp | GC10-DET | YOLOv8n | 640 | Mặc định (`mosaic=1.0`) | 0.6808 | 0.6352 | **0.6793** | **0.3512** |

**Phát hiện kỹ thuật cốt lõi:** Trên ảnh công nghiệp bề mặt kim loại đồng nhất, phép ghép 4 ảnh `Mosaic` tạo ra các đường biên cắt ghép giả tạo rất giống vết nứt hoặc vết xước. Khi tắt `Mosaic` (`mosaic=0.0`), chỉ số `mAP@50` trên tập `test` tăng từ `0.7641` lên **`0.7836` (`+1.95%`)**, đặc biệt lớp khó nhất là vết nứt chân chim (`crazing`) tăng vọt từ `0.388` lên **`0.491` (`+10.3%`)**.

### 4.2 Phát hiện bất thường không giám sát (PatchCore trên MVTec `metal_nut`)

Sử dụng backbone `wide_resnet50_2` trích xuất đặc trưng tại `layer2` và `layer3`, lấy mẫu coreset `1%` (`coreset_sampling_ratio = 0.01`) chỉ từ `220` ảnh `train/good`:

- **Image-level AUROC:** **`99.80%` (`0.9980`)** | **Image-level F1:** **`98.40%`**
- **Pixel-level AUROC:** **`98.69%` (`0.9869`)** | **Pixel-level F1:** **`84.45%`**

---

## 5. Đánh Giá Theo Ngôn Ngữ QC & Kiểm Thử Độ Bền (Phase 4)

### 5.1 Hiệu chuẩn ngưỡng theo mục tiêu `Escape Rate <= 2.0%`

Thay vì dùng ngưỡng `0.50` cảm tính, hệ thống quét đường cong đánh đổi trên tập `val` để tìm ngưỡng vận hành thỏa mãn `escape_rate <= 2.0%` rồi khóa lại để đo một lần duy nhất trên tập `test`:

| Nhánh / Bộ dữ liệu | Ngưỡng vận hành (chọn từ `val`) | Escape Rate trên `test` (Lọt lỗi) | Overkill Rate trên `test` (Báo giả) | Precision (`test`) | Recall (`test`) | F1 (`test`) |
|---|---:|---:|---:|---:|---:|---:|
| **PatchCore (`MVTec metal_nut`)** | 	au_anom = 0.53 | **0.00%** (`0/47`) | **0.00%** (`0/11` test, `4.55%` toàn tập) | **100.00%** | **100.00%** | **1.0000** |
| **YOLOv8n (`NEU-DET` + nền OK)** | 	au_det = 0.25 | **0.75%** (`2/266`) | **22.50%** (`18/80`) | **93.62%** | **99.25%** | **0.9635** |
| *YOLOv8n ngưỡng mặc định* | 	au_det = 0.50 | *9.40%* (`25/266`) | *2.50%* (`2/80`) | *99.18%* | *90.60%* | *0.9469* |
| **Hợp nhất 2 nhánh (Test lỗi mới)** | 	au_det = 0.25, tau_anom = 0.53 | **0.00%** (trên cả 4 nhóm lỗi ẩn) | **4.55%** | **98.94%** | **100.00%** | **0.9947** |

### 5.2 Giả lập xuất hiện "Lỗi Mới Chưa Từng Gặp" (Leave-One-Class-Out)

Khi lần lượt giấu từng lớp lỗi đi để đóng vai "loại lỗi mới phát sinh trên dây chuyền":
- Nếu chỉ dùng nhánh detector đơn lẻ: tỷ lệ lọt lỗi (**escape rate**) tăng vọt lên **`14.66%–17.29%`** trên `NEU-DET` và **`23.66%–26.88%`** trên `metal_nut`.
- Khi bật hợp nhất **OR-Gate (`YOLOv8 + PatchCore`)**: nhánh PatchCore bắt giữ toàn bộ các mẫu thuộc lớp lỗi bị ẩn, đưa **escape rate về đúng `0.00%`** ở cả 4 kịch bản.

### 5.3 Kiểm thử độ bền (6 phép nhiễu quang học) & Khác miền (`NEU-DET -> GC10-DET`)

- **Độ bền quang học:** Dưới 6 phép biến đổi (`brightness_up`, `brightness_down`, `contrast_low`, `gaussian_blur`, `gaussian_noise`, `rotate_5deg`), YOLOv8n giữ độ bền rất tốt trước thay đổi độ sáng/tương phản nhưng nhạy cảm với nhiễu hạt và mất nét. Trong khi đó, PatchCore giữ vững `image_AUROC >= 97.85%` ở mọi phép nhiễu nhưng bị dịch thang khoảng cách khi ảnh tối đi 40% (`brightness_down`), gây tăng overkill — đây chính là cơ sở thực nghiệm để thiết kế **Image Quality Gate** ở Phase 7.
- **Khác miền dữ liệu (`NEU-DET -> GC10-DET`):** Đánh giá chéo lớp `inclusion` từ ảnh chụp cận cảnh `200x200` (`NEU-DET`) sang ảnh toàn khổ `2048x1000` (`GC10-DET`) làm `AP@50` giảm xuống `0.0686` do chênh lệch tỷ lệ trường nhìn gấp 10 lần, khẳng định việc triển khai thực tế luôn cần chuẩn hóa tỷ lệ `mm/pixel` hoặc tinh chỉnh trên ảnh tại trạm.

---

## 6. Tối Ưu Hóa Suy Luận CPU & Benchmark Hiệu Năng (Phase 6)

Đo đạc thực tế trên vi xử lý `AMD Ryzen 7 5800H` (`batch=1`, 10 vòng khởi động + 50 vòng đo chính thức):

| Thành phần / Chế độ | Backend | Latency p50 (ms) | Latency p95 (ms) | FPS | RSS RAM (MB) | Tăng tốc so với PyTorch |
|---|---|---:|---:|---:|---:|---:|
| YOLOv8n (`640x640`) | PyTorch CPU | 66.65 | 72.39 | 15.00 | 1,166.22 | `1.00x` |
| YOLOv8n (`640x640`) | ONNX Runtime CPU | **58.37** | **67.60** | **17.13** | **241.38** | **`1.14x`** (RAM `-79.3%`) |
| PatchCore (`256x256`) | PyTorch CPU | 210.89 | 237.22 | 4.74 | 1,295.18 | `1.00x` |
| PatchCore (`256x256`) | ONNX Runtime CPU | **114.47** | **123.53** | **8.74** | **570.81** | **`1.84x`** (`p95 -47.9%`) |
| Toàn tuyến (Full Dual-Branch) | PyTorch CPU | 279.60 | 361.33 | 3.58 | 1,297.18 | `1.00x` |
| Toàn tuyến (Full Dual-Branch) | ONNX Runtime CPU | 236.23 | 282.66 | 4.23 | 626.36 | `1.18x` (RAM `-51.7%`) |
| **Toàn tuyến (Early-Exit OR-Gate)** | **ONNX Runtime CPU** | **63.78** | **100.22** | **15.68** | **626.37** | **`4.38x`** |

---

## 7. Đóng Gói Dịch Vụ, Lưu Vết SQLite & Giao Diện Quản Trị (Phase 7)

1. **Dịch vụ REST API (`src/steelvision/api/app.py`):**
   - `GET /health`: Kiểm tra trạng thái nạp mô hình ONNX, phiên bản ngưỡng và kết nối cơ sở dữ liệu.
   - `POST /inspect`: Nhận file ảnh, chạy `check_image_quality`, thực thi `OnnxYoloDetector` + `OnnxAnomalyDetector` + `decide`, lưu ảnh bằng chứng nếu `NG` và ghi bản ghi vào SQLite.
   - `GET /inspections`, `GET /inspections/{id}`, `GET /stats`: Truy vấn lịch sử kiểm tra và thống kê tỷ lệ lỗi theo thời gian thực.
2. **Giao diện Trạm QC (`src/steelvision/dashboard/app.py`):**
   - Tích hợp sẵn **11 ảnh mẫu chuẩn** trong `assets/samples/` (5 mẫu đai ốc `MVTec` gồm cả `OK` và `NG`, 6 mẫu lỗi thép `NEU-DET`) giúp kiểm thử trực tiếp chỉ với 1 cú nhấp chuột mà không cần chuẩn bị ảnh ngoài.
   - Hiển thị trực quan 3 khung hình song song: Ảnh gốc, Khung bao lỗi YOLOv8n và Bản đồ nhiệt PatchCore, kèm bảng lọc lịch sử SQLite và kho ảnh bằng chứng `NG`.
3. **Đóng gói Docker & Kiểm thử tự động:**
   - Cung cấp `Dockerfile`, `docker-compose.yml` khởi chạy đồng thời cả API (`:8000`) và Dashboard (`:8501`).
   - Toàn bộ hệ thống được bảo vệ bởi **39 unit tests** (`pytest`) và kiểm tra quy chuẩn mã nguồn bằng `ruff`.

---

## 8. Nội Dung Tóm Tắt Chuẩn Hóa Cho Hồ Sơ Năng Lực (CV)

> **SteelVision – Hệ thống kiểm tra lỗi bề mặt kim loại** | Python, PyTorch, YOLOv8, Anomalib (PatchCore), ONNX Runtime, FastAPI, Docker
> - Xây dựng pipeline kết hợp Object Detection (`YOLOv8n`) và Unsupervised Anomaly Detection (`PatchCore`) kèm cổng kiểm tra chất lượng ảnh (Image Quality Gate) quyết định **OK/NG** trên `4,420` ảnh công nghiệp (`NEU-DET`, `GC10-DET`, `MVTec AD metal_nut`) sau khử trùng lặp `pHash`; thiết kế giao thức thu thập & kiểm định nhất quán nhãn cho bộ dữ liệu tự chụp trong `3` điều kiện ánh sáng.
> - Đạt **mAP@50 = `0.784`** trên `NEU-DET` (`+1.95%` nhờ tắt Mosaic), **Image AUROC = `99.80%`** / **Pixel AUROC = `98.69%`** trên `MVTec metal_nut`; tối ưu ngưỡng theo mục tiêu QC giúp giảm **escape rate từ `9.40%` xuống `0.75%`** trên `NEU-DET` và đạt **escape rate `0.00%`** (overkill `0.00%` trên tập test, `4.55%` toàn tập) trên `metal_nut` kể cả khi giả lập lỗi mới chưa từng gặp.
> - Tối ưu suy luận bằng **ONNX Runtime + Early-Exit OR-Fusion**, giảm độ trễ CPU (`p50`) từ **`279.60 ms` xuống `63.78 ms` (`4.38x` nhanh hơn, `15.68 FPS`)** và giảm **51.7% RAM (`1,297 MB -> 626 MB`)**; đóng gói dịch vụ FastAPI (`/inspect`), lưu vết SQLite, Streamlit dashboard và Docker Compose (`39/39` unit test pass).


---

## 9. Bảng Giải Thích Nhanh Các Thuật Ngữ & Chỉ Số Cốt Lõi (Để Ôn Tập)

| Thuật ngữ / Ký hiệu | Ý nghĩa bản chất dễ hiểu | Con số trong dự án của bạn |
|---|---|---|
| **OK / NG** | **OK (Good):** Sản phẩm đạt chuẩn. **NG (No Good):** Sản phẩm bị lỗi, cần loại bỏ. | Phán quyết cuối cùng của hệ thống cho mỗi bức ảnh. |
| **Escape Rate (Tỷ lệ lọt lỗi)** | Trong 100 sản phẩm **thật sự bị hỏng**, có bao nhiêu sản phẩm bị máy chấm nhầm là OK và lọt ra thị trường (`FN / Tổng NG thật`). Đây là chỉ số sống còn nhất trong nhà máy. | Mục tiêu `<= 2.0%`. Đạt **`0.75%`** trên `NEU-DET` (so với `9.40%` nếu để ngưỡng `0.50`) và **`0.00%`** trên `MVTec metal_nut`. |
| **Overkill Rate (Tỷ lệ báo giả)** | Trong 100 sản phẩm **thật sự tốt (OK)**, có bao nhiêu sản phẩm bị máy quá nhạy báo nhầm là NG (`FP / Tổng OK thật`). | Đạt **`0.00%`** trên tập `test` của `metal_nut` (`4.55%` toàn tập 115 ảnh) và **`22.50%`** trên `NEU-DET` tại ngưỡng chặt `0.25`. |
| **pHash (Perceptual Hash)** | Thuật toán rút gọn bức ảnh thành chuỗi bit 64-bit dựa trên cấu trúc tần số thấp. Hai ảnh gần giống nhau sẽ có khoảng cách Hamming rất nhỏ (`<= 4`), giúp lọc ảnh trùng trước khi chia `train/test`. | Loại bỏ `1` cặp trùng trong `NEU-DET` (còn `1,799` ảnh) và `8` cặp trùng trong `GC10-DET` (còn `2,286` ảnh). |
| **mAP@50 & mAP@50-95** | Độ chính xác trung bình của khung bao (bounding box) xét tại ngưỡng giao trên hợp `IoU = 0.50` (`mAP@50`) và trung bình từ `IoU = 0.50` đến `0.95` (`mAP@50-95`). | YOLOv8n (`mosaic=0.0`) đạt **`mAP@50 = 0.7836`** và **`mAP@50-95 = 0.4404`** trên tập `test` `NEU-DET`. |
| **PatchCore & Coreset** | Mô hình phát hiện bất thường dùng mạng `WideResNet50` đã tiền huấn luyện để trích xuất đặc trưng từng vùng nhỏ (patch) của các ảnh `OK`, sau đó nén lại còn `1%` vector đại diện nhất (**Coreset**). Khi có ảnh mới, nó đo khoảng cách tới Coreset: càng xa thì điểm bất thường càng cao. | Chỉ học từ `220` ảnh `OK`, đạt **`Image AUROC = 99.80%`** và **`Pixel AUROC = 98.69%`** trên `metal_nut`. |
| **Latency p50 / p95** | **p50 (trung vị):** 50% số lần chạy nhanh hơn mức này. **p95 (phân vị thứ 95):** 95% số lần chạy nhanh hơn mức này (đo độ ổn định thực tế khi tải nặng). | Toàn tuyến giảm từ `p50 = 279.60 ms` (`p95 = 361.33 ms`) xuống **`p50 = 63.78 ms` (`p95 = 100.22 ms`)**. |

---

## 10. Cẩm Nang 8 Câu Hỏi Phỏng Vấn Xoáy Sâu & Cách Trả Lời

### Câu 1: "Tại sao bạn kết hợp cả YOLOv8 và PatchCore mà không dùng 1 mô hình thôi?"
- **Trả lời:** Vì hai mô hình giải quyết hai mặt đối lập của bài toán QC:
  1. **YOLOv8n (Supervised Object Detection):** Rất giỏi chỉ đích danh **tên loại lỗi và tọa độ khung bao** cho 6 loại lỗi quen thuộc đã có trong lịch sử nhà máy, nhưng nó hoàn toàn "mù" trước một loại lỗi thứ 7 chưa từng xuất hiện trong tập huấn luyện (khi gặp lỗi lạ, YOLO trả về `confidence = 0` và kết luận nhầm là `OK`).
  2. **PatchCore (Unsupervised Anomaly Detection):** Chỉ cần học từ ảnh sản phẩm tốt (`OK`) để ghi nhớ chuẩn bề mặt. Nó bắt được **mọi dị tật lạ** với `AUROC = 99.80%`, nhưng lại không phân biệt được tên gọi cụ thể của từng loại lỗi.
  - Khi ghép cả hai qua cổng **HOẶC (`OR-Gate`)**, hệ thống vừa định danh được lỗi quen thuộc (nhờ YOLO), vừa có lưới an toàn chặn đứng lỗi mới (nhờ PatchCore, kéo tỷ lệ lọt lỗi mới từ `23.7%–26.9%` về `0.00%`).

### Câu 2: "Trong phần Ablation, tại sao tắt Mosaic (`mosaic=0.0`) lại làm tăng mAP trên NEU-DET?"
- **Trả lời:** Phép tăng cường `Mosaic` ghép 4 bức ảnh khác nhau thành 1 ảnh rất hiệu quả cho tập dữ liệu tự nhiên như COCO. Tuy nhiên, trên ảnh chụp cận cảnh bề mặt thép cán nóng (`NEU-DET`), việc ghép 4 tấm thép có độ sáng nền khác nhau tạo ra **hai đường ranh giới cắt ngang chữ thập rất sắc nét**. Mô hình dễ nhầm các đường biên ghép ảnh này với vết xước (`scratches`) hoặc vết nứt (`crazing`), đồng thời làm nhiễu tương phản cục bộ giữa vết lỗi và nền kim loại. Khi tắt `mosaic=0.0`, `mAP@50` tăng từ `0.7641` lên `0.7836`, đặc biệt lớp nứt chân chim (`crazing`) tăng tới `+10.3%` (`0.388 -> 0.491`).

### Câu 3: "Tại sao bạn chọn ngưỡng `tau_det = 0.25` và `tau_anom = 0.53`? Chọn như thế nào?"
- **Trả lời:** Em tuyệt đối không chọn ngưỡng bằng mắt hay chạm vào tập `test`. Quy trình chuẩn là:
  1. Đặt mục tiêu kỹ thuật của bài toán QC lên hàng đầu: **Tỷ lệ lọt lỗi (`escape_rate`) trên tập `validation` phải `<= 2.0%`**.
  2. Quét một lưới các ngưỡng ứng viên trên tập `val`, lọc ra các ngưỡng thỏa mãn `escape_rate <= 2.0%`, và chọn ngưỡng có `overkill_rate` thấp nhất (hoặc `F1` cao nhất). Kết quả trên tập `val` cho ra `tau_det = 0.25` và `tau_anom = 0.53`.
  3. Sau đó khóa cứng cặp ngưỡng này và đem đánh giá đúng 1 lần trên tập `test` độc lập: trên `NEU-DET`, `escape_rate` giảm từ `9.40%` (ở ngưỡng mặc định `0.50`) xuống chỉ còn **`0.75%`** (`2/266` mẫu).

### Câu 4: "Làm thế nào bạn giảm được thời gian chạy trên CPU từ 280 ms xuống 64 ms (nhanh gấp 4.38 lần)?"
- **Trả lời:** Em áp dụng 2 tầng tối ưu:
  1. **Tầng đồ thị tính toán (ONNX Runtime):** Xuất cả YOLOv8n và PatchCore sang định dạng `ONNX` (opset 18, cố định `batch=1`) và viết bộ tiền xử lý/NMS thuần NumPy để gỡ bỏ hoàn toàn overhead của PyTorch/Anomalib. Riêng bước này giúp giảm **51.7% dung lượng RAM** (`1,297 MB -> 626 MB`) và tăng tốc PatchCore từ `210.89 ms` xuống `114.47 ms`.
  2. **Tầng thuật toán hệ thống (Early-Exit OR-Gate):** Vì luật quyết định sản phẩm là phép `HOẶC` (`NG` nếu `YOLO >= 0.25` HOẶC `PatchCore >= 0.53`), nhánh YOLOv8n nhẹ hơn nhiều (`58.37 ms`) được xếp chạy trước. Nếu YOLOv8n đã phát hiện lỗi vượt ngưỡng `0.25`, phán quyết cuối cùng chắc chắn 100% là `NG`, nên hệ thống ngắt sớm (`Early-Exit`) mà không cần tốn thêm `114 ms` chạy PatchCore. Nhờ vậy độ trễ trung vị `p50` toàn tuyến giảm xuống **`63.78 ms` (`15.68 FPS`)**.

### Câu 5: "Chuyện gì xảy ra khi đèn LED ở nhà máy bị yếu đi hoặc camera bị mờ?"
- **Trả lời:** Trong thực nghiệm độ bền (Phase 4) với 6 phép nhiễu quang học, em phát hiện ra rằng khi giảm độ sáng đi 40% (`brightness_down`), mặc dù PatchCore vẫn giữ nguyên khả năng xếp hạng (`AUROC = 99.66%`), nhưng toàn bộ vector đặc trưng của ảnh bị tối bị đẩy ra xa Coreset chuẩn, khiến điểm bất thường tăng đồng loạt và gây ra **100% báo giả (`overkill`)** nếu dùng ngưỡng cố định `0.53`.
- Từ phát hiện này, thay vì để mô hình âm thầm báo lỗi sai hàng loạt, em thiết kế **Cổng kiểm tra chất lượng ảnh (`check_image_quality`)** chạy trước mô hình (chỉ tốn `< 1 ms`), đo độ sáng trung bình, độ tương phản và phương sai Laplacian để gắn cờ cảnh báo `too_dark`, `too_bright`, `low_contrast`, `blurry` trả về trực tiếp trong JSON của API `/inspect` và lưu vào SQLite để kỹ thuật viên biết cần kiểm tra lại đèn/camera.

### Câu 6: "Khi đem mô hình huấn luyện trên NEU-DET sang chạy trên GC10-DET thì kết quả thế nào?"
- **Trả lời:** Khi kiểm thử chéo miền (`Cross-Domain`) lớp lỗi tạp chất (`inclusion`) từ mô hình huấn luyện trên `NEU-DET` sang tập `GC10-DET`, chỉ số `AP@50` sụt từ `0.749` xuống `0.0686`. Nguyên nhân vật lý rất rõ ràng: ảnh `NEU-DET` là ảnh chụp cận cảnh (`200x200`) bề mặt thép cán nóng nhám tối, trong khi `GC10-DET` là ảnh chụp toàn khổ rộng (`2048x1000`) bề mặt thép cán nguội có phản xạ gương. Khi ép ảnh `2048x1000` về `640x640`, vết tạp chất co nhỏ lại gấp 10 lần so với kích thước anchor mà mô hình đã học. Bài học rút ra là khi triển khai sang dây chuyền mới, bắt buộc phải đồng bộ tỷ lệ `mm/pixel` hoặc fine-tune trên dữ liệu chụp tại trạm đó.

### Câu 7: "Làm sao bạn đảm bảo kết quả thực nghiệm không bị gian lận dữ liệu (Data Leakage)?"
- **Trả lời:** Các bộ dữ liệu công khai thường chứa các khung hình cắt liên tiếp từ cùng một cuộn thép. Em đã viết module băm cảm nhận (`pHash` trong `src/steelvision/data/split.py`), tính ma trận khoảng cách Hamming giữa toàn bộ các ảnh và dùng thuật toán Union-Find gom tất cả các cặp ảnh có khoảng cách Hamming `<= 4` vào cùng một cụm *trước khi* chia `train / val / test`. Nhờ đó phát hiện và loại bỏ `1` cặp ảnh trùng trong `NEU-DET` và `8` cặp trùng trong `GC10-DET`, đảm bảo không có ảnh gần giống nhau nào vừa nằm ở `train` vừa nằm ở `test`.

### Câu 8: "Hạn chế lớn nhất hiện tại của hệ thống là gì và hướng nâng cấp tiếp theo?"
- **Trả lời:** Có 2 điểm có thể nâng cấp tiếp:
  1. Lớp lỗi vết nứt chân chim siêu mảnh (`crazing` trên `NEU-DET`) hiện đạt `mAP@50 = 0.491` do vết nứt chỉ rộng 1–2 pixel và lẫn vào vân nền thép. Hướng giải quyết là áp dụng kỹ thuật cắt ô **SAHI (Slicing Aided Hyper Inference)** trên ảnh độ phân giải cao.
  2. Nhánh PatchCore ONNX (`FP32`) hiện chiếm `114.47 ms` trên CPU; nếu lượng tử hóa tĩnh sang **INT8 (OpenVINO / ONNX Runtime Quantization)**, thời gian chạy nhánh PatchCore trên các sản phẩm `OK` có thể giảm thêm một nửa xuống dưới `55 ms`.
