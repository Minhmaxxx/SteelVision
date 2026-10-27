# Báo cáo Đánh giá QC & Độ bền (Phase 4) — SteelVision

> Toàn bộ số liệu trong báo cáo này được sinh tự động và tái lập hoàn toàn bằng lệnh:
> ```bash
> make eval   # hoặc: python -m steelvision.eval.run_qc --config configs/eval.yaml
> ```
> Kết quả JSON gốc lưu tại `runs/eval/qc_report.json` (`seed = 42`, mục tiêu QC: **escape rate ≤ 2.0%**).

---

## 1. Phương pháp đánh giá cấp sản phẩm (Product-level QC)

Trong nhà máy, mỗi ảnh kiểm tra tương ứng với **1 sản phẩm**. Thay vì chỉ dừng ở chỉ số học thuật (mAP hay AUROC), hệ thống được đánh giá theo hai chỉ số vận hành thực tế:

- **Escape rate (Tỷ lệ lọt lỗi):**
  $$\text{Escape Rate} = \frac{\text{FN}}{\text{TP} + \text{FN}} = \frac{\text{Số sản phẩm lỗi bị phán OK}}{\text{Tổng số sản phẩm lỗi thật}}$$
  → Ưu tiên số 1: giữ **Escape rate ≤ 2.0%** để lỗi không lọt xuống công đoạn sau hoặc tới tay khách hàng.
- **Overkill rate (Tỷ lệ bắt nhầm / loại nhầm hàng tốt):**
  $$\text{Overkill Rate} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{\text{Số sản phẩm tốt bị phán NG}}{\text{Tổng số sản phẩm tốt thật}}$$
  → Ưu tiên số 2: tối thiểu hóa Overkill rate trong miền ngưỡng đã thỏa mãn Escape rate ≤ 2.0%.

**Quy tắc chọn điểm vận hành (`select_operating_point`):**
Quét lưới ngưỡng trên tập **Validation**, lọc các cặp ngưỡng đạt `escape_rate <= 0.02`, chọn ngưỡng có `overkill_rate` thấp nhất, sau đó cố định ngưỡng này để đo một lần trên tập **Test** độc lập.

---

## 2. Đường đánh đổi ngưỡng & Điểm vận hành QC

### 2.1. Nhánh Anomaly Detection — PatchCore ONNX trên MVTec AD (`metal_nut`)

- Tập dữ liệu (`115` sản phẩm: `93` NG thuộc 4 kiểu lỗi `bent`, `color`, `flip`, `scratch` và `22` OK `good`), chia phân tầng 50/50 thành `val` (`56` ảnh) và `test` (`59` ảnh).
- Trên tập `val`, thuật toán chọn ngưỡng vận hành **	au_anom = 0.53** (`val escape_rate = 0.00%`, `val overkill_rate = 9.09%`).
- Khi áp ngưỡng **	au_anom = 0.53** lên tập **Test độc lập (`59` ảnh: `48` NG, `11` OK)**:
  - **Escape rate = `0.00%` (`0/48` lọt lỗi)**
  - **Overkill rate = `0.00%` (`0/11` bắt nhầm trên test; `4.55%` = `1/22` trên toàn tập 115 ảnh)**
  - **Precision = `100.0%`, Recall = `100.0%`, F1 = `1.000`** (`F1 = 0.9947` trên toàn tập).

#### Bảng đánh đổi theo ngưỡng 	au_anom (trên toàn tập 115 sản phẩm `metal_nut`)

| Ngưỡng 	au_anom | TP (Bắt đúng NG) | FN (Lọt lỗi) | FP (Loại nhầm OK) | TN (Đạt OK) | **Escape Rate** | **Overkill Rate** | F1-Score | Ghi chú |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.35 | 93 | 0 | 21 | 1 | 0.00% | 95.45% | 0.8986 | Quá nhạy |
| 0.40 | 93 | 0 | 14 | 8 | 0.00% | 63.64% | 0.9300 | |
| 0.45 | 93 | 0 | 6 | 16 | 0.00% | 27.27% | 0.9688 | |
| 0.48 | 93 | 0 | 3 | 19 | 0.00% | 13.64% | 0.9841 | |
| 0.50 | 93 | 0 | 2 | 20 | 0.00% | 9.09% | 0.9894 | Ngưỡng mặc định |
| 0.52 | 93 | 0 | 1 | 21 | 0.00% | 4.55% | 0.9947 | |
| **0.53** | **93** | **0** | **1** | **21** | **0.00%** | **4.55%** | **0.9947** | **✅ Điểm vận hành chọn từ Val** |
| 0.55 | 91 | 2 | 1 | 21 | 2.15% | 4.55% | 0.9838 | Vượt trần 2% escape |
| 0.60 | 90 | 3 | 0 | 22 | 3.23% | 0.00% | 0.9836 | 0% overkill nhưng lọt 3 lỗi |
| 0.65 | 89 | 4 | 0 | 22 | 4.30% | 0.00% | 0.9780 | |
| 0.70 | 87 | 6 | 0 | 22 | 6.45% | 0.00% | 0.9667 | |

---

### 2.2. Nhánh Object Detection — YOLOv8n (`mosaic=0.0`) trên NEU-DET

- Để đo đồng thời cả **Escape rate** và **Overkill rate** trên bề mặt thép cán nóng NEU-DET (vốn chỉ gồm các ảnh có ít nhất 1 bounding box lỗi), pipeline trích xuất thêm `80` vùng nền thép không lỗi (`OK` crops, tách biệt hoàn toàn với mọi ground-truth box kèm lề an toàn 6 px) trên mỗi tập `val` (`270` NG + `80` OK = `350` sản phẩm) và `test` (`266` NG + `80` OK = `346` sản phẩm).
- Trên tập `val`, thuật toán chọn ngưỡng vận hành **	au_det = 0.25** (`val escape_rate = 1.85% <= 2.0%`, `val overkill_rate = 23.75%`).
- Khi áp ngưỡng **	au_det = 0.25** lên tập **Test độc lập (`266` NG + `80` OK)**:
  - **Escape rate = `0.75%` (`2/266` lọt lỗi — đạt mục tiêu ≤ 2%)**
  - **Overkill rate = `22.50%` (`18/80`)**
  - **Precision = `93.62%`, Recall = `99.25%`, F1 = `0.9635`**.

#### Bảng đánh đổi theo ngưỡng 	au_det trên tập Test NEU-DET (`266` NG + `80` OK)

| Ngưỡng 	au_det | TP | FN (Lọt lỗi) | FP (Loại nhầm) | TN | **Escape Rate** | **Overkill Rate** | F1-Score | Ghi chú |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.10 | 266 | 0 | 39 | 41 | 0.00% | 48.75% | 0.9317 | Không lọt lỗi, loại nhầm cao |
| 0.15 | 266 | 0 | 29 | 51 | 0.00% | 36.25% | 0.9483 | |
| 0.20 | 266 | 0 | 22 | 58 | 0.00% | 27.50% | 0.9603 | |
| **0.25** | **264** | **2** | **18** | **62** | **0.75%** | **22.50%** | **0.9635** | **✅ Điểm vận hành chọn từ Val (F1 cực đại)** |
| 0.30 | 261 | 5 | 18 | 62 | 1.88% | 22.50% | 0.9578 | |
| 0.35 | 257 | 9 | 15 | 65 | 3.38% | 18.75% | 0.9554 | Vượt trần 2% escape |
| 0.40 | 251 | 15 | 11 | 69 | 5.64% | 13.75% | 0.9508 | |
| 0.50 | 241 | 25 | 6 | 74 | 9.40% | 7.50% | 0.9396 | Ngưỡng mặc định 0.5 làm lọt 9.4% lỗi! |
| 0.60 | 216 | 50 | 2 | 78 | 18.80% | 2.50% | 0.8926 | |
| 0.70 | 181 | 85 | 0 | 80 | 31.95% | 0.00% | 0.8098 | |

> **Bài học QC quan trọng:** Nếu chọn ngưỡng `0.50` theo thói quen, hệ thống để lọt tới **9.40% (`25/266`)** sản phẩm lỗi. Việc chọn 	au_det = 0.25 từ đường đánh đổi trên tập validation giúp **giảm 12.5 lần tỷ lệ lọt lỗi** (`9.40% → 0.75%`) và đồng thời đạt đỉnh F1 (`0.9635`).

---

## 3. Kiểm thử "Lỗi chưa từng gặp" (Leave-One-Class-Out Unseen Defect Test)

### 3.1. Khi dùng Detector đơn lẻ trên NEU-DET (ẩn lần lượt 1 trong 6 lớp lỗi)

Giả lập tình huống dây chuyền xuất hiện một loại lỗi mới mà bộ phát hiện có giám sát (YOLO) chưa từng được gán nhãn khi huấn luyện:

| Lớp lỗi bị ẩn (chưa từng gặp) | Escape Rate khi đủ 6 lớp | Escape Rate khi thiếu lớp này (Detector-only) | Số sản phẩm lỗi bị lọt (FN) | Mức tăng Escape Rate ($\Delta$) |
|---|---:|---:|---:|---:|
| `rolled_in_scale` | 0.75% | **17.29%** | 46 / 266 | +16.54% |
| `crazing` | 0.75% | **16.92%** | 45 / 266 | +16.17% |
| `inclusion` | 0.75% | **16.17%** | 43 / 266 | +15.41% |
| `patches` | 0.75% | **16.17%** | 43 / 266 | +15.41% |
| `scratches` | 0.75% | **15.04%** | 40 / 266 | +14.29% |
| `pitted_surface` | 0.75% | **14.66%** | 39 / 266 | +13.91% |

→ **Nhận xét:** Bộ object detector có giám sát hoàn toàn "mù" trước lớp lỗi chưa từng có trong tập nhãn: tỷ lệ lọt lỗi vọt từ `0.75%` lên **`14.66% – 17.29%`**.

### 3.2. Khi hợp nhất Detector + PatchCore (`fusion.py`) trên MVTec `metal_nut`

Ẩn lần lượt từng nhóm lỗi (`bent`, `color`, `flip`, `scratch`) khỏi nhánh Detector và kiểm chứng khả năng bọc lót của nhánh PatchCore (vốn chỉ học từ ảnh `good`) tại 	au_anom = 0.53:

| Nhóm lỗi chưa từng gặp | Số mẫu lỗi của nhóm | Escape Rate (Detector-only) | FN (Detector-only) | **Escape Rate sau hợp nhất (Detector + PatchCore)** | **FN sau hợp nhất** | Overkill Rate sau hợp nhất |
|---|---:|---:|---:|---:|---:|---:|
| `bent` (cong vênh) | 25 | 26.88% | 25 / 93 | **0.00%** | **0 / 93** | 4.55% |
| `color` (đổi màu) | 22 | 23.66% | 22 / 93 | **0.00%** | **0 / 93** | 4.55% |
| `flip` (lật ngược) | 23 | 24.73% | 23 / 93 | **0.00%** | **0 / 93** | 4.55% |
| `scratch` (trầy xước) | 23 | 24.73% | 23 / 93 | **0.00%** | **0 / 93** | 4.55% |

→ **Kết luận:** Luật hợp nhất `OR` trong [`fusion.py`](file:///e:/Vision/src/steelvision/decision/fusion.py) giải quyết trọn vẹn bài toán lỗi mới: ngay cả khi Detector bỏ sót 100% một kiểu lỗi chưa từng gặp, nhánh PatchCore vẫn bắt trọn **100% (`0%` escape rate)**.

---

## 4. Phân tích độ bền trước biến đổi môi trường (Perturbation Robustness)

6 phép biến đổi mô phỏng sự cố thực tế trên dây chuyền:
1. `brightness_up`: quá sáng ($\alpha=1.15, \beta=+45$).
2. `brightness_down`: thiếu sáng ($\alpha=0.70, \beta=-35$).
3. `contrast_low`: giảm tương phản ($0.55\times$ quanh mức xám 128).
4. `gaussian_blur`: mờ lấy nét / rung băng chuyền (nhân $5\times 5, \sigma=1.5$).
5. `gaussian_noise`: nhiễu cảm biến camera ($\sigma=18$).
6. `rotate_5deg`: lệch gá đặt sản phẩm ($+5^\circ$).

### 4.1. Độ bền của YOLOv8n trên tập Test NEU-DET (`266` NG + `80` OK, 	au_det = 0.25)

| Điều kiện | Class-agnostic AP50 | $\Delta$ AP50 | Box Precision | Box Recall | **Escape Rate** | **Overkill Rate** | Đánh giá độ bền |
|---|---:|---:|---:|---:|---:|---:|---|
| **clean (chuẩn)** | **0.7493** | **0.0000** | 0.7534 | 0.6954 | **0.75%** | **22.50%** | Mốc cơ sở |
| `brightness_up` | 0.7409 | −0.0084 | 0.7359 | 0.6954 | 1.50% | 21.25% | ✅ Rất bền (vẫn ≤ 2% escape) |
| `brightness_down` | 0.6939 | −0.0554 | 0.7087 | 0.6531 | 1.13% | 18.75% | ✅ Bền (vẫn ≤ 2% escape) |
| `rotate_5deg` | 0.6664 | −0.0829 | 0.7034 | 0.6217 | 1.13% | 27.50% | ✅ Bền với lệch góc nhẹ |
| `contrast_low` | 0.6900 | −0.0593 | 0.7764 | 0.5997 | **5.26%** | 12.50% | ⚠️ Lỗi mờ như `crazing` bị sót |
| `gaussian_blur` | 0.4283 | −0.3210 | 0.6009 | 0.4301 | 3.01% | 45.00% | ❌ Sụt mạnh AP50 (−32.1 điểm) |
| `gaussian_noise` | 0.0595 | −0.6898 | 0.1735 | 0.1130 | **16.54%** | **91.25%** | ❌ Sụp đổ do nhiễu hạt đè lên texture 200×200 |

### 4.2. Độ bền của PatchCore ONNX trên MVTec `metal_nut` (`115` ảnh, 	au_anom = 0.53)

| Điều kiện | Image AUROC | $\Delta$ AUROC | **Escape Rate** | **Overkill Rate** | F1-Score | Đánh giá độ bền |
|---|---:|---:|---:|---:|---:|---|
| **clean (chuẩn)** | **0.9990** | **0.0000** | **0.00%** | **4.55%** | 0.9947 | Mốc cơ sở |
| `contrast_low` | 0.9985 | −0.0005 | 0.00% | 4.55% | 0.9947 | ✅ Không suy suyển |
| `rotate_5deg` | 0.9980 | −0.0010 | 1.08% | 0.00% | 0.9946 | ✅ Rất bền (trích đặc trưng cục bộ) |
| `gaussian_blur` | 0.9980 | −0.0010 | 1.08% | 4.55% | 0.9892 | ✅ Rất bền nhờ AvgPool3×3 của PatchCore |
| `brightness_up` | 0.9976 | −0.0015 | 0.00% | 9.09% | 0.9894 | ✅ Bền |
| `brightness_down` | 0.9804 | −0.0186 | 0.00% | **100.00%** | 0.8942 | ⚠️ AUROC vẫn 98%, nhưng lệch thang điểm làm báo NG toàn bộ |
| `gaussian_noise` | 0.9785 | −0.0205 | 0.00% | **100.00%** | 0.8942 | ⚠️ AUROC vẫn 97.9%, nhưng nhiễu đẩy khoảng cách đặc trưng vượt ngưỡng |

> **Phát hiện kỹ thuật rất đáng giá cho thiết kế hệ thống (Phase 7):**
> 1. **Hai mô hình có điểm yếu bổ trợ nhau:** YOLOv8n rất nhạy với `gaussian_blur` (AP50 giảm 32 điểm), trong khi PatchCore gần như miễn nhiễm với `gaussian_blur` (AUROC `0.9980`, escape `1.08%`) nhờ lớp `AvgPool2d(3×3)` làm trơn đặc trưng cục bộ.
> 2. **Hướng suy biến an toàn (Fail-safe) của PatchCore:** Khi ánh sáng sụt mạnh (`brightness_down`) hoặc cảm biến nhiễu nặng (`gaussian_noise`), thứ tự xếp hạng của PatchCore vẫn giữ rất cao (`AUROC ~ 0.98`), nhưng toàn bộ điểm khoảng cách bị dịch lên trên 	au_anom = 0.53, khiến mô hình **từ chối toàn bộ (`overkill = 100%, escape = 0%`)** thay vì để lọt hàng lỗi.
> 3. **Hệ quả cho tiền xử lý ở API (`/inspect`):** Cần có bước **kiểm tra chất lượng ảnh đầu vào (Image Quality Gate)** — đo độ sáng trung bình, độ tương phản và độ nét (Laplacian variance) trước khi đưa vào mô hình, để cảnh báo lỗi đèn/camera thay vì báo NG giả hàng loạt.

---

## 5. Kiểm thử khác miền (Cross-Domain: Train NEU-DET → Test GC10-DET)

Đánh giá trực tiếp mô hình `YOLOv8n` huấn luyện trên `NEU-DET` (thép cán nóng, ảnh cận cảnh `200×200`) khi chạy không fine-tune trên tập test của `GC10-DET` (`349` ảnh tấm thép cán nguội độ phân giải cao `2048×1000`):

| Chỉ số | In-Domain (`NEU-DET → NEU-DET`) | Cross-Domain (`NEU-DET → GC10-DET`) | Độ sụt giảm ($\Delta$) |
|---|---:|---:|---:|
| Class-agnostic Box AP50 | **0.7493** | **0.0603** | **−0.6890** |
| Shared-class `inclusion` AP50 | **0.8550** | **0.0108** | **−0.8442** |
| Box Precision / Recall | 0.7534 / 0.6954 | 0.1554 / 0.1557 | −0.5980 / −0.5397 |
| Escape Rate tại $\tau_{\text{det}} = 0.25$ | **0.75%** | **58.45%** | **+57.70%** |
| Escape Rate tại $\tau_{\text{det}} = 0.15$ | **0.00%** | **44.99%** | **+44.99%** |

**Nguyên nhân gốc rễ:**
1. **Lệch tỷ lệ không gian (Scale shift):** Ảnh NEU-DET là vùng crop cận cảnh `200×200` quanh vết lỗi, trong khi GC10-DET chụp toàn bề rộng băng thép (`2048×1000`); khi resize về `640×640`, đặc trưng bề mặt và kích thước tương đối của vết `inclusion` thay đổi hoàn toàn.
2. **Khác biệt vật lý bề mặt:** Thép cán nóng (NEU) có nền nhám tối đồng nhất, còn thép cán nguội (GC10) có độ phản xạ gương cao và vệt sáng dọc băng chuyền.

→ **Kết luận thực tiễn:** Mô hình phát hiện lỗi bề mặt công nghiệp **không thể bê nguyên (zero-shot) giữa hai dây chuyền có trường nhìn (FOV) và hệ thống chiếu sáng khác nhau**. Điều này chứng minh sự cần thiết bắt buộc của **Phase 5 (thu thập dữ liệu tự chụp với khoảng cách và ánh sáng cố định trên đúng miền mục tiêu rồi fine-tune)**.
