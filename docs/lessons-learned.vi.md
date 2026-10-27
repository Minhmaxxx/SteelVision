# Bài Học Rút Ra Khi Xây Dựng Hệ Thống Kiểm Tra Lỗi Bề Mặt Kim Loại (SteelVision)

Dự án **SteelVision** bắt đầu từ một câu hỏi thực tế trong phỏng vấn Computer Vision cho sản xuất công nghiệp (QC): *"Tại sao các đồ án phân loại lỗi bề mặt trên tập NEU-DET đạt 98% accuracy nhưng đem ra dây chuyền thật lại không dùng được?"*

Dưới đây là 6 bài học kỹ thuật quan trọng nhất được rút ra trong suốt quá trình xây dựng hệ thống từ dữ liệu thô đến dịch vụ FastAPI + ONNX trên CPU.

---

## 1. Dữ liệu công khai luôn có rò rỉ (Data Leakage) và nhãn lỗi nếu không kiểm định trước

Khi tải `NEU-DET` (`1.800` ảnh) và `GC10-DET` (`2.294` ảnh) từ nguồn công khai:
- Nhờ chạy bước khử trùng lặp bằng **perceptual hash (`pHash`, ngưỡng khoảng cách Hamming $\le 4$)** trước khi chia tập `train / val / test`, chúng tôi phát hiện `1` cặp ảnh trùng trong `NEU-DET` và `8` cặp ảnh trùng trong `GC10-DET`. Nếu chia ngẫu nhiên mà không lọc pHash, các ảnh gần như giống hệt nhau sẽ nằm ở cả `train` và `test`, làm giả tạo chỉ số mAP.
- Trong quá trình viết hàm chia tập phân tầng đa nhãn (`stratified_split`), một lỗi kinh điển về chỉ số (`index` toàn cục so với vị trí trong danh sách đã lọc) đã được phát hiện nhờ `assert` kiểm tra tổng số mẫu các tập con. Bài học: **luôn viết unit test và invariant assertion cho pipeline dữ liệu trước khi huấn luyện bất kỳ mô hình nào**.

---

## 2. Tăng cường dữ liệu kiểu COCO (`Mosaic`) có thể làm hại bài toán lỗi bề mặt công nghiệp

Trong YOLO mặc định, `mosaic=1.0` ghép 4 ảnh khác nhau lại để giúp mô hình học ngữ cảnh đa dạng của vật thể tự nhiên (xe cộ, người, động vật). Tuy nhiên, trên bề mặt thép cán nóng (`NEU-DET`):
- Lỗi bề mặt phụ thuộc chặt chẽ vào **độ tương phản cục bộ giữa vết lỗi và nền kim loại đồng nhất xung quanh**.
- Việc cắt ghép 4 mảnh thép có độ sáng và vân nền khác nhau tạo ra các đường biên giả tạo (artificial seams) trông rất giống vết nứt (`crazing`) hoặc vết xước (`scratches`).
- Khi tắt Mosaic (**Ablation 2: `mosaic=0.0`**), mAP@50 trên tập test tăng từ **`0.7641` lên `0.7836` (`+1.95%`)** và mAP@50-95 tăng từ **`0.4272` lên `0.4404`**, đặc biệt lớp khó nhất là `crazing` tăng từ `0.388` lên `0.491` (`+10.3%`). Ngược lại, việc chỉ tăng kích thước mô hình lên `YOLOv8s` hay phóng đại ảnh `200x200` lên `800x800` không mang lại hiệu quả tương đương.

---

## 3. Bộ phát hiện có giám sát không biết "OK là gì" và mù trước lỗi mới — cần PatchCore làm lưới an toàn

Một bộ object detector (YOLOv8) chỉ được dạy tìm $K$ lớp lỗi đã định nghĩa trước. Điều này dẫn đến hai lỗ hổng chí mạng trên dây chuyền:
1. Nếu xuất hiện một **loại lỗi mới chưa từng có trong tập huấn luyện** (ví dụ vết bẩn màu `color` hoặc dập ngược `flip` trên ốc kim loại), detector sẽ im lặng (`confidence = 0`) và phán quyết nhầm là **OK**.
2. Trong thí nghiệm giả lập lỗi mới (*leave-one-class-out* ở Phase 4), khi giấu từng lớp lỗi đi, tỷ lệ lọt lỗi (**escape rate**) của riêng nhánh detector nhảy vọt lên **`14.66%–17.29%`** trên `NEU-DET` và **`23.66%–26.88%`** trên `metal_nut`.

Khi ghép song song **PatchCore** (chỉ học từ `220` ảnh `good` không lỗi, đạt `image_AUROC = 99.80%`) theo luật hợp quyết định **OR-gate** (`NG` nếu `det >= τ_det` HOẶC `anom >= τ_anom`), toàn bộ các mẫu lỗi mới bị detector bỏ sót đều bị nhánh PatchCore chặn lại, kéo **escape rate về đúng `0.00%`** trên cả 4 kịch bản lỗi ẩn.

---

## 4. Ngưỡng mặc định `0.50` là thảm họa trong QC: phải chọn ngưỡng theo `Escape Rate` và `Overkill Rate`

Trong học máy học thuật, người ta thường dùng ngưỡng xác suất mặc định `0.50`. Nhưng trong kiểm định chất lượng sản xuất:
- **Lọt lỗi (Escape / False Negative)** đồng nghĩa giao hàng hỏng cho khách hàng — chi phí cực đắt.
- **Báo giả (Overkill / False Positive)** chỉ tốn công kỹ thuật viên kiểm tra lại.

Trên tập test `NEU-DET`, nếu giữ ngưỡng mặc định `0.50`, tỷ lệ lọt lỗi lên tới **`9.40%`** (`25/266` sản phẩm lỗi bị lọt). Bằng cách vẽ đường cong đánh đổi trên tập `val` với mục tiêu cứng **`escape_rate <= 2.0%`**, chúng tôi chọn được điểm vận hành 	au_det = 0.25 và 	au_anom = 0.53. Khi khóa ngưỡng này đem đo một lần duy nhất trên tập `test` độc lập:
- **NEU-DET:** `escape_rate` giảm hơn **12 lần** (từ `9.40%` xuống **`0.75%`**), đạt `Recall = 99.25%` và `F1 = 0.9635`.
- **MVTec `metal_nut`:** `escape_rate = 0.00%`, `overkill_rate = 0.00%` trên phân nửa `test` độc lập (`4.55%` trên toàn bộ 115 ảnh).

---

## 5. Mô hình tốt vẫn hỏng khi sụt áp đèn LED — lý do phải có cổng kiểm tra chất lượng ảnh (Image Quality Gate)

Khi kiểm thử độ bền với 6 phép nhiễu quang học ở Phase 4, một hiện tượng rất thú vị đã lộ diện:
- **PatchCore giữ nguyên khả năng xếp hạng (`image_AUROC >= 97.85%`)** dưới mọi loại nhiễu.
- Tuy nhiên, khi ảnh bị tối đi 40% (`brightness_down`) hoặc nhiễu hạt cảm biến (`gaussian_noise`), khoảng cách đặc trưng đến coreset dịch chuyển đồng loạt lên cao, khiến **100% sản phẩm tốt bị báo nhầm là NG (`overkill = 100%`)** nếu dùng ngưỡng cố định 	au_anom = 0.53.
- Thay vì đổ lỗi cho mô hình, giải pháp kỹ thuật thực tế và rẻ nhất là đặt một **Image Quality Gate (`check_image_quality`)** ngay đầu API để đo độ sáng trung bình, độ tương phản và phương sai độ nét Laplacian chỉ trong `< 1 ms`, cảnh báo ngay lỗi đèn chiếu sáng (`too_dark`, `too_bright`) hoặc lệch nét camera (`blurry`) trước khi suy luận.

---

## 6. Tối ưu CPU không chỉ là xuất ONNX mà còn là tận dụng cấu trúc logic của bài toán (Early-Exit)

Khi triển khai trên máy tính công nghiệp chỉ có CPU (`AMD Ryzen 7 5800H`, `batch=1`):
- Xuất sang **ONNX Runtime** giúp giảm bộ nhớ RSS RAM một nửa (từ **`1,297 MB` xuống `626 MB`**) và tăng tốc riêng nhánh PatchCore gấp **`1.84x`** (`210.89 ms -> 114.47 ms`).
- Quan trọng hơn, vì luật quyết định sản phẩm là phép **HOẶC (`OR`)**, khi nhánh YOLOv8n (`58.37 ms`) đã phát hiện lỗi quen thuộc với độ tin cậy $\ge \tau_{\text{det}}^*$, phán quyết cuối cùng chắc chắn là **NG** kèm vị trí hộp bao và tên lỗi. Việc kích hoạt chế độ **Early-Exit** cho phép bỏ qua nhánh PatchCore trên các sản phẩm đã rõ lỗi, kéo độ trễ trung vị `p50` toàn tuyến từ **`279.60 ms` (PyTorch) xuống `63.78 ms` (`15.68 FPS`, nhanh gấp `4.38x`)** mà không làm thay đổi bất kỳ phán quyết OK/NG nào.
