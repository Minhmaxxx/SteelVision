# SteelVision – Hệ thống kiểm tra lỗi bề mặt kim loại (Machine Vision)

> Tên dự án chỉ là tên tạm, có thể đổi. Các con số kết quả để trống (TBD) và chỉ điền sau khi đã đo thật.

## 1. Tóm tắt

SteelVision là hệ thống kiểm tra chất lượng (QC) bề mặt kim loại end-to-end. Hệ thống nhận ảnh sản phẩm, quyết định **OK / NG**, chỉ ra **loại lỗi và vị trí lỗi**, và ghi lại lịch sử kiểm tra để truy vết. Dự án đi trọn pipeline: dữ liệu tự thu thập → huấn luyện → đánh giá theo chỉ số QC → tối ưu hóa → triển khai thành dịch vụ có API và dashboard.

## 2. Bối cảnh và vấn đề

Trong sản xuất, kiểm tra bề mặt bằng mắt người chậm, không đồng nhất và dễ bỏ sót. Các bài toán mẫu trên dataset công khai thường dừng ở độ chính xác phân loại, trong khi nhà máy quan tâm đến:

- Lỗi nằm **ở đâu** và **to bao nhiêu**.
- Sản phẩm lỗi có bị **lọt** qua không, và sản phẩm tốt có bị **loại nhầm** nhiều không.
- Hệ thống chạy đủ **nhanh** trên phần cứng thường (CPU) hay không.
- Hệ thống có còn ổn khi **ánh sáng hoặc góc chụp thay đổi** hay không.
- Nếu xuất hiện **loại lỗi chưa từng gặp** thì sao.

Dự án được thiết kế để trả lời trực tiếp các câu hỏi này.

## 3. Mục tiêu

1. Phát hiện và định vị lỗi bề mặt bằng mô hình object detection.
2. Quyết định OK/NG đáng tin cậy bằng cách kết hợp detection với anomaly detection (học từ ảnh bình thường), để bắt được cả lỗi chưa từng thấy.
3. Đánh giá bằng chỉ số QC (escape rate, overkill rate) thay vì chỉ accuracy hoặc mAP.
4. Kiểm chứng độ bền của mô hình trước thay đổi ánh sáng, nhiễu, mờ và khác miền dữ liệu.
5. Tối ưu để chạy được trên CPU, báo cáo latency trước và sau tối ưu.
6. Đóng gói thành dịch vụ (API + Docker) với dashboard và lưu vết kiểm tra.

## 4. Phạm vi

**Bắt buộc**

- Detection bằng YOLO trên dữ liệu công khai và dữ liệu tự chụp.
- Anomaly detection (PatchCore) huấn luyện chỉ từ ảnh OK.
- Bộ dữ liệu tự thu thập và tự gán nhãn, có hướng dẫn gán nhãn.
- Đánh giá QC và kiểm thử độ bền.
- Xuất ONNX và đo hiệu năng.
- FastAPI, Docker, kiểm thử tự động, dashboard.
- README, video demo, model card.

**Mở rộng (làm nếu còn điều kiện)**

- Segmentation để đo diện tích lỗi.
- Lượng tử hóa INT8.
- SAHI (cắt ảnh thành ô nhỏ) cho lỗi kích thước nhỏ.
- Giả lập tín hiệu loại sản phẩm (reject) qua Modbus TCP hoặc serial.

**Ngoài phạm vi**

- Điều khiển phần cứng dây chuyền thật.
- Huấn luyện mô hình từ đầu không dùng trọng số tiền huấn luyện.

## 5. Kiến trúc hệ thống

```
        Ảnh / Camera / Video
                │
                ▼
      ┌───────────────────┐
      │  Tiền xử lý       │  (resize, chuẩn hóa, kiểm tra chất lượng ảnh)
      └─────────┬─────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
 ┌─────────────┐  ┌────────────────┐
 │ Detector    │  │ Anomaly model  │
 │ (YOLO)      │  │ (PatchCore)    │
 │ loại + vị trí│  │ điểm bất thường│
 └──────┬──────┘  └───────┬────────┘
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │ Luật quyết định │  OK / NG + lý do
        └────────┬────────┘
                 ▼
   ┌─────────────┴──────────────┐
   ▼                            ▼
 FastAPI (/inspect)        Lưu vết (SQLite + ảnh NG)
                                │
                                ▼
                         Dashboard (Streamlit)
```

**Luật quyết định (dự kiến):** sản phẩm là NG nếu detector tìm thấy lỗi vượt ngưỡng tin cậy **hoặc** điểm anomaly vượt ngưỡng. Detector cho biết loại và vị trí, anomaly model là lớp bảo vệ cho lỗi chưa từng thấy. Các ngưỡng được chọn trên tập validation theo mục tiêu QC, không chọn theo cảm tính.

## 6. Dữ liệu

| Nguồn | Vai trò | Ghi chú |
|---|---|---|
| NEU-DET | Huấn luyện/đánh giá detection | Lỗi bề mặt thép cán nóng, 6 loại. Cần xác minh định dạng nhãn bounding box. |
| GC10-DET | Huấn luyện/đánh giá detection, kiểm thử khác miền | 10 loại lỗi trên thép. Cần xác minh nhãn bounding box. |
| MVTec AD (nhóm kim loại) | Kiểm chứng pipeline anomaly detection | Có sẵn ảnh bình thường để huấn luyện. |
| **Dữ liệu tự chụp** | Huấn luyện + đánh giá trên điều kiện thật | 300–500 ảnh, ít nhất 40% là ảnh OK, 2–3 điều kiện ánh sáng, gán nhãn bằng CVAT hoặc Label Studio. |

**Nguyên tắc dữ liệu**

- Chia train/val/test theo ảnh, kiểm tra trùng lặp bằng perceptual hash để tránh rò rỉ.
- Có tài liệu hướng dẫn gán nhãn (định nghĩa lỗi, kích thước tối thiểu).
- Gán nhãn lại một phần (khoảng 10%) để đo mức nhất quán của nhãn.
- Ghi rõ nguồn, giấy phép và cách tái tạo dataset.

## 7. Phương pháp

**Detection**
- YOLOv8 (n/s, thử m nếu cần) làm mô hình chính.
- Ablation theo nguyên tắc mỗi lần chỉ đổi một yếu tố: kích thước ảnh đầu vào, augmentation, xử lý mất cân bằng lớp, kích cỡ mô hình.
- Dòng baseline so sánh: mô hình phân loại + Grad-CAM++ từ dự án cũ.

**Anomaly detection**
- PatchCore (Anomalib) huấn luyện chỉ từ ảnh OK.
- Báo cáo image-level AUROC và pixel-level AUROC.

**Đánh giá**
- Detection: mAP50, mAP50-95, AP theo lớp và theo kích thước lỗi.
- Cấp sản phẩm (mỗi ảnh là một sản phẩm):
  - **Escape rate:** sản phẩm lỗi bị cho qua (ưu tiên giữ thấp).
  - **Overkill rate:** sản phẩm tốt bị loại nhầm.
  - Đường đánh đổi theo ngưỡng, từ đó chọn điểm vận hành.
- Hiệu năng: latency p50/p95 (batch=1, có warm-up), RAM, FPS.
- Độ bền: thay đổi sáng, mờ, nhiễu, xoay nhẹ, đổi điều kiện ánh sáng thật, và kiểm thử khác miền (train dataset này, test dataset kia).

**Mục tiêu chỉ số:** mục tiêu mAP và ngưỡng escape/overkill sẽ chốt sau khi có baseline, tham chiếu các công trình cùng dataset. Định hướng ban đầu là escape rate ≤ 2% với overkill ở mức chấp nhận được.

## 8. Công nghệ

| Mảng | Công cụ |
|---|---|
| Ngôn ngữ, nền tảng | Python, PyTorch, OpenCV |
| Detection | Ultralytics YOLOv8 |
| Anomaly | Anomalib (PatchCore) |
| Gán nhãn | CVAT hoặc Label Studio |
| Augmentation | Albumentations |
| Theo dõi thí nghiệm | Weights & Biases hoặc MLflow |
| Tối ưu | ONNX, ONNX Runtime, OpenVINO (INT8 là phần mở rộng) |
| Dịch vụ | FastAPI, Docker, SQLite |
| Dashboard | Streamlit |
| Chất lượng code | Pytest, GitHub Actions, pre-commit |
| Huấn luyện | Kaggle / Colab (GPU) |

## 9. Sản phẩm bàn giao

1. Repo GitHub sạch, tái lập được bằng `make data`, `make train`, `make eval`, `make serve`.
2. Bộ dữ liệu tự chụp kèm hướng dẫn gán nhãn (hoặc mẫu và script nếu không công khai được toàn bộ).
3. Báo cáo thí nghiệm: bảng baseline, bảng ablation, phân tích lỗi.
4. Báo cáo QC: escape/overkill, đường đánh đổi ngưỡng, kiểm thử độ bền.
5. Bảng hiệu năng trước và sau tối ưu.
6. API + Docker + dashboard chạy được bằng `docker compose up`.
7. README (bài toán, kiến trúc, kết quả, hạn chế), model card, video demo 60–90 giây.
8. Một bài viết ngắn về bài học rút ra.

## 10. Điểm khác biệt so với dự án sinh viên thông thường

- Có **dữ liệu tự thu thập** trong điều kiện ánh sáng được kiểm soát, kèm quy trình gán nhãn.
- Trả lời được **"OK là gì?"** nhờ anomaly detection, thay vì dùng ngưỡng tin cậy của bộ phân loại.
- Báo cáo theo **ngôn ngữ QC** (escape, overkill) chứ không chỉ accuracy.
- Có **phân tích độ bền** và kiểm thử khác miền.
- Có **đo đạc hiệu năng** và triển khai thành dịch vụ thật.
- Có câu chuyện cải tiến từ dự án trước (baseline phân loại + Grad-CAM++ so với detection + anomaly).

## 11. Rủi ro và cách giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Dataset công khai không có nhãn bounding box như kỳ vọng | Kiểm tra ngay từ đầu; nếu thiếu thì tự gán nhãn một phần hoặc chuyển sang dataset có nhãn. |
| Dữ liệu tự chụp ít, nhãn không nhất quán | Có hướng dẫn gán nhãn, gán lại một phần để đo nhất quán, giữ đúng phạm vi vật thể nhỏ. |
| Lỗi nhỏ khó phát hiện | Tăng kích thước ảnh đầu vào, thử SAHI. |
| Mô hình sụt mạnh khi đổi ánh sáng | Thu thêm dữ liệu ở điều kiện đó, tăng cường dữ liệu có chọn lọc, báo cáo trung thực mức sụt. |
| Thiếu GPU | Dùng Kaggle/Colab, ưu tiên model nhỏ (n/s). |
| Phạm vi phình to | Giữ phần mở rộng ở dạng tùy chọn, ưu tiên dữ liệu, phân tích lỗi và chỉ số QC. |

## 12. Tiêu chí hoàn thành

- Tái tạo được toàn bộ kết quả từ repo bằng các lệnh đã ghi.
- Mọi con số trong README và CV đều có nguồn từ thí nghiệm đã log.
- Có bảng so sánh với baseline và ít nhất 3 thay đổi trong ablation.
- Có số đo QC và độ bền trên dữ liệu tự chụp.
- Dịch vụ chạy được bằng Docker, có test tự động.
- Có video demo và README đầy đủ phần hạn chế.

## 13. Mẫu ghi trên CV (đã điền số thực nghiệm từ dự án)

> **SteelVision – Hệ thống kiểm tra lỗi bề mặt kim loại** | Python, PyTorch, YOLOv8, Anomalib (PatchCore), ONNX Runtime, FastAPI, Docker
> - Xây dựng pipeline kết hợp Object Detection (`YOLOv8n`) và Unsupervised Anomaly Detection (`PatchCore`) kèm cổng kiểm tra chất lượng ảnh (Image Quality Gate) quyết định **OK/NG** trên `4,420` ảnh công nghiệp (`NEU-DET`, `GC10-DET`, `MVTec AD metal_nut`) sau khử trùng lặp `pHash`; thiết kế giao thức thu thập & kiểm định nhất quán nhãn cho bộ dữ liệu tự chụp trong `3` điều kiện ánh sáng.
> - Đạt **mAP@50 = `0.784`** trên `NEU-DET` (`+1.95%` nhờ tắt Mosaic), **Image AUROC = `99.80%`** / **Pixel AUROC = `98.69%`** trên `MVTec metal_nut`; tối ưu ngưỡng theo mục tiêu QC giúp giảm **escape rate từ `9.40%` xuống `0.75%`** trên `NEU-DET` và đạt **escape rate `0.00%`** (overkill `0.00%` trên tập test, `4.55%` toàn tập) trên `metal_nut` kể cả khi giả lập lỗi mới chưa từng gặp.
> - Tối ưu suy luận bằng **ONNX Runtime + Early-Exit OR-Fusion**, giảm độ trễ CPU (`p50`) từ **`279.60 ms` xuống `63.78 ms` (`4.38×` nhanh hơn, `15.68 FPS`)** và giảm **51.7% RAM (`1,297 MB → 626 MB`)**; đóng gói dịch vụ FastAPI (`/inspect`), lưu vết SQLite, Streamlit dashboard và Docker Compose (`39/39` unit test pass).
