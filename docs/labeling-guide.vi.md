# Hướng dẫn gán nhãn dữ liệu SteelVision (Tiếng Việt)

> Tài liệu này là chuẩn duy nhất khi gán nhãn trên CVAT hoặc Label Studio.
> Mọi quyết định nhãn gây tranh cãi phải quay về đây; nếu chưa có quy tắc,
> ghi vào mục "Trường hợp mơ hồ" ở cuối và thảo luận trước khi gán tiếp.

## 1. Nguyên tắc chung

- Mỗi **ảnh** là một **sản phẩm**. Ảnh hoàn toàn không lỗi → không vẽ box nào
  (label file rỗng, nhưng vẫn giữ ảnh: đây là mẫu **OK**).
- Mỗi lỗi một **bounding box** ôm **chặt** biên lỗi (sai lệch cho phép ≤ 2 px
  ở ảnh gốc). Không ôm rộng sang vùng kim loại lành.
- Hai lỗi riêng biệt chạm nhau nhưng nguồn gốc khác nhau → vẽ **2 box**; nếu
  là cùng một vệt lỗi liền mạch → **1 box**.
- Kích thước tối thiểu: box phải ≥ **8 px** cạnh nhỏ nhất ở ảnh gốc. Box nhỏ
  hơn: chụp ảnh macro cận cảnh thay vì cố gán nhãn.
- Ảnh mờ/nhòe đến mức không xác định được lỗi có chắc là lỗi không → **bỏ
  ảnh khỏi tập có nhãn**, ghi vào `manifest.csv` cột `notes`.

## 2. Định nghĩa lớp

### 2.1. NEU-DET (6 lớp — tham chiếu để chuyển đổi, KHÔNG gán lại)

`crazing` (nứt mạng nhện), `inclusion` (lẫn tạp), `patches` (mảng), 
`pitted_surface` (bề mặt rỗ), `rolled-in_scale` (vảy cán), `scratches` 
(vết trầy).

### 2.2. Dữ liệu tự chụp (lớp khởi điểm, chỉnh sau khi chụp 50 ảnh đầu)

| Lớp | Định nghĩa | Ví dụ điển hình |
|---|---|---|
| `scratch` | Vết trầy xước tuyến tính do ma sát | vệt cứng, có hướng, thường dài |
| `rust` | Gỉ/ố màu hoá học trên bề mặt | mảng nâu/cam loang |
| `pit` | Rỗ/hố nhỏ gây bằng ăn mòn hoặc va chạm điểm | chấm sâu, có bóng đổ |
| `stain` | Vết bẩn/dầu/nước, không phá hủy bề mặt | loang trơn, có thể lau |
| `dent` | Móp/lõm hình học | biến dạng phản chiếu sáng |
| `weld_seam` | Đường hàn (lỗi nếu không theo chuẩn mối hàn) | dải sần đều |

> `stain` vẫn được gán nhãn vì ảnh hưởng thị giác của QC; ghi rõ trong model
> card rằng đây là lớp "giả lỗi" để huấn luyện mô hình phân biệt.

## 3. Quy tắc vẽ box theo từng lớp

- `scratch` / `weld_seam`: với vệt dài, được phép vẽ box đứng/nghiêng ôm toàn
  vệt; KHÔNG cắt vệt thành nhiều box nếu nó liền mạch.
- `rust` / `stain`: ôm vùng đổi màu bao gồm rìa loang nhẹ, không ôm cả bóng
  đổ của đèn.
- `pit`: chụm nhiều chấm trong bán kính ≤ 10 px → 1 box "cụm rỗ"; các chấm
  rời xa nhau (> 20 px) → box riêng.
- `dent`: ranh giới theo mép biến dạng phản chiếu, thường cần zoom 200% để
  xác định.

## 4. Quy trình trên CVAT

1. Tạo project `steelvision`, định nghĩa label theo mục 2.2 (đúng thứ tự chữ
   cái — thứ tự này cố định `class_id` khi xuất).
2. Task theo **đợt chụp** (một đợt = một ngày + một bộ điều kiện sáng), tên:
   `self_YYYYMMDD_<dieu_kien_sang>`.
3. Gán nhãn theo lô 100 ảnh, nghỉ kiểm tra lại hướng dẫn mỗi lô.
4. Export **YOLO 1.1** (zip chứa `obj.data`, `train.txt`, images, labels).
5. Đặt vào `data/self/raw/<đợt>/` và chạy script nhập (Phase 5).

## 5. Kiểm soát chất lượng nhãn

- **Gán lại 10%**: chọn ngẫu nhiên 10% ảnh sau khi gán xong, gán lại từ đầu
  (không nhìn nhãn cũ) trên CVAT job riêng.
- Đo nhất quán: mean IoU giữa 2 lần gán cùng lớp (mục tiêu ≥ 0.75) và F1
  nhãn cấp ảnh (mục tiêu ≥ 0.90). Số liệu vào `docs/reports/`.
- Ảnh 2 lần gán lệch lớn → chụp màn hình, đối chiếu định nghĩa, cập nhật
  hướng dẫn nếu quy tắc thiếu.

## 6. Trường hợp mơ hồ (cập nhật liên tục)

| Tình huống | Quyết định |
|---|---|
| Vết trầy nhưng không chắc là lỗi hay phản chiếu | đổi góc ảnh gốc; nếu vẫn mơ hồ → đánh dấu `skip` |
| Lỗi nằm sát mép ảnh, bị cắt một phần | gán nhãn phần nhìn thấy, ghi `notes = edge` |
| Hai lớp chồng nhau (rỉ trên vết trầy) | gán lớp **gốc** (`scratch`), ghi `notes = rust overlay` |
