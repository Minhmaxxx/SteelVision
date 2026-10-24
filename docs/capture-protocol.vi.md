# Quy trình thu thập dữ liệu tự chụp — SteelVision (Tiếng Việt)

> Mục tiêu: 300–500 ảnh, **≥ 40% ảnh OK**, 2–3 điều kiện ánh sáng, truy vết
> được từng ảnh về mẫu và điều kiện chụp. Bắt đầu được ngay khi có: một
> điện thoại/ camera, 3 nguồn sáng, một số mẫu tấm kim loại.

## 1. Thiết bị

| Thành phần | Yêu cầu tối thiểu | Ghi chú |
|---|---|---|
| Camera | Điện thoại ≥ 12 MP, tắt HDR/tự làm đẹp | Khoá ISO,WB nếu được; ghi lại thông số |
| Giá đỡ | Tripod/hộp cố định, camera **vuông góc** bề mặt | Khoảng cách chụp cố định 25–35 cm |
| Ánh sáng | 3 cấu hình: (A) đèn trắng khuếch tán 2 bên 45°, (B) 1 đèn trên đầu, (C) ánh sáng phòng thường | Đúng 3 điều kiện này cho mọi mẫu |
| Nền | Tấm nền tối, không phản chiếu | Tránh quần áo/người vào khung hình |

## 2. Mẫu

- 30–50 tấm kim loại/thép phẳng (15×15 đến 20×30 cm), bao gồm:
  - ít nhất **15 tấm OK** (bề mặt lành, có thể là cùng một tấm chụp nhiều
    vị trí khác nhau — nhưng mỗi ảnh OK phải ở vị trí/khung khác nhau),
  - tấm có lỗi thật: trầy, rỉ, rỗ, dằn cán, vết dầu, móp.
- Nếu lỗi tự nhiên quá ít: tạo lỗi có kiểm soát (chà nhám các cấp, nhỏ giọt
  nước để ố, dùng đinh gõ nhẹ tạo móp) — **ghi lại cách tạo** vào manifest.

## 3. Kịch bản chụp

Với mỗi mẫu và mỗi vị trí khung hình:

1. Đặt mẫu cố định trên nền, đánh dấu vị trí bằng băng dính.
2. Chụp 3 ảnh = 3 điều kiện sáng A/B/C, không dịch chuyển máy.
3. Lặp 3–5 vị trí/tấm (đổi vị trí mẫu, không đổi máy).
4. Mỗi 50 ảnh chụp 1 ảnh thẻ màu (color checker hoặc giấy trắng) để chuẩn
   hoá màu sau này.

Quy mô mục tiêu: 40–60 mẫu vị trí × 3 sáng × ~2 lượt ≈ **300–500 ảnh**;
tỷ lệ OK tự nhiên sẽ ~40–50% nếu dành riêng các vị trí chỉ chụp tấm lành.

## 4. Đặt tên và manifest

- Tên file giữ nguyên tên gốc của máy; **không** đổi tên thủ công.
- Ngay sau mỗi đợt chụp, điền `manifest.csv`:

```csv
file,session,panel,position,lighting,is_defective,label_status,notes
IMG_0001.jpg,2026-10-25-P1,P07,pos2,A,unknown,pending,
```

| Cột | Ý nghĩa |
|---|---|
| `session` | ngày + đợt (khớp tên task CVAT) |
| `panel` | mã tấm mẫu, in nhãn dán lên tấm |
| `position` | số vị trí khung hình trên tấm |
| `lighting` | A / B / C |
| `is_defective` | `ok` / `defective` / `unknown` (điền sau khi xem) |
| `label_status` | `pending` → `labeled` → `qc-passed` |

## 5. Sau khi chụp

1. Sao chép thẻ nhớ vào `data/self/raw/<session>/` (không nén lại, không
   lọc ảnh trong thẻ).
2. Lọc loại ảnh lỗi kỹ thuật (run tay, mép máy, thiếu sáng nặng) — ghi lý do
   vào `notes`, không xoá file gốc.
3. Điền `is_defective` cho toàn bộ (xem nhanh, chưa cần box).
4. Tạo task CVAT theo `session` và gán nhãn theo `docs/labeling-guide.vi.md`.
5. Chạy gán lại 10% theo đúng mục 5 của hướng dẫn gán nhãn.

## 6. Điều kiện dừng / mở rộng

- Dừng chụp khi: ≥ 300 ảnh đã gán nhãn QC-pass **và** mỗi lớp lỗi có ≥ 40 box
  **và** mỗi điều kiện sáng có ≥ 15 ảnh OK.
- Chỉ mở rộng thêm điều kiện mới (ánh sáng gắt, góc nghiêng…) sau khi baseline
  Phase 2 đã chạy — dùng để đo độ bền, không trộn vào train ngay.
