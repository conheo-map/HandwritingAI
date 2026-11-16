# Enterprise IDP & Document AI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng hệ thống Xử lý Tài liệu Thông minh Doanh nghiệp (Enterprise IDP) chuyên sâu cho Văn bản Hợp đồng và Giấy tờ tùy thân (CCCD gắn chip / Hộ chiếu) đạt chuẩn 5 trụ cột công nghệ.

**Architecture:** Kiến trúc 2 tầng (Dual-Engine Enterprise Architecture). Tầng 1 đảm nhiệm tiền xử lý hình học (4-point perspective warp, khử lóa flash LAB-CLAHE) và trích xuất OCR tiếng Việt đa ngữ. Tầng 2 đảm nhiệm nhận thức ngữ nghĩa: phân loại tài liệu, bóc tách Key-Value sang JSON, kiểm toán thuật toán Checksum 12 số CCCD chuẩn Bộ Công An, và phát hiện con dấu / chữ ký hợp đồng.

**Tech Stack:** Python 3.13, PyTorch 2.6 CUDA 12.4 (RTX 2050), OpenCV (cv2), NumPy, Pandas, Scipy, Streamlit.

**Spec:** [docs/DESIGN_ENTERPRISE_IDP.md](file:///C:/HocC/HandwritingAI/docs/DESIGN_ENTERPRISE_IDP.md)

## Global Constraints
- Nền tảng: Chạy 100% On-Premise trên máy trạm Windows, GPU RTX 2050, bảo mật tuyệt đối không truyền dữ liệu ra internet.
- Độ trễ toàn trình: $< 800\text{ ms}$ cho một mặt thẻ CCCD, $< 1.5\text{ s}$ cho một trang hợp đồng.
- Độ chính xác tiếng Việt: Hỗ trợ đầy đủ bộ gõ Unicode tiếng Việt có dấu thanh và các nguyên âm đặc thù (`ă, â, đ, ê, ô, ơ, ư`).

## Review Focus
1. Thẻ CCCD bị chụp nghiêng góc hoặc lóa sáng do đèn flash: Thuật toán 4-point transform và LAB-CLAHE phải nắn thẳng và phục hồi vùng lóa.
2. Số CCCD bị làm giả hoặc sai quy chuẩn: Thuật toán Checksum phải phát hiện mâu thuẫn giữa ngày sinh, giới tính và mã định danh 12 số.
3. Chữ viết tay điền thêm trên hợp đồng: Module OCR phải phân tách và nhận diện được cả chữ in lẫn chữ viết tay.
4. Hợp đồng thiếu chữ ký hoặc con dấu đỏ: Hệ thống phải cảnh báo rõ ràng trường `has_stamp: false` hoặc `has_signature: false`.
5. Đầu ra có cấu trúc: Trả về JSON sạch, chuẩn hóa kiểu dữ liệu số, ngày tháng chuẩn ISO-8601.

---

### Task 1: Module Tiền Xử Lý Hình Học & Khử Lóa Flash (`src/idp_preprocessor.py`)

**Files:**
- Create: `C:/HocC/HandwritingAI/src/idp_preprocessor.py`
- Test: `C:/HocC/HandwritingAI/tests/test_idp_preprocessor.py`

- [ ] **Step 1: Viết test case kiểm thử tiền xử lý hình học**
  - Test phát hiện 4 góc tài liệu và nắn phối cảnh `perspective_transform`.
  - Test khử lóa flash qua kênh màu `LAB + CLAHE`.
  - Test tự động xoay góc trang thẳng `auto_deskew`.
- [ ] **Step 2: Chạy test để xác nhận kiểm thử thất bại (TDD Red)**
- [ ] **Step 3: Cài đặt mã nguồn `idp_preprocessor.py`**
  - Cài đặt lớp `DocumentPreprocessor`.
  - Hàm `find_document_corners(image)`.
  - Hàm `four_point_transform(image, pts)`.
  - Hàm `remove_glare_and_shadows(image)`.
  - Hàm `enhance_contrast(image)`.
- [ ] **Step 4: Chạy test xác nhận toàn bộ test case vượt qua (TDD Green)**
- [ ] **Step 5: Commit mã nguồn Task 1 vào Git**

---

### Task 2: Module Động Cơ OCR Tiếng Việt & Bounding Box (`src/idp_ocr.py`)

**Files:**
- Create: `C:/HocC/HandwritingAI/src/idp_ocr.py`
- Test: `C:/HocC/HandwritingAI/tests/test_idp_ocr.py`

- [ ] **Step 1: Viết test case kiểm thử trích xuất văn bản tiếng Việt**
  - Test trích xuất văn bản có dấu: "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc".
  - Test trích xuất tọa độ Bounding Box 4 điểm của từng dòng.
  - Test tính độ tin cậy trung bình của kết quả nhận diện.
- [ ] **Step 2: Chạy test xác nhận thất bại**
- [ ] **Step 3: Cài đặt mã nguồn `idp_ocr.py`**
  - Cài đặt lớp `DocumentOCREngine`.
  - Tích hợp pipeline OCR chuyên sâu cho tài liệu tiếng Việt.
  - Hỗ trợ vẽ bọc Bounding Box lên ảnh gốc.
- [ ] **Step 4: Chạy test xác nhận thành công**
- [ ] **Step 5: Commit mã nguồn Task 2 vào Git**

---

### Task 3: Module Bóc Tách Giấy Tờ Tùy Thân & Thuật Toán Checksum CCCD (`src/idp_cccd_parser.py`)

**Files:**
- Create: `C:/HocC/HandwritingAI/src/idp_cccd_parser.py`
- Test: `C:/HocC/HandwritingAI/tests/test_idp_cccd_parser.py`

- [ ] **Step 1: Viết test case kiểm thử bóc tách CCCD và quy chuẩn Checksum**
  - Test trích xuất 8 trường thông tin mặt trước CCCD (Số CCCD, Họ tên, Ngày sinh, Giới tính, Quê quán, Nơi thường trú...).
  - Test thuật toán Checksum hợp lệ: Mã tỉnh `001` (Hà Nội), Giới tính Nam TK20 (`0`), Năm sinh 98 (`98`) $\rightarrow$ `001098xxxxxx`.
  - Test cảnh báo giả mạo: Sinh năm 2002 nhưng số CCCD lại có mã giới tính `0` (Sai thế kỷ).
- [ ] **Step 2: Chạy test xác nhận thất bại**
- [ ] **Step 3: Cài đặt mã nguồn `idp_cccd_parser.py`**
  - Cài đặt lớp `CCCDParser`.
  - Từ điển 63 mã tỉnh thành Việt Nam theo quy chuẩn Bộ Công An.
  - Thuật toán `validate_cccd_checksum(id_number, dob, gender)`.
  - Hàm `parse_cccd_front(ocr_results)`.
  - Xuất cấu trúc JSON chuẩn hóa.
- [ ] **Step 4: Chạy test xác nhận thành công**
- [ ] **Step 5: Commit mã nguồn Task 3 vào Git**

---

### Task 4: Module Bóc Tách Hợp Đồng, Điều Khoản & Phát Hiện Chữ Ký Con Dấu (`src/idp_contract_parser.py`)

**Files:**
- Create: `C:/HocC/HandwritingAI/src/idp_contract_parser.py`
- Test: `C:/HocC/HandwritingAI/tests/test_idp_contract_parser.py`

- [ ] **Step 1: Viết test case kiểm thử trích xuất hợp đồng pháp lý**
  - Test bóc tách Bên A, Bên B (Đại diện, MST, Địa chỉ).
  - Test trích xuất giá trị hợp đồng (bằng số và bằng chữ).
  - Test thuật toán phát hiện con dấu đỏ (Color Thresholding trong không gian HSV) và vùng chữ ký.
- [ ] **Step 2: Chạy test xác nhận thất bại**
- [ ] **Step 3: Cài đặt mã nguồn `idp_contract_parser.py`**
  - Cài đặt lớp `ContractParser`.
  - Bóc tách thực thể hợp đồng kinh tế.
  - Hàm `detect_stamps_and_signatures(image)`.
- [ ] **Step 4: Chạy test xác nhận thành công**
- [ ] **Step 5: Commit mã nguồn Task 4 vào Git**

---

### Task 5: Tích Hợp Giao Diện Web Studio Đa Năng (`app.py`)

**Files:**
- Modify: `C:/HocC/HandwritingAI/app.py`

- [ ] **Step 1: Thiết kế giao diện Dashboard Enterprise IDP**
  - Bổ sung Tab: `📄 Giám Định Hợp Đồng & Giấy Tờ Tùy Thân (Enterprise IDP)`.
  - Hỗ trợ chế độ tải tài liệu: Ảnh chụp thẻ CCCD, Hộ chiếu, hoặc Hợp đồng kinh tế.
  - Nút bấm 1-Click: "Tự Động Nắn Phối Cảnh & Khử Lóa Flash".
- [ ] **Step 2: Hiển thị kết quả trực quan hai cột**
  - Cột 1: Ảnh tài liệu đã nắn thẳng, bôi màu trực quan Bounding Box tương ứng từng trường thông tin.
  - Cột 2: Bảng hồ sơ dữ liệu JSON đã bóc tách, bảng điều khoản hợp đồng dạng Markdown, và huy hiệu cảnh báo Checksum Bộ Công An (Hợp lệ / Nghi vấn sai lệch).
  - Nút bấm xuất dữ liệu: "Tải File JSON Kết Quả" và "Sao Chép Vào Clipboard".
- [ ] **Step 3: Khởi động kiểm thử giao diện tại `http://localhost:8502`**
- [ ] **Step 4: Commit mã nguồn Task 5 vào Git**

---

### Task 6: Kiểm Thử Toàn Trình (End-to-End Verification) & Bàn Giao

- [ ] **Step 1: Chạy toàn bộ test suite (`pytest tests/`)**
- [ ] **Step 2: Thử nghiệm thực tế với ảnh mẫu CCCD và Hợp đồng**
- [ ] **Step 3: Cập nhật tài liệu thuyết trình Chương 6 và README**
- [ ] **Step 4: Đóng gói commit Git hoàn chỉnh**
