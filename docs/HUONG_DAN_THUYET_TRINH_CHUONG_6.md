# 🎓 CẨM NANG THUYẾT TRÌNH & BẢO VỆ ĐỒ ÁN HỌC MÁY (CHƯƠNG 6)
> **Đề tài:** Hệ Thống Nhận Diện Ký Tự & Văn Bản Viết Tay Thời Gian Thực (HandwritingAI Studio)  
> **Môn học:** Trí Tuệ Nhân Tạo — Chương 6: Học Máy (Machine Learning)  
> **Tác giả / Nhóm thực hiện:** Trương Duy Cường & Nhóm Sinh Viên  

---

## 📌 PHẦN 1: ÁNH XẠ TRỰC TIẾP GIỮA LÝ THUYẾT CHƯƠNG 6 VÀ MÃ NGUỒN (CODE MAPPING)

Khi giảng viên chỉ tay vào bất kỳ phần nào trên màn hình hoặc hỏi: *"Thuật toán này nằm ở đâu trong bài giảng?"*, bạn chỉ cần mở bảng ánh xạ này:

| Khái Niệm Lý Thuyết Chương 6 | Vị Trí Trong Mã Nguồn Dự Án | Ý Nghĩa Kỹ Thuật Trả Lời Giảng Viên |
| :--- | :--- | :--- |
| **Định nghĩa $(T, P, E)$ của Tom Mitchell** | [README.md](file:///C:/HocC/HandwritingAI/README.md) & [config.yaml](file:///C:/HocC/HandwritingAI/configs/config.yaml) | - $T$: Phân loại 62 lớp ký tự từ ảnh $28\times 28$<br>- $P$: Accuracy, Precision, Recall, Macro F1<br>- $E$: Tập dữ liệu gán nhãn EMNIST ByClass |
| **Mạng Nơ-ron Nhiều Tầng (Multilayer Perceptron - ANN)** | [onnx_engine.py](file:///C:/HocC/HandwritingAI/src/onnx_engine.py#L18-L38) (`SimpleEMNISTCNN`) | Mạng nơ-ron học sâu gồm tầng Conv2D (trích xuất đặc trưng cạnh/góc), MaxPool2D (giảm chiều), và Dense Layers (tính tổng trọng số $z = \mathbf{w}^T \mathbf{x} + b$) |
| **Hàm kích hoạt ReLU (Non-linear Activation)** | [onnx_engine.py](file:///C:/HocC/HandwritingAI/src/onnx_engine.py#L31-L36) | $f(z) = \max(0, z)$ giúp mạng học được các quan hệ phi tuyến phức tạp của nét chữ cong mà không bị bão hòa gradient |
| **Hàm kích hoạt Softmax (Tầng ra cùng)** | [onnx_engine.py](file:///C:/HocC/HandwritingAI/src/onnx_engine.py#L90) | Chuyển đổi vector Logits thành phân bố xác suất $P(y=c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum e^{z_j}}$ với $\sum P = 1.0$ |
| **Hàm mất mát Cross-Entropy Loss** | [train.py](file:///C:/HocC/HandwritingAI/src/train.py#L68) | Đo lường độ sai lệch phân bố giữa nhãn thực tế $y$ và dự đoán $\hat{y}$: $\mathcal{L} = -\sum y_i \log(\hat{y}_i)$ |
| **Thuật toán Tối ưu hóa Gradient Descent** | [train.py](file:///C:/HocC/HandwritingAI/src/train.py#L69-L70) | Cập nhật trọng số theo hướng ngược chiều gradient: Sử dụng **AdamW** kết hợp **Cosine Annealing Learning Rate** |
| **Kỹ thuật Chống Học Vẹt (Overfitting)** | [train.py](file:///C:/HocC/HandwritingAI/src/train.py#L42-L70) | Tích hợp 4 lớp bảo vệ: **Dropout** (ngắt kết nối ngẫu nhiên 25-50%), **Weight Decay** ($L_2$ Regularization), **Batch Normalization**, và **Data Augmentation** |
| **Thuật toán Tiền xử lý Trọng tâm (NIST Centering)** | [nist_preprocessor.py](file:///C:/HocC/HandwritingAI/src/nist_preprocessor.py#L78-L96) | Tính mô-men ảnh để đưa trọng tâm nét vẽ về tọa độ tâm $(14, 14)$, giúp mô hình bất biến với kích thước và vị trí vẽ |

---

## 🎤 PHẦN 2: KỊCH BẢN PHÂN CHIA THUYẾT TRÌNH NHÓM (4 THÀNH VIÊN)

### 👤 Thành viên 1: Đặt Vấn Đề & Khung Lý Thuyết (Thời lượng: ~2 phút)
* Giới thiệu bài toán: Nhận diện chữ viết tay – ví dụ mẫu số 1 trong Chương 6.
* Trình bày bộ ba $(T, P, E)$ theo giáo trình Tom Mitchell.
* Nêu sự khác biệt giữa MNIST (10 số đơn giản) và **EMNIST ByClass (62 lớp)** gồm số, chữ hoa và chữ thường.

### 👤 Thành viên 2: Thuật Toán Tiền Xử Lý NIST & Bóc Tách (Thời lượng: ~3 phút)
* Bật Tab 3 trên Web App: **"🔬 Bóc Tách Thuật Toán Trực Quan"**.
* Trình bày công thức mô-men ảnh: $c_x = \frac{\sum x \cdot I}{\sum I}, c_y = \frac{\sum y \cdot I}{\sum I}$.
* Giải thích tại sao phương pháp cũ chỉ *resize* thô thì người dùng vẽ lệch góc sẽ sai, còn thuật toán căn trọng tâm giúp tịnh tiến nét chữ chuẩn xác về $(14, 14)$.

### 👤 Thành viên 3: Kiến Trúc Mạng Nơ-ron & Quá Trình Huấn Luyện (Thời lượng: ~3 phút)
* Giải thích kiến trúc mạng nơ-ron: Tầng tích chập Conv2D, hàm kích hoạt ReLU, Dropout chống overfitting, tầng Dense và tầng Softmax.
* Trình bày cơ chế lan truyền ngược (Backpropagation) và thuật toán tối ưu Gradient Descent (AdamW).
* Phân tích tại sao mô hình đạt tốc độ suy luận siêu tốc $< 1.5\text{ ms}$ trên GPU RTX 2050.

### 👤 Thành viên 4: Demo Trực Tiếp & Kết Luận (Thời lượng: ~2 phút)
* Trình chiếu trực tiếp trên Web App tại `http://localhost:8502`:
  * Mời giảng viên hoặc các bạn bên dưới chọn chữ cái/chữ số để vẽ trực tiếp trên Canvas.
  * Mở Tab 2 tải ảnh một từ viết tay để cho thấy hệ thống bóc tách từng chữ cái tự động (OCR).
* Tóm tắt bài học kinh nghiệm và kết thúc bài báo cáo.

---

## ❓ PHẦN 3: BỘ 6 CÂU HỎI VẤN ĐÁP PHẢN BIỆN CHUẨN ĐIỂM 10

#### Câu 1: Tại sao nhóm không dùng mạng MLP (Multilayer Perceptron) thông thường mà phải dùng CNN?
> **👉 Trả lời mẫu:**  
> *"Dạ thưa Thầy/Cô, nếu dùng MLP, chúng ta phải duỗi phẳng ảnh $28\times 28$ thành một vector 784 chiều. Điều này làm mất hoàn toàn mối tương quan không gian 2 chiều giữa các điểm ảnh lân cận và tạo ra số lượng tham số khổng lồ dễ gây Overfitting. Trong khi đó, mạng CNN sử dụng các bộ lọc tích chập (Filters) quét qua ảnh, có đặc tính chia sẻ trọng số (Weight Sharing) và tính bất biến tịnh tiến (Translation Invariance), giúp học được các đặc trưng từ nét cạnh, góc đến nét cong giống hệt cơ chế thị giác con người."*

#### Câu 2: Giả sử người dùng vẽ một nét chữ quá mỏng hoặc quá dày thì hệ thống xử lý thế nào?
> **👉 Trả lời mẫu:**  
> *"Dạ, trong file [nist_preprocessor.py](file:///C:/HocC/HandwritingAI/src/nist_preprocessor.py), nhóm em đã tích hợp bộ lọc Gaussian Blur $3\times 3$ để làm mượt cạnh, sau đó áp dụng phép co giãn chuẩn tỷ lệ Aspect Ratio vào khung $20\times 20$. Quá trình này tự động chuẩn hóa độ dày nét chữ về tương quan chuẩn của tập EMNIST trước khi đưa vào mạng nơ-ron."*

#### Câu 3: Hàm Softmax và hàm Sigmoid khác nhau như thế nào trong bài toán này?
> **👉 Trả lời mẫu:**  
> *"Dạ, hàm Sigmoid chỉ áp dụng cho bài toán phân loại nhị phân 2 lớp (hoặc đa nhãn độc lập). Còn bài toán EMNIST ByClass là phân loại đa lớp loại trừ lẫn nhau (Multi-class Single-label với 62 lớp). Hàm Softmax có tính chất chuẩn hóa mẫu số bằng tổng $\sum e^{z_j}$, đảm bảo tổng xác suất của tất cả 62 ký tự luôn bằng đúng 100%, giúp ta dễ dàng chọn ra ký tự có xác suất cực đại."*

#### Câu 4: Làm sao nhóm biết mô hình không bị hiện tượng Học vẹt (Overfitting)?
> **👉 Trả lời mẫu:**  
> *"Dạ, nhóm theo dõi sát sao biểu đồ Loss giữa 2 tập Train và Validation qua từng Epoch: cả 2 đường đều giảm đều và bám sát nhau. Đồng thời nhóm áp dụng Dropout 25% ở các khối tích chập và 50% ở tầng Dense, kết hợp Weight Decay $10^{-4}$ để phạt các trọng số quá lớn."*

#### Câu 5: Tại sao có một số chữ cái như 'O' và số '0', hoặc chữ 'c' thường và 'C' hoa dễ bị nhầm?
> **👉 Trả lời mẫu:**  
> *"Dạ, đây là hiện tượng đẳng hình (homoglyphs) kinh điển trong bài toán EMNIST 62 lớp. Khi đứng độc lập không có dòng kẻ làm chuẩn tỷ lệ chiều cao, ngay cả mắt người cũng không thể phân biệt được 'c' thường hay 'C' hoa. Hệ thống của nhóm thể hiện rất trung thực điều này qua bảng Softmax Top-5: khi vẽ chữ 'O', cả 'O' và số '0' đều có xác suất cao ở vị trí Top-1 và Top-2."*

#### Câu 6: Dự án này có thể mở rộng ứng dụng vào thực tế như thế nào?
> **👉 Trả lời mẫu:**  
> *"Dạ, với độ trễ suy luận dưới 1.5 ms và module Segmentation đã xây dựng ở Tab 2, dự án có thể tích hợp trực tiếp vào: 1) Ứng dụng số hóa tài liệu, biểu mẫu hành chính; 2) Hệ thống chấm điểm bài thi trắc nghiệm và tự luận; 3) Ứng dụng tập viết chữ số thông minh trên máy tính bảng dành cho trẻ em."*
