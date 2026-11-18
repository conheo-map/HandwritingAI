# CẨM NANG THUYẾT TRÌNH VÀ BẢO VỆ ĐỒ ÁN HỌC MÁY (CHƯƠNG 6)

> **Đề tài:** Hệ Thống Nhận Diện Ký Tự và Từ Ngữ Viết Tay Thời Gian Thực (HandwritingAI)  
> **Môn học:** Trí Tuệ Nhân Tạo — Chương 6: Học Máy (Machine Learning)  
> **Nhóm thực hiện:** Nhóm Sinh Viên  

---

## PHẦN 1: ÁNH XẠ TRỰC TIẾP GIỮA LÝ THUYẾT CHƯƠNG 6 VÀ MÃ NGUỒN (CODE MAPPING)

Bảng ánh xạ các khái niệm lý thuyết cốt lõi trong Chương 6 vào kiến trúc mã nguồn của dự án:

| Khái Niệm Lý Thuyết Chương 6 | Vị Trí Trong Mã Nguồn Dự Án | Ý Nghĩa Kỹ Thuật Trả Lời Giảng Viên |
| :--- | :--- | :--- |
| **Định nghĩa (T, P, E) của Tom Mitchell** | [README.md](file:///C:/HocC/HandwritingAI/README.md) & [config.yaml](file:///C:/HocC/HandwritingAI/configs/config.yaml) | - T: Nhận diện ký tự và từ viết tay thời gian thực<br>- P: Accuracy, Word Recognition Rate, Latency<br>- E: EMNIST Balanced (47 lớp) và Chars74k Word Bank |
| **Mạng Nơ-ron Nhiều Tầng (Multilayer Perceptron - ANN / CNN)** | [modern_resnet.py](file:///C:/HocC/HandwritingAI/src/modern_resnet.py) | Mạng tích chập sâu kết hợp Residual Connections (Skip Connections) và SE Attention Block giúp trích xuất đặc trưng hình thái chữ viết tay |
| **Mạng Tái Tụ Tuần Tự (Recurrent Neural Network - BiLSTM)** | [crnn_model.py](file:///C:/HocC/HandwritingAI/src/crnn_model.py) | 2 tầng BiLSTM 2 chiều (hidden_size=256) nắm bắt ngữ cảnh nét bút liền mạch cả 2 chiều trái sang phải và ngược lại |
| **Hàm mất mát CTC Loss (Connectionist Temporal Classification)** | [crnn_model.py](file:///C:/HocC/HandwritingAI/src/crnn_model.py) | Giải quyết bài toán chuỗi thời gian không cần cắt phân đoạn (Segmentation-Free), vượt qua Nghịch lý Sayre |
| **Hàm kích hoạt ReLU / GELU** | [modern_resnet.py](file:///C:/HocC/HandwritingAI/src/modern_resnet.py) & [crnn_model.py](file:///C:/HocC/HandwritingAI/src/crnn_model.py) | Hàm kích hoạt phi tuyến giúp mô hình học các quan hệ nét bút uốn lượn phức tạp |
| **Hàm kích hoạt Softmax (Tầng đầu ra)** | [onnx_engine.py](file:///C:/HocC/HandwritingAI/src/onnx_engine.py) | Chuyển đổi vector Logits thành phân bố xác suất có tổng bằng 1.0 với tham số nhiệt độ Temperature T |
| **Hàm mất mát Cross-Entropy Loss** | [train.py](file:///C:/HocC/HandwritingAI/src/train.py) & [train_experts.py](file:///C:/HocC/HandwritingAI/src/train_experts.py) | Đo lường độ sai lệch phân bố giữa nhãn thực tế y và nhãn dự đoán |
| **Thuật toán Tối ưu hóa Gradient Descent** | [train.py](file:///C:/HocC/HandwritingAI/src/train.py) & [train_crnn.py](file:///C:/HocC/HandwritingAI/src/train_crnn.py) | Cập nhật trọng số bằng thuật toán AdamW kết hợp Cosine Annealing Learning Rate |
| **Kỹ thuật Chống Học Vẹt (Overfitting)** | [train.py](file:///C:/HocC/HandwritingAI/src/train.py) & [train_crnn.py](file:///C:/HocC/HandwritingAI/src/train_crnn.py) | Áp dụng Dropout (25-50%), Weight Decay, Data Augmentation và Elastic Distortion |
| **Chuẩn hóa Trọng tâm Điểm ảnh (NIST Centering)** | [nist_preprocessor.py](file:///C:/HocC/HandwritingAI/src/nist_preprocessor.py) | Sử dụng mô-men ảnh đưa trọng tâm về (14, 14), giúp mô hình bất biến với kích thước và vị trí vẽ |
| **Phân tích Hình thái học Liên thông (Morphological Refinement)** | [nist_preprocessor.py](file:///C:/HocC/HandwritingAI/src/nist_preprocessor.py) | Dùng Connected Components phát hiện cấu trúc dấu chấm rời để khôi phục chữ 'i'/'j' thường bị NIST gộp lớp |

---

## PHẦN 2: KỊCH BẢN PHÂN CHIA THUYẾT TRÌNH NHÓM (4 THÀNH VIÊN)

### Thành viên 1: Đặt Vấn Đề và Cơ Sở Lý Thuyết (Thời lượng: ~2 phút)
* Giới thiệu bài toán: Nhận diện chữ viết tay theo thời gian thực – ví dụ điển hình trong Chương 6: Học Máy.
* Trình bày bộ ba (T, P, E) theo định nghĩa của Tom Mitchell.
* Nêu thách thức thực tế: Chữ viết tay tự do có nét nghiêng, nét uốn lượn, và đặc biệt là hiện tượng viết liền nét (Cursive Writing).

### Thành viên 2: Thuật Toán Tiền Xử Lý Hình Thái Học và NIST (Thời lượng: ~3 phút)
* Trình bày công thức mô-men ảnh NIST Center-of-Mass: c_x = sum(x * I) / sum(I), c_y = sum(y * I) / sum(I).
* Giải thích tại sao phương pháp resize thô truyền thống bị sai lệch góc, trong khi thuật toán căn trọng tâm đưa nét vẽ chuẩn xác về (14, 14).
* Trình bày thuật toán Connected Components phát hiện dấu chấm phía trên thân chữ để giải quyết quy chuẩn gộp lớp EMNIST Balanced (chữ 'i' thường vs 'I' in hoa).

### Thành viên 3: Kiến Trúc Mạng Nơ-ron và Giải Pháp Nghịch Lý Sayre (Thời lượng: ~3 phút)
* Phân tích Nghịch lý Sayre (Sayre's Paradox): Để nhận diện từ cần cắt chữ, nhưng muốn cắt chữ chuẩn thì phải nhận diện được từ trước.
* Giới thiệu giải pháp đột phá: Kiến trúc CRNN (CNN + 2 tầng BiLSTM + CTC Loss) hoạt động hoàn toàn không cần cắt phân đoạn (Segmentation-Free).
* Trình bày quá trình huấn luyện: Tập từ vựng 10,000 từ, biến dạng đàn hồi (Elastic Distortion), Mixed Precision FP16 trên GPU NVIDIA RTX 2050, đạt Word Accuracy 96.83%.

### Thành viên 4: Thực Nghiệm Trực Quan và Kết Luận (Thời lượng: ~2 phút)
* Trình chiếu ứng dụng Streamlit tại http://localhost:8502:
  * Tab 1 (Bảng vẽ tương tác): Thử nghiệm vẽ ký tự đơn lẻ, chữ số, và chữ 'i' thường có dấu chấm.
  * Tab 2 (Nhận diện từ ngữ): Thử nghiệm viết từ liền nét như 'account', 'hello', 'balance' và kiểm chứng kết quả nhận diện chuỗi tức thì.
* Tóm tắt bài học kinh nghiệm và trả lời câu hỏi phản biện của hội đồng.

---

## PHẦN 3: BỘ CÂU HỎI VẤN ĐÁP PHẢN BIỆN CHUẨN

#### Câu 1: Tại sao nhóm không dùng mạng MLP (Multilayer Perceptron) thông thường mà phải dùng CNN và BiLSTM?
> **Trả lời:**  
> Nếu dùng MLP thông thường, ảnh ma trận 2 chiều phải bị duỗi phẳng thành vector 1 chiều, làm mất hoàn toàn mối tương quan không gian giữa các điểm ảnh lân cận và tạo ra lượng tham số khổng lồ dễ gây quá khớp (overfitting). Mạng CNN trích xuất đặc trưng không gian thông qua các bộ lọc tích chập có tính chia sẻ trọng số. Đối với từ ngữ viết liền nét, BiLSTM tiếp tục mô hình hóa tiến trình nét bút theo thời gian cả 2 chiều trái-phải và phải-trái.

#### Câu 2: Nghịch lý Sayre (Sayre's Paradox) là gì và nhóm đã giải quyết nó như thế nào?
> **Trả lời:**  
> Trong nhận diện chữ viết tay dạng từ (Word Recognition), Nghịch lý Sayre nêu rõ: Muốn nhận diện từng chữ cái thì phải bóc tách (cắt box) chúng ra trước; nhưng với chữ viết tay nối nét uốn lượn (như từ 'account' hay 'hi'), các nét chữ dính liền vào nhau khiến thuật toán cắt box hình học bị chia cắt sai nét. Muốn cắt chuẩn thì phải biết từ đó là gì trước. Nhóm giải quyết triệt để bằng kiến trúc CRNN + CTC: Không cắt ký tự (Segmentation-Free), mạng CNN nén ảnh thành chuỗi đặc trưng theo chiều ngang, BiLSTM nắm bắt ngữ cảnh, và CTC Loss tự động giải mã chuỗi ký tự.

#### Câu 3: Tại sao khi vẽ chữ 'i' thường có dấu chấm, mô hình ResNet EMNIST 47 lớp lại đoán ra 'I' in hoa, và nhóm xử lý ra sao?
> **Trả lời:**  
> Viện NIST khi xây dựng tập chuẩn EMNIST Balanced (47 lớp) đã chủ đích gộp chung 15 cặp chữ hoa - chữ thường có hình dạng tương đồng (như c/C, i/I, j/J, o/O, s/S, v/V...) thành một nhãn in hoa duy nhất là 'I'. Do đó mạng ResNet thuần túy không có nhãn 'i' thường. Nhóm đã bổ sung module phân tích hình thái học Connected Components: Phân tích các thành phần đồ họa tách rời. Khi phát hiện cấu trúc 2 phần liên thông gồm thân chữ và dấu chấm nhỏ nằm thẳng hàng phía trên, hệ thống sẽ tự động khôi phục nhãn 'i' thường chuẩn xác.

#### Câu 4: Làm sao nhóm kiểm soát và ngăn chặn hiện tượng quá khớp (Overfitting)?
> **Trả lời:**  
> Nhóm áp dụng đồng thời 4 kỹ thuật chính quy hóa: 1) Tầng Dropout (25% - 50%) ngắt kết nối nơ-ron ngẫu nhiên trong quá trình huấn luyện; 2) Phạt trọng số Weight Decay (L2 Regularization = 1e-4); 3) Tăng cường dữ liệu phong phú (Data Augmentation) với biến dạng đàn hồi (Elastic Distortion), xoay nghiêng ngẫu nhiên và biến thiên nét bút; 4) Cơ chế Early Stopping theo dõi hàm mất mát Validation Loss.

#### Câu 5: Hệ thống đạt độ trễ suy luận (Latency) là bao nhiêu và có đáp ứng thời gian thực không?
> **Trả lời:**  
> Trên GPU NVIDIA RTX 2050, độ trễ suy luận của mô hình ký tự đơn ResNet chỉ mất dưới 2.0 ms/ký tự. Mô hình CRNN nhận diện cả từ viết tay liền nét (như 'account') chỉ mất khoảng 24 - 28 ms/từ. Độ trễ này hoàn toàn đáp ứng tiêu chuẩn thời gian thực (Real-time Interaction) trên trình duyệt.
