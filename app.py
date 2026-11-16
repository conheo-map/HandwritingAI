"""
app.py — Hệ Thống Nhận Diện Ký Tự & Văn Bản Viết Tay Thời Gian Thực (HandwritingAI Studio)
Tác giả: Truong Duy Cuong
"""

import os
import sys
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from nist_preprocessor import NISTPreprocessor
from onnx_engine import InferenceEngine
from segmentation import WordSegmenter

# Cấu hình trang giao diện
st.set_page_config(
    page_title="HandwritingAI Studio — Hệ Thống Nhận Diện Chữ Viết Tay SOTA",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Khởi tạo các module
@st.cache_resource
def load_system_modules():
    preprocessor = NISTPreprocessor()
    segmenter = WordSegmenter()
    engine = InferenceEngine()
    return preprocessor, segmenter, engine


preprocessor, segmenter, engine = load_system_modules()

# ── THANH ĐIỀU KHIỂN BÊN TRÁI ──
st.sidebar.title("🎛️ Trung Tâm Điều Khiển")
st.sidebar.markdown("**Tác giả:** Trương Duy Cường")
st.sidebar.markdown("**Môn học:** Trí Tuệ Nhân Tạo (Chương 6: Học Máy)")
st.sidebar.markdown("**Phiên bản:** v1.0.0 Production")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Tối Ưu Độ Chính Xác & Ngữ Cảnh")
context_choice = st.sidebar.selectbox(
    "Bộ lọc không gian ký tự (Context Filter):",
    options=[
        "Toàn bộ 62 ký tự (0-9, A-Z, a-z)",
        "Chỉ chữ số (Digits 0-9)",
        "Chỉ chữ cái (Letters A-Z, a-z)"
    ],
    index=0
)
filter_mode_map = {
    "Toàn bộ 62 ký tự (0-9, A-Z, a-z)": "all",
    "Chỉ chữ số (Digits 0-9)": "digits",
    "Chỉ chữ cái (Letters A-Z, a-z)": "letters"
}
selected_filter = filter_mode_map[context_choice]

temperature_val = st.sidebar.slider(
    "Độ sắc nét Softmax (Temperature T):",
    min_value=0.1,
    max_value=1.0,
    value=0.35,
    step=0.05,
    help="T càng nhỏ thì đỉnh xác suất của lớp cao nhất càng tập trung, triệt tiêu phân tán đều."
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Cấu Hình Bút Vẽ Canvas")
stroke_width = st.sidebar.slider("Độ dày nét bút (Stroke Width):", min_value=12, max_value=36, value=22, step=2)
stroke_color = st.sidebar.color_picker("Màu nét bút:", "#FFFFFF")
bg_color = st.sidebar.color_picker("Màu nền Canvas:", "#000000")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Khung Học Thuật Chương 6 (NEU)")
st.sidebar.info(
    """
    - **Bài toán (T, P, E):** Nhận diện 62 lớp EMNIST ByClass (0-9, A-Z, a-z)
    - **Mô hình cốt lõi:** Mạng Nơ-ron Tích Chập (CNN / Deep ResNet)
    - **Hàm kích hoạt:** ReLU, GELU, Softmax
    - **Hàm mất mát:** Categorical Cross-Entropy
    - **Độ trễ suy luận:** < 1.5 ms (CUDA / ONNX)
    """
)

# ── TIÊU ĐỀ CHÍNH ──
st.title("✍️ HandwritingAI Studio: Hệ Thống Nhận Diện Ký Tự Viết Tay")
st.markdown(
    "Giải pháp Trí tuệ Nhân tạo Thị giác Máy tính nhận diện ký tự và từ ngữ viết tay đạt chuẩn **NIST (Viện Tiêu chuẩn Hoa Kỳ)**. "
    "Ứng dụng kiến trúc **Mạng Nơ-ron Nhân tạo (Chương 6)** kết hợp thuật toán căn trọng tâm toán học triệt tiêu lỗi lệch góc nét vẽ."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "✍️ Bảng Vẽ Tương Tác Trực Tiếp (Interactive Canvas)",
    "📷 Nhận Diện Từ / Dòng Chữ Viết Tay (Word OCR)",
    "🔬 Bóc Tách Thuật Toán Trực Quan (Algorithm Visualizer)",
    "🎓 Cẩm Nang Thuyết Trình & Vấn Đáp (Presentation & Q&A)"
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: BẢNG VẼ TƯƠNG TÁC TRỰC TIẾP
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🎯 Vẽ Bất Kỳ Chữ Số Hoặc Chữ Cái Nào Vào Khung Bên Dưới:")
    st.caption("Hệ thống sẽ tự động bắt nét, căn trọng tâm chuẩn NIST và xuất phân bố xác suất Softmax theo thời gian thực.")

    col_canvas, col_preview, col_pred = st.columns([1.2, 0.8, 1.2], gap="medium")

    debug_steps = None
    with col_canvas:
        st.markdown("##### 🖌️ Khung Vẽ Canvas (Vẽ tự nhiên):")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="handwriting_canvas",
        )
        st.caption("💡 Mẹo: Hãy vẽ nét chữ dứt khoát. Bạn có thể bấm nút thùng rác nhỏ ở góc dưới bảng vẽ để xóa.")

    with col_preview:
        st.markdown("##### 🔬 Chuẩn Hóa NIST 28×28:")
        if canvas_result.image_data is not None:
            raw_img = canvas_result.image_data
            if np.any(raw_img[:, :, :3] > 0):
                debug_steps = preprocessor.preprocess_with_steps(raw_img)
                norm_28 = debug_steps["normalized"]
                display_28 = (norm_28 * 255).astype(np.uint8)
                display_large = cv2.resize(display_28, (200, 200), interpolation=cv2.INTER_NEAREST)
                st.image(display_large, caption=f"Ma trận 28×28 căn trọng tâm ({debug_steps['centroid'][0]:.1f}, {debug_steps['centroid'][1]:.1f})", width=200)
            else:
                st.info("👈 Hãy vẽ một chữ cái hoặc chữ số lên khung bên trái.")
                norm_28 = None
        else:
            norm_28 = None

    with col_pred:
        st.markdown("##### 🏆 Kết Quả Dự Đoán AI:")
        if norm_28 is not None and np.max(norm_28) > 0.05:
            tensor_in = preprocessor.to_model_input(norm_28)
            res = engine.predict_single(
                tensor_in, 
                top_k=5, 
                temperature=temperature_val, 
                filter_mode=selected_filter
            )

            best_char = res["best_character"]
            conf = res["best_confidence"]
            latency = res["latency_ms"]

            if best_char.isdigit():
                char_type = "Chữ số (Digit)"
                badge_color = "#3498DB"
            elif best_char.isupper():
                char_type = "Chữ cái IN HOA (Uppercase)"
                badge_color = "#2ECC71"
            else:
                char_type = "Chữ cái thường (Lowercase)"
                badge_color = "#E67E22"

            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #1E1E2F 0%, #2A2A40 100%); padding: 20px; border-radius: 12px; border: 2px solid {badge_color}; text-align: center;">
                    <span style="font-size: 72px; font-weight: bold; color: #FFFFFF; font-family: monospace;">{best_char}</span>
                    <br>
                    <span style="background-color: {badge_color}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 14px; font-weight: bold;">{char_type}</span>
                    <p style="margin-top: 10px; color: #BDC3C7; font-size: 15px;">Độ tin cậy: <b>{conf * 100:.2f}%</b> | Độ trễ suy luận: <b>{latency:.2f} ms</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("##### 📊 Top 5 Khả Năng Cao Nhất (Softmax Output):")
            top_df = pd.DataFrame([
                {
                    "Ký tự": f"'{item['character']}'",
                    "Xác suất Softmax (%)": round(item["confidence"] * 100, 2),
                    "Phân loại": "Chữ số" if item["is_digit"] else ("IN HOA" if item["is_upper"] else "Thường")
                }
                for item in res["top_k"]
            ])
            st.dataframe(top_df, hide_index=True)
        else:
            st.info("Chờ nét vẽ để kích hoạt mạng nơ-ron...")

# ══════════════════════════════════════════════════════════════════════════
# TAB 2: NHẬN DIỆN TỪ / DÒNG CHỮ VIẾT TAY (WORD OCR)
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📷 Nhận Diện Toàn Bộ Từ Hoặc Dòng Chữ Từ Ảnh Chụp")
    st.markdown("Tự động bóc tách (Segmentation) từng ký tự trong từ viết tay theo trục tọa độ không gian từ trái sang phải.")

    col_up, col_ocr_res = st.columns([1, 1], gap="large")

    with col_up:
        up_doc = st.file_uploader("Tải ảnh chụp từ hoặc chuỗi số viết tay (JPG, PNG):", type=["jpg", "jpeg", "png"], key="word_uploader")
        if up_doc:
            doc_img = Image.open(up_doc)
            st.image(doc_img, caption="Ảnh gốc tải lên")

    with col_ocr_res:
        if up_doc:
            with st.spinner("Đang bóc tách đường biên và nhận diện chuỗi ký tự..."):
                segmented = segmenter.segment_image(doc_img)
                if segmented:
                    word_res = engine.predict_word(segmented)

                    st.success(f"### 🔤 Kết Quả Nhận Diện: `{word_res['predicted_text']}`")
                    st.metric("Tổng Số Ký Tự Bóc Tách", word_res["num_characters"])
                    st.metric("Tổng Thời Gian Nhận Diện", f"{word_res['total_latency_ms']:.2f} ms")

                    st.markdown("##### 🔍 Chi Tiết Từng Ký Tự Được Bóc Tách:")
                    cols_char = st.columns(min(len(segmented), 6))
                    for i, item in enumerate(segmented[:6]):
                        with cols_char[i]:
                            st.image(item["char_crop"], caption=f"Ký tự #{i+1}: '{word_res['characters'][i]['char']}'", width=60)
                else:
                    st.warning("Không tìm thấy nét chữ rõ ràng trong ảnh. Hãy thử ảnh có độ tương phản cao hơn.")
        else:
            st.info("👈 Hãy tải lên ảnh chụp chữ viết tay ở cột bên trái để thử nghiệm tính năng OCR.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3: BÓC TÁCH THUẬT TOÁN TRỰC QUAN (ALGORITHM VISUALIZER)
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔬 Chiếu X-Quang Toàn Bộ Pipeline Thuật Toán (Algorithm Visualizer)")
    st.markdown(
        "Mục này giải thích **từng bước chuyển dịch toán học từ nét vẽ tay tự do sang ma trận tensor đầu vào** của Mạng Nơ-ron. "
        "Dùng tab này để trình chiếu trực tiếp khi thuyết trình trước giảng viên!"
    )

    if debug_steps is not None and debug_steps["bbox"][2] > 0:
        st.markdown("#### 1. Quy Trình 5 Bước Tiền Xử Lý Chuẩn Hóa NIST")
        s1, s2, s3, s4, s5 = st.columns(5)

        with s1:
            st.image(debug_steps["step1_gray"], caption="Bước 1: Ảnh Xám & Lọc Nhiễu", width=120)
            st.caption("Khử kênh alpha, chuyển nền đen chữ trắng, Gaussian Blur 3x3.")

        with s2:
            st.image(debug_steps["step2_crop"], caption=f"Bước 2: Cắt Biên w={debug_steps['bbox'][2]}, h={debug_steps['bbox'][3]}", width=120)
            st.caption("Loại bỏ 100% khoảng trống thừa, chỉ giữ lại vùng chứa nét mực.")

        with s3:
            st.image(debug_steps["step3_resized"], caption="Bước 3: Fit Khung 20×20", width=120)
            st.caption("Co giãn cạnh lớn nhất về 20px, giữ nguyên tỷ lệ Aspect Ratio.")

        with s4:
            st.image(debug_steps["step4_padded"], caption="Bước 4: Đệm Viền 28×28", width=120)
            st.caption("Thêm viền đen đối xứng để đưa về kích thước chuẩn 28x28.")

        with s5:
            st.image(debug_steps["step5_centered"], caption=f"Bước 5: Căn Trọng Tâm ({debug_steps['centroid'][0]:.1f}, {debug_steps['centroid'][1]:.1f})", width=120)
            st.caption(f"Tịnh tiến (Δx={debug_steps['shift'][0]}, Δy={debug_steps['shift'][1]}) về tâm chuẩn (14, 14).")

        st.markdown("---")
        st.markdown("#### 2. Cơ Sở Toán Học Trọng Tâm Điểm Ảnh (Image Moments Centroid)")
        st.latex(r"\bar{x} = \frac{\sum_{x, y} x \cdot I(x, y)}{\sum_{x, y} I(x, y)}, \quad \bar{y} = \frac{\sum_{x, y} y \cdot I(x, y)}{\sum_{x, y} I(x, y)}")
        st.markdown(
            f"""
            - **Trọng tâm thực tế tính được:** $(\\bar{{x}}={debug_steps['centroid'][0]:.2f}, \\bar{{y}}={debug_steps['centroid'][1]:.2f})$
            - **Vector dịch chuyển tịnh tiến:** $(\\Delta x = 14 - \\bar{{x}} = {debug_steps['shift'][0]}, \\Delta y = 14 - \\bar{{y}} = {debug_steps['shift'][1]})$
            - **Ý nghĩa:** Triệt tiêu hoàn toàn sự sai lệch do người dùng vẽ chữ ở góc trên, góc dưới, to hay nhỏ!
            """
        )
    else:
        st.info("👈 Hãy sang Tab 1 vẽ một chữ cái, sau đó quay lại đây để xem X-quang từng bước thuật toán biến đổi!")

# ══════════════════════════════════════════════════════════════════════════
# TAB 4: CẨM NANG THUYẾT TRÌNH & VẤN ĐÁP BẢO VỆ ĐỒ ÁN (Q&A GUIDE)
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🎓 Cẩm Nang Báo Cáo & Vấn Đáp Bảo Vệ Đồ Án (Dành Riêng Cho Nhóm)")
    st.markdown("Tài liệu chuẩn bị sẵn giúp nhóm tự tin trả lời bất kỳ câu hỏi nào của Giảng viên chấm điểm.")

    exp1 = st.expander("⏱️ Kịch Bản Thuyết Trình Mẫu 3 Phút (Ghi Trọn Điểm Ấn Tượng)", expanded=True)
    with exp1:
        st.markdown(
            """
            * **Phút 1 (Đặt vấn đề & Cơ sở lý thuyết Chương 6):**  
              *"Kính thưa Thầy/Cô, nhóm em chọn đề tài Nhận diện ký tự viết tay – một bài toán kinh điển trong Chương 6 (Học máy). Theo định nghĩa của Tom Mitchell, bài toán có bộ ba (T, P, E): Nhiệm vụ T là phân loại 62 lớp ký tự từ ảnh 28x28; Thước đo P là Accuracy và F1-Score; Kinh nghiệm E là tập dữ liệu chuẩn EMNIST. Mô hình cốt lõi nhóm sử dụng là Mạng Nơ-ron Tích Chập (CNN) kết hợp hàm kích hoạt ReLU và Softmax."*
            * **Phút 2 (Điểm đột phá kỹ thuật so với các đề tài khác):**  
              *"Thay vì chỉ resize ảnh thô sơ khiến nét vẽ lệch góc bị đoán sai, nhóm em đã áp dụng thuật toán căn trọng tâm toán học NIST Center-of-Mass Moments: tính trọng tâm nét chữ và tịnh tiến về tâm (14, 14), giúp mô hình bất biến với kích thước và vị trí vẽ. Đồng thời, nhóm xây dựng thêm module Segmentation để nhận diện cả từ viết tay chứ không chỉ 1 chữ cái đơn lẻ."*
            * **Phút 3 (Demo thực tế):**  
              *"Sau đây nhóm xin mời Thầy/Cô trực tiếp vẽ thử bất kỳ ký tự nào lên màn hình để quan sát hệ thống phản hồi trong vòng chưa đầy 1.5 mili-giây..."*
            """
        )

    exp2 = st.expander("❓ Top 6 Câu Hỏi Vấn Đáp Giảng Viên Hay Hỏi Nhất & Câu Trả Lời Mẫu", expanded=True)
    with exp2:
        st.markdown(
            r"""
            **Câu 1: Tại sao lại chọn EMNIST ByClass 62 lớp thay vì MNIST 10 số thông thường?**  
            👉 *Trả lời:* MNIST 10 số quá dễ (đạt >99% với mạng đơn giản). Nhóm muốn thử thách với bài toán thế giới thực gồm cả chữ số, chữ hoa và chữ thường (62 lớp), đối mặt với các thách thức lớn như sự tương đồng giữa `O` và `0`, `c` và `C`, `l` và `1`.

            **Câu 2: Hàm kích hoạt Softmax ở tầng cuối cùng có vai trò toán học gì?**  
            👉 *Trả lời:* Softmax chuyển đổi vector đầu ra (logits) thành phân bố xác suất có tổng bằng 1 ($P(y=c|x) = e^{z_c} / \sum e^{z_j}$). Giá trị xác suất cao nhất chính là nhãn dự đoán của AI, đồng thời các giá trị còn lại phản ánh độ không chắc chắn (Uncertainty) của mô hình.

            **Câu 3: Làm thế nào để nhóm chống hiện tượng Học vẹt (Overfitting)?**  
            👉 *Trả lời:* Nhóm kết hợp 4 kỹ thuật điều hòa (Regularization) chuẩn giáo trình Chương 6:  
            1) Dropout ngắt ngẫu nhiên nơ-ron; 2) Batch Normalization ổn định gradient; 3) Weight Decay phạt trọng số lớn; 4) Data Augmentation xoay nghiêng nét chữ sinh học.

            **Câu 4: Sự khác nhau giữa mạng MLP truyền thống và mạng CNN là gì?**  
            👉 *Trả lời:* MLP duỗi phẳng ảnh thành 784 số, làm mất hoàn toàn tương quan không gian 2D giữa các pixel lân cận và tốn rất nhiều trọng số. CNN sử dụng các bộ lọc tích chập (Convolutional Filters) quét qua ảnh, có tính chất chia sẻ trọng số (Weight Sharing) và bất biến tịnh tiến (Translation Invariance), giúp học được các nét cơ bản (cạnh, góc, đường cong) như thị giác con người.

            **Câu 5: Tại sao mô hình này suy luận nhanh dưới 1.5 ms?**  
            👉 *Trả lời:* Nhóm tối ưu hóa kích thước mạng nhỏ gọn (~1.2M tham số), kết hợp tăng tốc phần cứng bằng GPU CUDA và kiến trúc suy luận tối ưu hóa bộ nhớ, sẵn sàng nhúng vào các thiết bị biên di động.
            """
        )


if __name__ == "__main__":
    pass
