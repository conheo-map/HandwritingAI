"""
app.py — Hệ thống nhận diện ký tự và văn bản viết tay (Handwriting Recognition Studio)
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

# Cấu hình trang
st.set_page_config(
    page_title="Handwriting Recognition",
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
st.sidebar.title("Cấu hình")

st.sidebar.subheader("Không gian ký tự")
context_choice = st.sidebar.selectbox(
    "Bộ lọc ký tự (Context Filter):",
    options=[
        "Toàn bộ ký tự (0-9, A-Z, a-z)",
        "Chỉ chữ số (Digits 0-9)",
        "Chỉ chữ cái (Letters A-Z, a-z)"
    ],
    index=0
)
filter_mode_map = {
    "Toàn bộ ký tự (0-9, A-Z, a-z)": "all",
    "Chỉ chữ số (Digits 0-9)": "digits",
    "Chỉ chữ cái (Letters A-Z, a-z)": "letters"
}
selected_filter = filter_mode_map[context_choice]

temperature_val = st.sidebar.slider(
    "Hệ số nhiệt độ Softmax (Temperature T):",
    min_value=0.1,
    max_value=1.0,
    value=0.35,
    step=0.05,
    help="T càng nhỏ thì độ tập trung xác suất của nhãn cao nhất càng lớn."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Nét bút vẽ")
stroke_width = st.sidebar.slider("Độ dày nét bút (pixels):", min_value=12, max_value=36, value=22, step=2)
stroke_color = st.sidebar.color_picker("Màu nét bút:", "#FFFFFF")
bg_color = st.sidebar.color_picker("Màu nền canvas:", "#000000")

# ── TIÊU ĐỀ CHÍNH ──
st.title("Handwriting Recognition")
st.caption("Nhận diện ký tự và từ viết tay thời gian thực bằng mô hình Deep ResNet-18 và chuẩn hóa NIST Center-of-Mass.")

# CSS định hình đường viền khung vẽ rõ ràng
st.markdown(
    """
    <style>
    iframe[title="streamlit_drawable_canvas.st_canvas"] {
        border: 1px solid #4B5563 !important;
        border-radius: 6px !important;
        background-color: #0E1117 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

active_mode = st.segmented_control(
    "Điều hướng chế độ:",
    options=["Bảng vẽ tương tác", "Nhận diện từ ngữ", "Phân tích thuật toán"],
    default="Bảng vẽ tương tác",
    label_visibility="collapsed"
)
if active_mode is None:
    active_mode = "Bảng vẽ tương tác"

# ══════════════════════════════════════════════════════════════════════════
# CHẾ ĐỘ 1: BẢNG VẼ TƯƠNG TÁC TRỰC TIẾP
# ══════════════════════════════════════════════════════════════════════════
if active_mode == "Bảng vẽ tương tác":
    st.subheader("Nhận diện ký tự trực tiếp")
    st.caption("Vẽ ký tự vào khung bên trái. Mô hình sẽ trích xuất ma trận 28x28 và dự đoán phân bố xác suất Softmax.")

    col_canvas, col_preview, col_pred = st.columns([1.2, 0.8, 1.2], gap="medium")

    debug_steps = None
    with col_canvas:
        st.markdown("##### Khung vẽ:")
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
        st.caption("Mẹo: Vẽ nét dứt khoát. Sử dụng biểu tượng thùng rác ở góc dưới để xóa.")

    with col_preview:
        st.markdown("##### Chuẩn hóa NIST (28x28):")
        if canvas_result.image_data is not None:
            raw_img = canvas_result.image_data
            if np.any(raw_img[:, :, :3] > 0):
                debug_steps = preprocessor.preprocess_with_steps(raw_img)
                st.session_state["debug_steps"] = debug_steps
                norm_28 = debug_steps["normalized"]
                display_28 = (norm_28 * 255).astype(np.uint8)
                display_large = cv2.resize(display_28, (200, 200), interpolation=cv2.INTER_NEAREST)
                st.image(display_large, caption=f"Trọng tâm: ({debug_steps['centroid'][0]:.1f}, {debug_steps['centroid'][1]:.1f})", width=200)
            else:
                st.info("Vẽ một ký tự vào khung bên trái.")
                norm_28 = None
        else:
            norm_28 = None

    with col_pred:
        st.markdown("##### Kết quả dự đoán:")
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
            elif best_char.isupper():
                char_type = "Chữ cái in hoa (Uppercase)"
            else:
                char_type = "Chữ cái thường (Lowercase)"

            st.markdown(
                f"""
                <div style="background-color: #1E1E2E; padding: 20px; border-radius: 8px; border: 1px solid #4B5563; text-align: center;">
                    <span style="font-size: 72px; font-weight: bold; color: #FFFFFF; font-family: monospace;">{best_char}</span>
                    <br>
                    <span style="background-color: #374151; color: #E5E7EB; padding: 4px 12px; border-radius: 12px; font-size: 13px;">{char_type}</span>
                    <p style="margin-top: 10px; color: #9CA3AF; font-size: 14px;">Độ tin cậy: <b>{conf * 100:.2f}%</b> | Độ trễ: <b>{latency:.2f} ms</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("##### Top 5 xác suất Softmax:")
            top_df = pd.DataFrame([
                {
                    "Ký tự": f"'{item['character']}'",
                    "Xác suất Softmax (%)": round(item["confidence"] * 100, 2),
                    "Phân loại": "Chữ số" if item["is_digit"] else ("In hoa" if item["is_upper"] else "Thường")
                }
                for item in res["top_k"]
            ])
            st.dataframe(top_df, hide_index=True)
        else:
            st.info("Chờ nét vẽ để thực hiện suy luận...")

# ══════════════════════════════════════════════════════════════════════════
# CHẾ ĐỘ 2: NHẬN DIỆN CHUỖI KÝ TỰ & TỪ VIẾT TAY (WORD OCR)
# ══════════════════════════════════════════════════════════════════════════
elif active_mode == "Nhận diện từ ngữ":
    st.subheader("Nhận diện từ ngữ & chuỗi viết tay")
    st.caption("Hỗ trợ cả chữ viết tay liền nét (Cursive Words) bằng mạng CRNN + CTC và chữ số / in hoa rời bằng ResNet-18.")

    col_ctrl1, col_ctrl2 = st.columns([1, 1], gap="medium")
    with col_ctrl1:
        input_mode = st.segmented_control(
            "Phương thức nhập liệu:",
            options=["Vẽ chuỗi ký tự trực tiếp", "Tải ảnh từ tệp"],
            default="Vẽ chuỗi ký tự trực tiếp"
        )
        if input_mode is None:
            input_mode = "Vẽ chuỗi ký tự trực tiếp"

    with col_ctrl2:
        algo_mode = st.segmented_control(
            "Thuật toán nhận diện:",
            options=["CRNN Liền Nét (BiLSTM + CTC)", "Phân Đoạn Rời (MoE ResNet)"],
            default="CRNN Liền Nét (BiLSTM + CTC)"
        )
        if algo_mode is None:
            algo_mode = "CRNN Liền Nét (BiLSTM + CTC)"

    if input_mode == "Vẽ chuỗi ký tự trực tiếp":
        if algo_mode == "CRNN Liền Nét (BiLSTM + CTC)":
            st.caption("Khuyên dùng: Viết tự do từ ngữ uốn lượn hoặc liền nét (ví dụ: 'account', 'hello', 'balance', '2026'). Không cần nhấc bút tạo khoảng cách.")
        else:
            st.caption("Phương pháp phân đoạn: Viết từng ký tự rời nhau một khoảng nhỏ từ trái sang phải.")

        canvas_word = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=180,
            width=700,
            drawing_mode="freedraw",
            key="word_canvas",
        )

        has_strokes = (
            canvas_word.image_data is not None
            and np.any(canvas_word.image_data[:, :, :3] > 30)
        )

        if has_strokes:
            if algo_mode == "CRNN Liền Nét (BiLSTM + CTC)":
                with st.spinner("Đang suy luận chuỗi thời gian bằng CRNN + CTC..."):
                    crnn_res = engine.predict_cursive_word(canvas_word.image_data)

                st.markdown(
                    f"""
                    <div style="background-color: #1E1E2E; padding: 18px; border-radius: 8px; border: 1px solid #4B5563; margin-top: 15px; margin-bottom: 20px;">
                        <span style="font-size: 13px; color: #10B981; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">CRNN Sequence-to-Sequence Result:</span>
                        <div style="font-size: 56px; font-weight: bold; color: #FFFFFF; font-family: monospace; letter-spacing: 4px;">
                            {crnn_res['predicted_text']}
                        </div>
                        <span style="color: #9CA3AF; font-size: 14px;">
                            Kiến trúc: <b>ResNet Backbone + 2 tầng BiLSTM (512-dim) + CTC Loss</b> | 
                            Độ trễ suy luận: <b>{crnn_res['latency_ms']:.2f} ms</b>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c_vis, c_desc = st.columns([1, 1], gap="medium")
                with c_vis:
                    st.markdown("##### Dữ liệu chuẩn hóa đưa vào mạng CRNN (32x160):")
                    disp_crnn = cv2.resize(crnn_res["preprocessed_img"], (320, 64), interpolation=cv2.INTER_NEAREST)
                    st.image(disp_crnn, caption="Đặc trưng ảnh nguyên từ được quét trượt theo trục thời gian W (Segmentation-Free)")

                with c_desc:
                    st.markdown("##### Cơ sở lý thuyết giải quyết Nghịch lý Sayre:")
                    st.markdown(
                        """
                        - **Không cắt ký tự (Segmentation-Free):** Giữ nguyên vẹn các nét nối (*ligatures*) giữa các chữ cái.
                        - **BiLSTM 2 chiều:** Nắm bắt mạch nét bút theo cả 2 chiều từ trái sang phải và ngược lại.
                        - **Hàm mất mát CTC:** Tự động gộp các bước thời gian lặp và loại bỏ token rỗng (*blank token*).
                        """
                    )
            else:
                with st.spinner("Đang phân đoạn và nhận diện chuỗi ký tự..."):
                    segmented_items, annotated = segmenter.segment_image(canvas_word.image_data)

                if segmented_items:
                    word_res = engine.predict_word(
                        segmented_items, 
                        temperature=temperature_val, 
                        filter_mode=selected_filter
                    )

                    avg_conf = (
                        sum(item["confidence"] for item in word_res["characters"]) / len(word_res["characters"])
                        if word_res["characters"] else 0.0
                    )

                    st.markdown(
                        f"""
                        <div style="background-color: #1E1E2E; padding: 18px; border-radius: 8px; border: 1px solid #4B5563; margin-top: 15px; margin-bottom: 20px;">
                            <span style="font-size: 14px; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px;">Kết Quả Nhận Diện:</span>
                            <div style="font-size: 54px; font-weight: bold; color: #FFFFFF; font-family: monospace; letter-spacing: 6px;">
                                {word_res['predicted_text']}
                            </div>
                            <span style="color: #9CA3AF; font-size: 14px;">
                                Số ký tự: <b>{word_res['num_characters']}</b> | 
                                Độ tin cậy TB: <b>{avg_conf * 100:.2f}%</b> | 
                                Tổng thời gian: <b>{word_res['total_latency_ms']:.2f} ms</b>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    c_vis, c_details = st.columns([1, 1], gap="medium")
                    with c_vis:
                        st.markdown("##### Kết quả phân đoạn không gian (Bounding Boxes):")
                        st.image(annotated, channels="BGR", caption="Các ký tự được tách độc lập theo thứ tự từ trái sang phải")

                    with c_details:
                        st.markdown("##### Chi tiết từng ký tự sau chuẩn hóa:")
                        cols_char = st.columns(min(len(segmented_items), 8))
                        for i, item in enumerate(segmented_items[:8]):
                            with cols_char[i]:
                                char_info = word_res["characters"][i]
                                st.image(item["normalized_28x28"], caption=f"#{i+1}: '{char_info['char']}'\n({char_info['confidence']*100:.1f}%)", width=55)
                else:
                    st.warning("Chưa tách được ký tự. Vui lòng viết nét liền và để khoảng cách nhỏ giữa các ký tự.")
        else:
            st.info("Hãy viết một từ ngữ hoặc chuỗi ký tự lên bảng vẽ ở trên để nhận diện.")

    else:
        col_up, col_ocr_res = st.columns([1, 1], gap="large")

        with col_up:
            up_doc = st.file_uploader("Tải ảnh chứa chữ viết tay (JPG, PNG):", type=["jpg", "jpeg", "png"], key="word_uploader")
            if up_doc:
                doc_img = Image.open(up_doc)
                st.image(doc_img, caption="Ảnh gốc tải lên")

        with col_ocr_res:
            if up_doc:
                if algo_mode == "CRNN Liền Nét (BiLSTM + CTC)":
                    with st.spinner("Đang nhận diện bằng mạng CRNN..."):
                        crnn_res = engine.predict_cursive_word(doc_img)

                    st.markdown(
                        f"""
                        <div style="background-color: #1E1E2E; padding: 18px; border-radius: 8px; border: 1px solid #4B5563; margin-bottom: 20px;">
                            <span style="font-size: 13px; color: #10B981; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">CRNN Sequence Result:</span>
                            <div style="font-size: 52px; font-weight: bold; color: #FFFFFF; font-family: monospace; letter-spacing: 4px;">
                                {crnn_res['predicted_text']}
                            </div>
                            <span style="color: #9CA3AF; font-size: 14px;">
                                Độ trễ: <b>{crnn_res['latency_ms']:.2f} ms</b> | Phương pháp: <b>Segmentation-Free CTC</b>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("##### Ảnh tiền xử lý đưa vào mạng CRNN:")
                    disp_crnn = cv2.resize(crnn_res["preprocessed_img"], (320, 64), interpolation=cv2.INTER_NEAREST)
                    st.image(disp_crnn, caption="Đặc trưng ảnh nguyên từ 32x160")
                else:
                    with st.spinner("Đang xử lý phân đoạn và nhận diện..."):
                        segmented_items, annotated = segmenter.segment_image(doc_img)

                        if segmented_items:
                            word_res = engine.predict_word(
                                segmented_items, 
                                temperature=temperature_val, 
                                filter_mode=selected_filter
                            )

                            avg_conf = (
                                sum(item["confidence"] for item in word_res["characters"]) / len(word_res["characters"])
                                if word_res["characters"] else 0.0
                            )

                            st.markdown(
                                f"""
                                <div style="background-color: #1E1E2E; padding: 18px; border-radius: 8px; border: 1px solid #4B5563; margin-bottom: 20px;">
                                    <span style="font-size: 14px; color: #9CA3AF; text-transform: uppercase; letter-spacing: 1px;">Kết Quả Nhận Diện:</span>
                                    <div style="font-size: 48px; font-weight: bold; color: #FFFFFF; font-family: monospace; letter-spacing: 4px;">
                                        {word_res['predicted_text']}
                                    </div>
                                    <span style="color: #9CA3AF; font-size: 14px;">
                                        Số ký tự: <b>{word_res['num_characters']}</b> | 
                                        Độ tin cậy TB: <b>{avg_conf * 100:.2f}%</b> | 
                                        Tổng thời gian: <b>{word_res['total_latency_ms']:.2f} ms</b>
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            st.markdown("##### Kết quả phân đoạn ảnh:")
                            st.image(annotated, channels="BGR", caption="Khung nhận diện bounding box")

                            st.markdown("##### Chi tiết từng ký tự:")
                            cols_char = st.columns(min(len(segmented_items), 8))
                            for i, item in enumerate(segmented_items[:8]):
                                with cols_char[i]:
                                    char_info = word_res["characters"][i]
                                    st.image(item["normalized_28x28"], caption=f"#{i+1}: '{char_info['char']}'\n({char_info['confidence']*100:.1f}%)", width=55)
                        else:
                            st.warning("Không tìm thấy nét chữ rõ ràng trong ảnh.")
            else:
                st.info("Tải lên ảnh chứa chữ viết tay ở cột bên trái.")

# ══════════════════════════════════════════════════════════════════════════
# CHẾ ĐỘ 3: BÓC TÁCH THUẬT TOÁN TRỰC QUAN (ALGORITHM VISUALIZER)
# ══════════════════════════════════════════════════════════════════════════
else:
    st.subheader("Trực quan hóa pipeline tiền xử lý")
    st.markdown("Biểu diễn quy trình chuẩn hóa từ nét vẽ ban đầu đến ma trận 28x28 đưa vào mạng nơ-ron.")

    debug_steps = st.session_state.get("debug_steps", None)

    if debug_steps is not None and debug_steps["bbox"][2] > 0:
        st.markdown("#### 1. Quy trình chuẩn hóa NIST 5 bước")
        s1, s2, s3, s4, s5 = st.columns(5)

        with s1:
            st.image(debug_steps["step1_gray"], caption="Bước 1: Ảnh xám", width=120)
            st.caption("Khử kênh alpha, chuyển nền đen chữ trắng, Gaussian Blur.")

        with s2:
            st.image(debug_steps["step2_crop"], caption=f"Bước 2: Cắt biên ({debug_steps['bbox'][2]}x{debug_steps['bbox'][3]})", width=120)
            st.caption("Cắt sát vùng nét mực.")

        with s3:
            st.image(debug_steps["step3_resized"], caption="Bước 3: Tỷ lệ 20x20", width=120)
            st.caption("Co giãn giữ nguyên Aspect Ratio.")

        with s4:
            st.image(debug_steps["step4_padded"], caption="Bước 4: Đệm viền 28x28", width=120)
            st.caption("Thêm viền đen đối xứng.")

        with s5:
            st.image(debug_steps["step5_centered"], caption=f"Bước 5: Căn trọng tâm", width=120)
            st.caption(f"Tịnh tiến (Δx={debug_steps['shift'][0]}, Δy={debug_steps['shift'][1]}) về (14, 14).")

        st.markdown("---")
        st.markdown("#### 2. Cơ sở toán học trọng tâm điểm ảnh (Image Moments Centroid)")
        st.latex(r"\bar{x} = \frac{\sum_{x, y} x \cdot I(x, y)}{\sum_{x, y} I(x, y)}, \quad \bar{y} = \frac{\sum_{x, y} y \cdot I(x, y)}{\sum_{x, y} I(x, y)}")
        st.markdown(
            f"""
            - **Trọng tâm tính được:** $(\\bar{{x}}={debug_steps['centroid'][0]:.2f}, \\bar{{y}}={debug_steps['centroid'][1]:.2f})$
            - **Độ lệch tịnh tiến:** $(\\Delta x = {debug_steps['shift'][0]}, \\Delta y = {debug_steps['shift'][1]})$
            """
        )
    else:
        st.info("Vẽ một chữ cái ở Tab 1 để xem các bước chuyển đổi.")


if __name__ == "__main__":
    pass
