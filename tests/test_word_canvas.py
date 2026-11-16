import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from segmentation import WordSegmenter
from onnx_engine import InferenceEngine


def test_word_canvas_digits_recognition():
    """Kiểm tra phân đoạn và nhận diện chuỗi ký tự '1047' vẽ trên canvas nền tối."""
    canvas = np.zeros((180, 700, 3), dtype=np.uint8)

    # Giả lập vẽ chuỗi chữ số "1 0 4 7" bằng nét vẽ trắng trên nền đen
    digits = ["1", "0", "4", "7"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, d in enumerate(digits):
        x = 60 + i * 150
        y = 130
        cv2.putText(canvas, d, (x, y), font, 3.5, (255, 255, 255), 7, cv2.LINE_AA)

    segmenter = WordSegmenter()
    segmented_items, annotated = segmenter.segment_image(canvas)

    assert len(segmented_items) == 4, f"Cần phát hiện đúng 4 chữ số, thực tế: {len(segmented_items)}"

    engine = InferenceEngine()
    res = engine.predict_word(segmented_items, filter_mode="digits")

    assert res["num_characters"] == 4
    assert res["predicted_text"] == "1047", f"Dự đoán mong đợi '1047', thực tế: '{res['predicted_text']}'"
