import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from onnx_engine import InferenceEngine


def test_crnn_pipeline_prediction():
    """Kiểm tra luồng dự đoán từ viết tay với mạng CRNN trong InferenceEngine."""
    engine = InferenceEngine()
    assert engine.crnn_model is not None, "Mô hình CRNN chưa được nạp thành công!"

    font_path = "C:/Windows/Fonts/segoesc.ttf"
    if not Path(font_path).exists():
        font_path = "C:/Windows/Fonts/comic.ttf"

    font = ImageFont.truetype(font_path, 42)

    for word in ["h", "hi", "account"]:
        canvas = np.zeros((180, 700), dtype=np.uint8)
        img_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img_pil)
        draw.text((60, 40), word, fill=255, font=font)
        drawn_arr = np.array(img_pil)

        res = engine.predict_cursive_word(drawn_arr)

        assert "predicted_text" in res
        assert "latency_ms" in res
        assert res["latency_ms"] > 0
        assert res["predicted_text"].lower() == word.lower(), f"CRNN dự đoán '{res['predicted_text']}' cho từ '{word}'"
