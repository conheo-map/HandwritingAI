"""
test_nist_preprocessor.py — Unit Tests for NIST Center-of-Mass Preprocessor
"""

import sys
import numpy as np
import cv2
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from nist_preprocessor import NISTPreprocessor


def test_nist_preprocessor_initialization():
    """Kiểm tra khởi tạo NISTPreprocessor."""
    p = NISTPreprocessor(target_size=(28, 28), inner_box_size=20)
    assert p.target_size == (28, 28)
    assert p.inner_box_size == 20


def test_nist_preprocessor_empty_canvas():
    """Kiểm tra xử lý canvas rỗng (không có nét vẽ)."""
    p = NISTPreprocessor()
    empty = np.zeros((280, 280, 4), dtype=np.uint8)
    steps = p.preprocess_with_steps(empty)

    assert steps["normalized"].shape == (28, 28)
    assert np.max(steps["normalized"]) == 0.0
    assert steps["centroid"] == (14.0, 14.0)


def test_nist_preprocessor_center_of_mass():
    """Kiểm tra khả năng dịch chuyển trọng tâm về (14, 14)."""
    p = NISTPreprocessor()
    # Vẽ chữ 'A' ở góc trên cùng bên trái của Canvas
    canvas = np.zeros((280, 280, 3), dtype=np.uint8)
    cv2.putText(canvas, "A", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)

    steps = p.preprocess_with_steps(canvas)
    norm = steps["normalized"]

    assert norm.shape == (28, 28)
    assert np.max(norm) == 1.0  # Peak normalized

    # Trọng tâm sau khi căn chỉnh phải nằm trong vùng trung tâm [11, 17]
    from scipy import ndimage
    cy, cx = ndimage.center_of_mass(norm)
    assert 11.0 <= cx <= 17.0, f"cx={cx} nằm ngoài vùng trung tâm"
    assert 11.0 <= cy <= 17.0, f"cy={cy} nằm ngoài vùng trung tâm"


def test_nist_preprocessor_scale_invariance():
    """Kiểm tra tính bất biến tỷ lệ (Scale Invariance): chữ vẽ nhỏ vs chữ vẽ to."""
    p = NISTPreprocessor()

    # Vẽ chữ 'M' nhỏ (40x40)
    small_canvas = np.zeros((280, 280, 3), dtype=np.uint8)
    cv2.putText(small_canvas, "M", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    small_norm = p.preprocess_canvas_image(small_canvas)

    # Vẽ chữ 'M' to (180x180)
    large_canvas = np.zeros((280, 280, 3), dtype=np.uint8)
    cv2.putText(large_canvas, "M", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 6.0, (255, 255, 255), 16)
    large_norm = p.preprocess_canvas_image(large_canvas)

    assert small_norm.shape == (28, 28)
    assert large_norm.shape == (28, 28)

    # Cả hai đều phải fit vào khung 20x20 bên trong 28x28
    small_non_zero = np.sum(small_norm > 0.1)
    large_non_zero = np.sum(large_norm > 0.1)
    assert small_non_zero > 20
    assert large_non_zero > 20


def test_refine_dotted_character():
    """Kiểm tra khôi phục chữ i/j thường từ dự đoán I/J khi có dấu chấm."""
    p = NISTPreprocessor()

    # Tạo chữ 'i' gồm thân (dưới) và dấu chấm (trên)
    canvas = np.zeros((280, 280, 3), dtype=np.uint8)
    # Dấu chấm: hình tròn bán kính 6 tại (140, 70)
    cv2.circle(canvas, (140, 70), 6, (255, 255, 255), -1)
    # Thân chữ: nét thẳng từ (140, 100) đến (140, 180) dày 12px
    cv2.line(canvas, (140, 100), (140, 180), (255, 255, 255), 12)

    refined, is_refined = p.refine_dotted_character(canvas, 'I')
    assert refined == 'i'
    assert is_refined is True

    # Nếu mô hình dự đoán chữ 'A', không tinh chỉnh
    refined_a, is_ref_a = p.refine_dotted_character(canvas, 'A')
    assert refined_a == 'A'
    assert is_ref_a is False

    # Nếu chỉ có 1 nét thẳng (chữ I in hoa), không tinh chỉnh
    canvas_cap = np.zeros((280, 280, 3), dtype=np.uint8)
    cv2.line(canvas_cap, (140, 60), (140, 200), (255, 255, 255), 16)
    refined_cap, is_ref_cap = p.refine_dotted_character(canvas_cap, 'I')
    assert refined_cap == 'I'
    assert is_ref_cap is False

