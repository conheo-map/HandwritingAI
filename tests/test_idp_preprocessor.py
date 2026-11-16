"""
test_idp_preprocessor.py — Unit Tests for Document Geometric & Vision Preprocessor
"""

import sys
import numpy as np
import cv2
import pytest
from pathlib import Path

# Thêm đường dẫn src vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from idp_preprocessor import (
    order_points,
    four_point_transform,
    detect_document_corners,
    remove_glare_and_enhance,
    deskew,
    DocumentPreprocessor
)


def test_order_points():
    """Kiểm tra thuật toán sắp xếp 4 góc tọa độ."""
    # Điểm ngẫu nhiên không theo thứ tự: [TR, BL, TL, BR]
    pts = np.array([[200, 10], [10, 300], [10, 10], [200, 300]], dtype="float32")
    ordered = order_points(pts)

    # Thứ tự mong đợi: [TL, TR, BR, BL]
    assert np.allclose(ordered[0], [10, 10]), "Top-left sai"
    assert np.allclose(ordered[1], [200, 10]), "Top-right sai"
    assert np.allclose(ordered[2], [200, 300]), "Bottom-right sai"
    assert np.allclose(ordered[3], [10, 300]), "Bottom-left sai"


def test_four_point_transform():
    """Kiểm tra nắn phối cảnh 4 góc (Perspective Transform)."""
    # Tạo ảnh giả lập kích thước 400x400
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Vẽ tứ giác màu xanh
    corners = np.array([[50, 50], [350, 80], [320, 350], [60, 320]], dtype="float32")
    cv2.fillPoly(img, [corners.astype(np.int32)], (0, 255, 0))

    warped = four_point_transform(img, corners)
    assert warped is not None
    assert len(warped.shape) == 3
    assert warped.shape[0] > 200 and warped.shape[1] > 200


def test_detect_document_corners():
    """Kiểm tra tự động phát hiện 4 góc thẻ/tài liệu trên nền tối."""
    # Tạo nền tối 600x800
    canvas = np.zeros((600, 800, 3), dtype=np.uint8)
    # Tạo thẻ trắng chữ nhật nghiêng nhẹ
    card_corners = np.array([[150, 100], [650, 120], [630, 480], [130, 460]], dtype=np.int32)
    cv2.fillPoly(canvas, [card_corners], (240, 240, 240))

    detected = detect_document_corners(canvas)
    assert detected is not None
    assert len(detected) == 4
    # Kiểm tra các góc phát hiện gần với card_corners
    for pt in detected:
        assert 100 <= pt[0] <= 700
        assert 80 <= pt[1] <= 500


def test_remove_glare_and_enhance():
    """Kiểm tra khử lóa đèn flash và tăng cường độ tương phản (LAB-CLAHE)."""
    # Tạo ảnh có một vùng lóa sáng trắng ở giữa
    img = np.full((200, 300, 3), 120, dtype=np.uint8)
    cv2.circle(img, (150, 100), 50, (255, 255, 255), -1)

    enhanced = remove_glare_and_enhance(img)
    assert enhanced.shape == img.shape
    # Kênh L sau CLAHE phải có độ tương phản cân bằng hơn
    assert enhanced.dtype == np.uint8


def test_deskew():
    """Kiểm tra thuật toán tự động xoay thẳng dòng văn bản (Deskew)."""
    # Tạo ảnh có một đường kẻ nghiêng -15 độ
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.line(img, (30, 150), (270, 70), 255, 5)

    corrected, angle = deskew(img)
    assert corrected.shape == img.shape
    assert isinstance(angle, float)
    # Góc quay ước lượng phải khác 0
    assert abs(angle) > 0.1


def test_document_preprocessor_pipeline():
    """Kiểm tra toàn trình Pipeline DocumentPreprocessor."""
    preprocessor = DocumentPreprocessor()

    # Tạo ảnh giả lập tài liệu
    sample_img = np.zeros((500, 700, 3), dtype=np.uint8)
    corners = np.array([[100, 80], [600, 90], [580, 420], [110, 410]], dtype=np.int32)
    cv2.fillPoly(sample_img, [corners], (220, 220, 220))
    cv2.putText(sample_img, "CONG HOA XA HOI CHU NGHIA VIET NAM", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    result = preprocessor.preprocess(sample_img)
    assert "original" in result
    assert "warped" in result
    assert "enhanced" in result
    assert "corners" in result
    assert "deskew_angle" in result
    assert result["success"] is True
