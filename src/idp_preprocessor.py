"""
idp_preprocessor.py — Module Tiền Xử Lý Hình Học & Khử Lóa Thẻ Doanh Nghiệp (Enterprise Vision Preprocessor)
Tác giả: Truong Duy Cuong
Dự án: Enterprise IDP & Document AI (HandwritingAI Studio)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 tọa độ đỉnh của đa giác theo thứ tự chuẩn hình học:
      [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    
    Thuật toán:
      - Top-Left: Có tổng (x + y) nhỏ nhất.
      - Bottom-Right: Có tổng (x + y) lớn nhất.
      - Top-Right: Có hiệu (y - x) nhỏ nhất (hoặc x - y lớn nhất).
      - Bottom-Left: Có hiệu (y - x) lớn nhất.
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.asarray(pts, dtype="float32").reshape((4, 2))

    # 1. Tổng (x + y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left
    rect[2] = pts[np.argmax(s)]  # Bottom-Right

    # 2. Hiệu (y - x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Nắn phối cảnh (Perspective Warp) duỗi thẳng tài liệu/thẻ CCCD từ ảnh chụp góc xiên.
    
    Tính toán kích thước mới dựa trên khoảng cách Euclidean lớn nhất giữa các cặp cạnh,
    sau đó áp dụng ma trận biến đổi phối cảnh tương thích 100% với OpenCV.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Tính chiều rộng tối đa (Width)
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Tính chiều cao tối đa (Height)
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Đảm bảo kích thước tối thiểu
    max_width = max(max_width, 100)
    max_height = max(max_height, 100)

    # Tọa độ đích phẳng
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Tính ma trận phối cảnh M và warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def detect_document_corners(image: np.ndarray) -> np.ndarray:
    """
    Tự động phát hiện 4 góc cạnh lớn nhất của tài liệu / thẻ CCCD trên ảnh gốc.
    
    Quy trình:
      1. Thu nhỏ ảnh tỉ lệ để tăng tốc xử lý contour.
      2. Chuyển ảnh xám + Làm mờ Gaussian Blur + Dò biên Canny.
      3. Giãn nở hình thái học (Morphological Dilate) nối liền nét đứt.
      4. Tìm các đường bao lớn nhất và xấp xỉ đa giác (approxPolyDP).
      5. Nếu tìm thấy đa giác 4 đỉnh có diện tích đủ lớn (>10% ảnh), trả về 4 góc.
      6. Fallback: Nếu không tìm thấy, trả về 4 góc của khung hình hoặc bounding box lớn nhất.
    """
    orig_h, orig_w = image.shape[:2]
    
    # Scale tỉ lệ về chiều cao 600px để tính toán nhanh
    ratio = orig_h / 600.0
    scaled_h, scaled_w = 600, int(orig_w / ratio)
    resized = cv2.resize(image, (scaled_w, scaled_h))

    # Chuyển xám và lọc nhiễu
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 180)

    # Giãn nở nét biên
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=2)

    # Tìm contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    target_contour = None
    min_area = (scaled_w * scaled_h) * 0.08  # Tối thiểu 8% diện tích

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            target_contour = approx
            break

    # Nếu không tìm thấy tứ giác hoàn hảo, thử lấy Convex Hull của contour lớn nhất
    if target_contour is None and len(contours) > 0 and cv2.contourArea(contours[0]) >= min_area:
        hull = cv2.convexHull(contours[0])
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.025 * peri, True)
        if len(approx) == 4:
            target_contour = approx
        else:
            # Dùng bounding box xoay tối thiểu
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            target_contour = box.reshape((4, 1, 2))

    # Nếu vẫn không thấy, fallback về toàn bộ biên ảnh
    if target_contour is None:
        target_corners = np.array([
            [0, 0],
            [orig_w - 1, 0],
            [orig_w - 1, orig_h - 1],
            [0, orig_h - 1]
        ], dtype="float32")
    else:
        # Nhân lại tỉ lệ ban đầu
        target_corners = (target_contour.reshape(4, 2) * ratio).astype("float32")

    return order_points(target_corners)


def remove_glare_and_enhance(
    image: np.ndarray, 
    clip_limit: float = 2.0, 
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Khử bóng lóa đèn flash (Flash Glare Removal) và tăng cường độ tương phản văn bản.
    
    Kỹ thuật:
      - Chuyển đổi sang không gian màu LAB.
      - Tách kênh L (Luminance - Độ sáng).
      - Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) trên kênh L
        để cân bằng độ sáng cục bộ mà không làm biến dạng màu sắc thực của thẻ/văn bản.
      - Gộp lại các kênh và chuyển về BGR.
    """
    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)

    merged = cv2.merge((cl, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced


def deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ước lượng góc nghiêng và xoay thẳng dòng văn bản (Deskew Algorithm).
    
    Quy trình:
      1. Binarize ảnh (nền đen chữ trắng).
      2. Tìm tọa độ các pixel tiền cảnh (Foreground pixels).
      3. Tính góc quay tối thiểu (cv2.minAreaRect).
      4. Xoay ảnh quanh tâm bằng ma trận affine để đưa về góc 0 độ.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Nhị phân hóa thích ứng theo độ sáng nền
    if np.mean(gray) > 127:
        # Nền sáng, chữ tối -> đảo ngược để chữ thành màu trắng
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    else:
        # Nền tối, chữ sáng -> giữ nguyên chữ trắng
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Kiểm tra tỷ lệ pixel trắng
    white_pixels = np.column_stack(np.where(thresh > 0))
    if len(white_pixels) < 20:
        return image, 0.0

    # Ước lượng góc xoay
    angle = cv2.minAreaRect(white_pixels)[-1]
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle

    # Giới hạn góc xoay tối đa +-30 độ để tránh xoay ngược lộn đầu
    angle = max(-30.0, min(float(angle), 30.0))

    if abs(angle) < 0.3:
        return image, 0.0

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, 
        M, 
        (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


class DocumentPreprocessor:
    """
    Lớp điều phối toàn trình (Pipeline Orchestrator) tiền xử lý tài liệu doanh nghiệp.
    """

    def __init__(self, clahe_clip: float = 2.0):
        self.clahe_clip = clahe_clip

    def preprocess(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Thực hiện toàn trình tiền xử lý ảnh:
          Ảnh gốc -> Phát hiện 4 góc -> Nắn phối cảnh (Warp) -> Khử lóa (CLAHE) -> Xoay thẳng (Deskew)
        """
        if image is None or image.size == 0:
            return {
                "success": False,
                "error": "Ảnh đầu vào rỗng (Empty Image)"
            }

        # 1. Phát hiện 4 góc
        corners = detect_document_corners(image)

        # 2. Nắn phối cảnh phẳng
        warped = four_point_transform(image, corners)

        # 3. Khử lóa flash & cân bằng sáng
        enhanced = remove_glare_and_enhance(warped, clip_limit=self.clahe_clip)

        # 4. Tự động xoay thẳng dòng văn bản
        deskewed, angle = deskew(enhanced)

        return {
            "success": True,
            "original": image,
            "corners": corners,
            "warped": warped,
            "enhanced": enhanced,
            "deskewed": deskewed,
            "deskew_angle": angle
        }
