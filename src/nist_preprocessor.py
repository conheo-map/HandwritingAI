"""
nist_preprocessor.py — Thuật toán Chuẩn hóa Nét vẽ Chuẩn NIST (Center of Mass Normalization)
Tác giả: Truong Duy Cuong

Quy trình chuẩn hóa toán học 5 bước chuẩn EMNIST:
  1. Nhị phân hóa & Cắt vùng biên (Bounding Box Extraction)
  2. Co giãn giữ nguyên tỷ lệ cạnh (Aspect Ratio Preserved Rescaling to 20x20)
  3. Đệm viền đối xứng vào khung 28x28 (Padding)
  4. Tính toán trọng tâm khối điểm ảnh (Center of Mass via Image Moments)
  5. Dịch chuyển tịnh tiến trọng tâm về tâm hình học (14, 14)
"""

import cv2
import numpy as np
from scipy import ndimage
from PIL import Image


class NISTPreprocessor:
    """Bộ tiền xử lý ảnh viết tay đạt chuẩn Viện Tiêu chuẩn và Kỹ thuật Quốc gia Hoa Kỳ (NIST)."""

    def __init__(self, target_size=(28, 28), inner_box_size=20):
        self.target_size = target_size
        self.inner_box_size = inner_box_size

    def preprocess_canvas_image(self, canvas_array: np.ndarray) -> np.ndarray:
        """
        Xử lý ảnh vẽ tay từ Canvas (RGBA hoặc RGB) sang ma trận xám chuẩn EMNIST 28x28.
        Nền đen (0), chữ trắng (255), giá trị chuẩn hóa [0.0, 1.0].
        """
        # 1. Chuyển đổi sang ảnh xám Grayscale
        if len(canvas_array.shape) == 3:
            if canvas_array.shape[2] == 4:
                # Nếu là RGBA: Kiểm tra kênh Alpha
                alpha = canvas_array[:, :, 3]
                rgb = canvas_array[:, :, :3]
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                # Nếu nền trong suốt có nét vẽ
                if np.max(alpha) > 0 and np.mean(gray) > 200:
                    gray = 255 - gray
            else:
                gray = cv2.cvtColor(canvas_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = canvas_array.copy()

        # 2. Đảm bảo nền đen chữ trắng (Invert nếu nền sáng chữ tối)
        if np.mean(gray) > 127:
            gray = 255 - gray

        # 3. Lọc nhiễu nhẹ bằng Gaussian Blur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # 4. Cắt vùng nét chữ (Bounding Box)
        coords = cv2.findNonZero(gray)
        if coords is None:
            # Khung hình rỗng (không có nét vẽ)
            return np.zeros(self.target_size, dtype=np.float32)

        x, y, w, h = cv2.boundingRect(coords)
        char_crop = gray[y:y+h, x:x+w]

        # 5. Co giãn vào khung 20x20 giữ nguyên tỷ lệ cạnh
        if w > h:
            new_w = self.inner_box_size
            new_h = max(1, int(round((h * self.inner_box_size) / w)))
        else:
            new_h = self.inner_box_size
            new_w = max(1, int(round((w * self.inner_box_size) / h)))

        resized_char = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 6. Đặt vào giữa khung 28x28
        pad_top = (self.target_size[0] - new_h) // 2
        pad_bottom = self.target_size[0] - new_h - pad_top
        pad_left = (self.target_size[1] - new_w) // 2
        pad_right = self.target_size[1] - new_w - pad_left

        padded = cv2.copyMakeBorder(
            resized_char,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )

        # 7. Tính Trọng tâm (Center of Mass) và dịch chuyển về tâm (14, 14)
        cy, cx = ndimage.center_of_mass(padded)
        if not np.isnan(cx) and not np.isnan(cy):
            shift_x = int(round(14.0 - cx))
            shift_y = int(round(14.0 - cy))

            # Giới hạn khoảng dịch chuyển tránh trôi nét ra ngoài khung
            shift_x = max(-4, min(4, shift_x))
            shift_y = max(-4, min(4, shift_y))

            translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            centered = cv2.warpAffine(padded, translation_matrix, self.target_size)
        else:
            centered = padded

        # 8. Chuẩn hóa về dải [0.0, 1.0]
        normalized = centered.astype(np.float32) / 255.0
        return normalized

    def to_model_input(self, normalized_array: np.ndarray) -> np.ndarray:
        """Định hình ma trận ảnh về dạng NCHW (1, 1, 28, 28) sẵn sàng cho PyTorch / ONNX."""
        return np.expand_dims(np.expand_dims(normalized_array, axis=0), axis=0)
