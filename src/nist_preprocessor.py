"""

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
from typing import Dict, Any, Tuple


class NISTPreprocessor:
    """Bộ tiền xử lý ảnh viết tay đạt chuẩn Viện Tiêu chuẩn và Kỹ thuật Quốc gia Hoa Kỳ (NIST)."""

    def __init__(self, target_size=(28, 28), inner_box_size=20):
        self.target_size = target_size
        self.inner_box_size = inner_box_size

    def preprocess_canvas_image(self, canvas_array: np.ndarray) -> np.ndarray:
        """Xử lý nhanh ảnh vẽ tay sang ma trận 28x28 chuẩn hóa [0.0, 1.0]."""
        res = self.preprocess_with_steps(canvas_array)
        return res["normalized"]

    def preprocess_with_steps(self, canvas_array: np.ndarray) -> Dict[str, Any]:
        """
        Bóc tách chi tiết từng bước thuật toán tiền xử lý phục vụ trực quan hóa và thuyết trình.
        """
        # 1. Chuyển đổi sang ảnh xám Grayscale
        if len(canvas_array.shape) == 3:
            if canvas_array.shape[2] == 4:
                alpha = canvas_array[:, :, 3]
                rgb = canvas_array[:, :, :3]
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                if np.max(alpha) > 0 and np.mean(gray) > 200:
                    gray = 255 - gray
            else:
                gray = cv2.cvtColor(canvas_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = canvas_array.copy()

        # 2. Đảm bảo nền đen chữ trắng
        if np.mean(gray) > 127:
            gray = 255 - gray

        # 3. Lọc nhiễu nhẹ bằng Gaussian Blur
        gray_clean = cv2.GaussianBlur(gray, (3, 3), 0)

        # 4. Cắt vùng nét chữ (Bounding Box)
        coords = cv2.findNonZero(gray_clean)
        if coords is None:
            empty = np.zeros(self.target_size, dtype=np.float32)
            return {
                "step1_gray": gray_clean,
                "step2_crop": np.zeros((20, 20), dtype=np.uint8),
                "step3_resized": np.zeros((20, 20), dtype=np.uint8),
                "step4_padded": np.zeros((28, 28), dtype=np.uint8),
                "step5_centered": np.zeros((28, 28), dtype=np.uint8),
                "centroid": (14.0, 14.0),
                "shift": (0, 0),
                "bbox": (0, 0, 0, 0),
                "normalized": empty
            }

        x, y, w, h = cv2.boundingRect(coords)
        char_crop = gray_clean[y:y+h, x:x+w]

        # 5. Co giãn vào khung 20x20 giữ nguyên tỷ lệ cạnh
        if w > h:
            new_w = self.inner_box_size
            new_h = max(1, int(round((h * self.inner_box_size) / w)))
        else:
            new_h = self.inner_box_size
            new_w = max(1, int(round((w * self.inner_box_size) / h)))

        resized_char = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 6. Đệm viền đối xứng vào khung 28x28
        pad_top = (self.target_size[0] - new_h) // 2
        pad_bottom = self.target_size[0] - new_h - pad_top
        pad_left = (self.target_size[1] - new_w) // 2
        pad_right = self.target_size[1] - new_w - pad_left

        padded = cv2.copyMakeBorder(
            resized_char,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )

        # 7. Tính Trọng tâm điểm ảnh (Center of Mass) theo công thức mô-men
        cy, cx = ndimage.center_of_mass(padded)
        if not np.isnan(cx) and not np.isnan(cy):
            shift_x = int(round(14.0 - cx))
            shift_y = int(round(14.0 - cy))

            # Giới hạn dịch chuyển tránh mất nét
            shift_x = max(-4, min(4, shift_x))
            shift_y = max(-4, min(4, shift_y))

            translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            centered = cv2.warpAffine(padded, translation_matrix, self.target_size)
        else:
            shift_x, shift_y = 0, 0
            cx, cy = 14.0, 14.0
            centered = padded

        # 8. Chuẩn hóa về [0.0, 1.0] với cân bằng cường độ đỉnh (Peak Normalization)
        max_val = float(np.max(centered))
        if max_val > 10.0:
            normalized = centered.astype(np.float32) / max_val
        else:
            normalized = centered.astype(np.float32) / 255.0

        return {
            "step1_gray": gray_clean,
            "step2_crop": char_crop,
            "step3_resized": resized_char,
            "step4_padded": padded,
            "step5_centered": centered,
            "centroid": (float(cx), float(cy)),
            "shift": (shift_x, shift_y),
            "bbox": (x, y, w, h),
            "normalized": normalized
        }

    def to_model_input(self, normalized_array: np.ndarray) -> np.ndarray:
        """Định hình ma trận ảnh về dạng NCHW (1, 1, 28, 28) sẵn sàng cho PyTorch / ONNX."""
        return np.expand_dims(np.expand_dims(normalized_array, axis=0), axis=0)
