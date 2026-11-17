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

    def refine_dotted_character(self, raw_img: np.ndarray, predicted_char: str) -> Tuple[str, bool]:
        """
        Khôi phục chính xác chữ 'i' / 'j' thường khi mô hình dự đoán ra 'I' / 'J'.
        Trong tập EMNIST Balanced 47 lớp của Viện NIST, chữ 'I' in hoa và 'i' thường bị gộp chung vào nhãn 'I'.
        Phương thức này phân tích hình thái học (Connected Components): Nếu phát hiện dấu chấm tách rời
        ở phía trên thân chữ, nhãn sẽ được tinh chỉnh thành chữ 'i' hoặc 'j' thường chuẩn xác.
        Trả về: (refined_character, is_refined)
        """
        if predicted_char not in ['I', 'J', 'l', '1']:
            return predicted_char, False

        # 0. Chuẩn hóa kiểu dữ liệu
        if raw_img.dtype != np.uint8:
            if np.max(raw_img) <= 1.0:
                img_u8 = (raw_img * 255).astype(np.uint8)
            else:
                img_u8 = raw_img.astype(np.uint8)
        else:
            img_u8 = raw_img.copy()

        # 1. Chuyển đổi sang ảnh xám
        if len(img_u8.shape) == 3:
            if img_u8.shape[2] == 4:
                rgb = img_u8[:, :, :3]
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_u8.copy()

        # Nền đen chữ trắng
        if np.mean(gray) > 127:
            gray = 255 - gray

        # 2. Nhị phân hóa và tìm thành phần liên thông
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        valid_comps = []
        for idx in range(1, num_labels):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area > 8:  # Lọc bỏ nhiễu cực nhỏ
                valid_comps.append({
                    "area": area,
                    "y": stats[idx, cv2.CC_STAT_TOP],
                    "x": stats[idx, cv2.CC_STAT_LEFT],
                    "w": stats[idx, cv2.CC_STAT_WIDTH],
                    "h": stats[idx, cv2.CC_STAT_HEIGHT]
                })

        # Chữ 'i' hoặc 'j' thường luôn có ít nhất 2 thành phần: Dấu chấm ở trên và thân chữ ở dưới
        if len(valid_comps) >= 2:
            # Component có diện tích lớn nhất chính là thân chữ
            body_comp = max(valid_comps, key=lambda c: c["area"])

            for comp in valid_comps:
                if comp == body_comp:
                    continue

                # Điều kiện dấu chấm:
                # 1. Nằm phía trên thân chữ
                is_above = (comp["y"] + comp["h"]) <= (body_comp["y"] + 14)
                # 2. Diện tích nhỏ hơn thân chữ rõ rệt
                area_ratio = comp["area"] / max(1, body_comp["area"])
                # 3. Thẳng hàng theo phương ngang (cho phép sai số nhỏ)
                overlap_x = min(comp["x"] + comp["w"], body_comp["x"] + body_comp["w"]) - max(comp["x"], body_comp["x"])

                if area_ratio < 0.65 and is_above and overlap_x > -25:
                    if predicted_char in ['I', 'l', '1']:
                        return 'i', True
                    elif predicted_char == 'J':
                        return 'j', True

        return predicted_char, False
