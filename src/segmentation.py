
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from PIL import Image

try:
    from nist_preprocessor import NISTPreprocessor
except ImportError:
    from src.nist_preprocessor import NISTPreprocessor


class WordSegmenter:
    """Bộ phân đoạn hình ảnh dòng chữ viết tay thành chuỗi các ký tự đơn lẻ."""

    def __init__(self):
        self.preprocessor = NISTPreprocessor()

    def segment_image(self, image_input) -> List[Dict[str, Any]]:
        """
        Bóc tách các ký tự từ ảnh chụp đầu vào.
        Input: Đường dẫn file, PIL.Image hoặc numpy.ndarray.
        Output: Danh sách các dict chứa {bbox, cropped_char_28x28, tensor_input}.
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input.convert("RGB"))
        else:
            img = image_input.copy()

        # 1. Chuyển sang ảnh xám
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        # 2. Tiền xử lý lọc nhiễu & nhị phân hóa Otsu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Nền trắng chữ đen -> Invert thành nền đen chữ trắng
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Phép toán hình thái học (Morphological Closing) để liền nét chữ
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # 4. Tìm kiếm đường biên (Contours) của từng ký tự
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        min_area = 25  # Loại bỏ chấm nhiễu li ti
        img_h, img_w = gray.shape

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            # Lọc bỏ các contour quá nhỏ (nhiễu scan) hoặc quá lớn (khung viền giấy)
            if area > min_area and w < img_w * 0.95 and h < img_h * 0.95:
                bounding_boxes.append((x, y, w, h))

        # 5. Sắp xếp các ký tự từ trái sang phải
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

        segmented_items = []
        for x, y, w, h in bounding_boxes:
            # Thêm viền đệm nhỏ xung quanh nét chữ
            pad = 2
            y1 = max(0, y - pad)
            y2 = min(img_h, y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(img_w, x + w + pad)

            char_crop = thresh[y1:y2, x1:x2]
            processed_28 = self.preprocessor.preprocess_canvas_image(char_crop)
            tensor_input = self.preprocessor.to_model_input(processed_28)

            segmented_items.append({
                "bbox": (x, y, w, h),
                "char_crop": char_crop,
                "normalized_28x28": processed_28,
                "tensor_input": tensor_input
            })

        return segmented_items
