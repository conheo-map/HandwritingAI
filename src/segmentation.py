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

    def segment_image(self, image_input) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Bóc tách các ký tự từ ảnh chụp hoặc canvas đầu vào.
        Hỗ trợ cả nền trắng chữ đen (ảnh chụp) và nền đen chữ trắng (canvas vẽ tay).
        Trả về: (segmented_items, annotated_image)
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input.convert("RGB"))
        else:
            img = image_input.copy()

        # 1. Chuyển sang ảnh xám Grayscale
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                # Xử lý RGBA từ Canvas
                alpha = img[:, :, 3]
                rgb = img[:, :, :3]
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        # 2. Tiền xử lý lọc nhiễu & nhị phân hóa thích ứng
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        if np.mean(gray) > 127:
            # Nền sáng, chữ tối (ảnh giấy/tài liệu) -> Invert thành nền đen chữ trắng
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Nền tối, chữ sáng (Canvas vẽ tay) -> Giữ nguyên chữ trắng
            _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)

        # 3. Phép toán hình thái học (Morphological Dilation) nối liền nét rời
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # 4. Tìm kiếm đường bao (Contours) của từng ký tự
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        min_area = 20
        img_h, img_w = gray.shape

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            # Lọc bỏ các contour quá nhỏ (nhiễu) hoặc quá lớn (bao phủ toàn màn hình)
            if area > min_area and w < img_w * 0.98 and h < img_h * 0.98 and w >= 4 and h >= 8:
                bounding_boxes.append((x, y, w, h))

        # 5. Sắp xếp các ký tự từ trái sang phải theo tọa độ x
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

        # Hợp nhất các nét của cùng 1 ký tự nếu bounding box lồng nhau hoặc quá gần theo chiều dọc (như dấu chấm của chữ i, j)
        merged_boxes = []
        for box in bounding_boxes:
            bx, by, bw, bh = box
            if not merged_boxes:
                merged_boxes.append(box)
                continue
            prev_x, prev_y, prev_w, prev_h = merged_boxes[-1]
            # Nếu 2 box chồng lấn đáng kể theo trục x (ví dụ dấu chấm nằm ngay trên thân chữ)
            overlap_x = min(bx + bw, prev_x + prev_w) - max(bx, prev_x)
            if overlap_x > 0 and (overlap_x / min(bw, prev_w) > 0.4):
                # Gộp 2 box thành 1 box bao phủ cả 2
                nx = min(prev_x, bx)
                ny = min(prev_y, by)
                nw = max(prev_x + prev_w, bx + bw) - nx
                nh = max(prev_y + prev_h, by + bh) - ny
                merged_boxes[-1] = (nx, ny, nw, nh)
            else:
                merged_boxes.append(box)

        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        if len(annotated.shape) == 3 and annotated.shape[2] == 4:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_RGBA2BGR)

        segmented_items = []
        for idx, (x, y, w, h) in enumerate(merged_boxes):
            pad = 2
            y1 = max(0, y - pad)
            y2 = min(img_h, y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(img_w, x + w + pad)

            char_crop = thresh[y1:y2, x1:x2]
            processed_28 = self.preprocessor.preprocess_canvas_image(char_crop)
            tensor_input = self.preprocessor.to_model_input(processed_28)

            segmented_items.append({
                "index": idx + 1,
                "bbox": (x, y, w, h),
                "char_crop": char_crop,
                "normalized_28x28": processed_28,
                "tensor_input": tensor_input
            })

            # Vẽ khung chữ nhật và số thứ tự lên ảnh trực quan
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, str(idx + 1), (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return segmented_items, annotated
