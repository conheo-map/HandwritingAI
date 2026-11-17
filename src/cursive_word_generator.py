import os
import random
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset

try:
    from crnn_model import CTCLabelConverter
except ImportError:
    from src.crnn_model import CTCLabelConverter


class ElasticDistortion:
    """Biến dạng đàn hồi (Elastic Distortion) mô phỏng nét viết tay tự nhiên của con người."""
    def __init__(self, alpha: float = 12.0, sigma: float = 3.0):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        # Sinh trường vector dịch chuyển ngẫu nhiên
        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
        indices_y = np.clip(y + dy, 0, h - 1).astype(np.float32)

        distorted = cv2.remap(image, indices_x, indices_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return distorted


class SyntheticCursiveWordDataset(Dataset):
    """
    Tập dữ liệu sinh từ ngữ viết tay đa phong cách (Synthetic Cursive Word Dataset).
    Hỗ trợ sinh chữ thảo liền nét, chữ xấu, chữ nghiêng, nét thanh nét đậm thời gian thực.
    """

    COMMON_WORDS = [
        "account", "balance", "payment", "credit", "debit", "deposit", "transfer", "amount",
        "number", "system", "computer", "network", "machine", "learning", "model", "neural",
        "artificial", "intelligence", "python", "data", "science", "vision", "image", "text",
        "hello", "world", "street", "address", "company", "service", "customer", "invoice",
        "signature", "document", "digital", "secure", "status", "active", "total", "value",
        "money", "card", "bank", "check", "order", "product", "report", "result", "record",
        "action", "center", "point", "market", "trade", "price", "rate", "level", "stage",
        "class", "target", "sample", "train", "test", "valid", "score", "index", "layer",
        "paper", "pen", "hand", "write", "note", "letter", "word", "line", "page", "book",
        "time", "year", "month", "date", "first", "last", "name", "user", "pass", "code",
        "2026", "12345", "9876", "5500", "1000", "0123", "7890", "AI2026", "USD", "VND"
    ]

    HANDWRITING_FONTS = [
        "segoesc.ttf",   # Segoe Script (chữ thảo uốn lượn nối nét chuẩn mực)
        "segoepr.ttf",   # Segoe Print (chữ viết tay tự nhiên)
        "inkfree.ttf",   # Ink Free (bút máy nét mực nhòe)
        "mvboli.ttf",    # MV Boli (nét chữ tròn tay)
        "comic.ttf",     # Comic Sans (chữ viết phóng khoáng)
        "arial.ttf",
        "calibri.ttf",
        "consola.ttf",
        "times.ttf"
    ]

    def __init__(
        self,
        num_samples: int = 10000,
        img_h: int = 32,
        img_w: int = 160,
        word_list: Optional[List[str]] = None
    ):
        self.num_samples = num_samples
        self.img_h = img_h
        self.img_w = img_w
        self.words = word_list or self.COMMON_WORDS
        self.converter = CTCLabelConverter()
        self.elastic = ElasticDistortion(alpha=8.0, sigma=2.5)

        # Quét tìm phông chữ trong hệ thống Windows
        fonts_dir = Path("C:/Windows/Fonts")
        self.available_fonts = []
        for f in self.HANDWRITING_FONTS:
            p = fonts_dir / f
            if p.exists():
                self.available_fonts.append(str(p))

        if not self.available_fonts:
            self.available_fonts = [str(p) for p in fonts_dir.glob("*.ttf")][:5]

        print(f"[CursiveGenerator] Đã nạp {len(self.available_fonts)} phông chữ viết tay, quy mô: {num_samples} mẫu.")

    def __len__(self) -> int:
        return self.num_samples

    def generate_single_image(self, word: str) -> np.ndarray:
        """Kết xuất 1 ảnh từ viết tay với các biến dạng ngẫu nhiên."""
        font_path = random.choice(self.available_fonts)
        font_size = random.randint(22, 34)

        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        # 1. Tạo ảnh tạm trên nền đen chữ trắng
        temp_w = max(self.img_w * 2, 320)
        temp_h = max(self.img_h * 2, 80)
        img = Image.new("L", (temp_w, temp_h), color=0)
        draw = ImageDraw.Draw(img)

        # Căn đo kích thước chữ
        bbox = draw.textbbox((0, 0), word, font=font)
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]

        draw_x = 20
        draw_y = (temp_h - bh) // 2 - bbox[1]
        draw.text((draw_x, draw_y), word, fill=255, font=font)

        arr = np.array(img)

        # 2. Cắt sát vùng chữ (Bounding Box)
        coords = cv2.findNonZero(arr)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            char_crop = arr[max(0, y-2):min(temp_h, y+h+2), max(0, x-2):min(temp_w, x+w+2)]
        else:
            char_crop = np.zeros((20, 40), dtype=np.uint8)

        # 3. Biến dạng góc nghiêng (Random Slant Shear)
        ch_h, ch_w = char_crop.shape
        shear_factor = random.uniform(-0.25, 0.25)
        m_shear = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        new_w = int(ch_w + abs(shear_factor * ch_h))
        if shear_factor < 0:
            m_shear[0, 2] = abs(shear_factor * ch_h)
        sheared = cv2.warpAffine(char_crop, m_shear, (new_w, ch_h))

        # 4. Biến dạng nét thanh / nét đậm (Random Stroke Thickness)
        if random.random() > 0.4:
            k = random.choice([2, 3])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            if random.random() > 0.5:
                sheared = cv2.dilate(sheared, kernel, iterations=1)  # Nét đậm
            else:
                sheared = cv2.erode(sheared, kernel, iterations=1)   # Nét thanh

        # 5. Biến dạng đàn hồi uốn lượn (Elastic Distortion mô phỏng tay run)
        if random.random() > 0.3:
            sheared = self.elastic(sheared)

        # 6. Co giãn giữ nguyên tỷ lệ vào khung [H=32, W=160]
        cur_h, cur_w = sheared.shape
        scale_h = (self.img_h - 4) / max(1, cur_h)
        scaled_w = int(round(cur_w * scale_h))

        if scaled_w > self.img_w - 8:
            # Nếu từ quá dài, nén chiều ngang vừa đủ khung
            scaled_w = self.img_w - 8

        scaled_h = self.img_h - 4
        resized = cv2.resize(sheared, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        # Đệm vào khung chuẩn 32x160 với viền đen đối xứng
        canvas = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        offset_y = (self.img_h - scaled_h) // 2
        offset_x = (self.img_w - scaled_w) // 2 + random.randint(-2, 2)
        offset_x = max(2, min(self.img_w - scaled_w - 2, offset_x))

        canvas[offset_y:offset_y+scaled_h, offset_x:offset_x+scaled_w] = resized

        # 7. Nhiễu hạt Gaussian nhẹ (mô phỏng cảm biến/bút vẽ)
        if random.random() > 0.5:
            noise = np.random.normal(0, random.uniform(3, 8), canvas.shape).astype(np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return canvas

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        word = random.choice(self.words)
        # 30% trường hợp đổi thành viết hoa hoặc viết thường
        r = random.random()
        if r < 0.2:
            word = word.upper()
        elif r < 0.5:
            word = word.capitalize()

        img_arr = self.generate_single_image(word)
        norm_img = img_arr.astype(np.float32) / 255.0
        tensor_img = torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0)  # [1, 32, 160]

        return tensor_img, word


def pad_collate_fn(batch):
    """Gom nhóm batch và đóng gói targets cho CTC Loss."""
    images, texts = zip(*batch)
    images = torch.stack(images, 0)
    converter = CTCLabelConverter()
    targets, target_lengths = converter.encode(list(texts))
    return images, targets, target_lengths, list(texts)
