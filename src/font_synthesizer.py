import os
from pathlib import Path
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple

try:
    from nist_preprocessor import NISTPreprocessor
    from chars74k_loader import Chars74kDataset
except ImportError:
    from src.nist_preprocessor import NISTPreprocessor
    from src.chars74k_loader import Chars74kDataset


class SyntheticFontDataset(Dataset):
    """
    Sinh mẫu ký tự viết tay và phông chữ tự động từ các TTF fonts hệ thống.
    Đặc biệt khai thác các font mô phỏng chữ viết tay (Segoe Script, Segoe Print, Ink Free, MV Boli, Comic Sans).
    """

    HANDWRITING_FONTS = [
        "segoesc.ttf",   # Segoe Script
        "segoepr.ttf",   # Segoe Print
        "inkfree.ttf",   # Ink Free
        "mvboli.ttf",    # MV Boli
        "comic.ttf",     # Comic Sans MS
        "arial.ttf",
        "calibri.ttf",
        "consola.ttf",
        "times.ttf"
    ]

    def __init__(
        self,
        mode: str = "digits",      # "digits", "letters", "balanced"
        samples_per_class: int = 150
    ):
        self.mode = mode
        self.preprocessor = NISTPreprocessor()
        self.font_paths = []
        fonts_dir = Path("C:/Windows/Fonts")

        for f in self.HANDWRITING_FONTS:
            p = fonts_dir / f
            if p.exists():
                self.font_paths.append(str(p))

        if not self.font_paths:
            self.font_paths = [str(p) for p in fonts_dir.glob("*.ttf")][:5]

        # Xây dựng danh sách lớp
        if mode == "digits":
            self.chars = [str(i) for i in range(10)]
            self.labels = list(range(10))
        elif mode == "letters":
            self.chars = [chr(ord('A') + i) for i in range(26)]
            self.labels = list(range(26))
        elif mode == "balanced":
            self.chars = Chars74kDataset.BALANCED_CLASSES
            self.labels = list(range(len(self.chars)))

        self.samples = []
        for label, ch in zip(self.labels, self.chars):
            for _ in range(samples_per_class):
                font_path = random.choice(self.font_paths)
                self.samples.append((ch, label, font_path))

        print(f"[SyntheticFont] Đã khởi tạo {len(self.samples)} mẫu tổng hợp đa font cho '{mode}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ch, label, font_path = self.samples[idx]

        # Sinh ảnh với kích thước 120x120
        img_size = 120
        font_size = random.randint(65, 95)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        img = Image.new("L", (img_size, img_size), color=0)
        draw = ImageDraw.Draw(img)

        # Tính bounding box để căn giữa
        bbox = draw.textbbox((0, 0), ch, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (img_size - w) // 2 - bbox[0] + random.randint(-4, 4)
        y = (img_size - h) // 2 - bbox[1] + random.randint(-4, 4)

        draw.text((x, y), ch, fill=255, font=font)

        # Chuyển sang numpy và xoay nhẹ ngẫu nhiên
        arr = np.array(img)
        angle = random.uniform(-12, 12)
        m = cv2.getRotationMatrix2D((img_size // 2, img_size // 2), angle, 1.0)
        arr = cv2.warpAffine(arr, m, (img_size, img_size))

        # Đổi độ dày nét (Dilation hoặc Erosion ngẫu nhiên)
        if random.random() > 0.5:
            k_size = random.choice([2, 3])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            arr = cv2.dilate(arr, kernel, iterations=1)

        norm_28 = self.preprocessor.preprocess_canvas_image(arr)
        tensor = torch.tensor(norm_28, dtype=torch.float32).unsqueeze(0)

        return tensor, int(label)
