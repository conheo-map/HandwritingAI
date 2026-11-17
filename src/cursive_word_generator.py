import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
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
    def __init__(self, alpha: float = 10.0, sigma: float = 3.0):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        if h < 4 or w < 4:
            return image
        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
        indices_y = np.clip(y + dy, 0, h - 1).astype(np.float32)

        distorted = cv2.remap(image, indices_x, indices_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return distorted


class RealHandwrittenCharBank:
    """Kho ký tự viết tay của người thật từ Chars74k để ghép thành từ ngữ tự nhiên."""
    def __init__(self, chars74k_dir: str = "C:/HocC/HandwritingAI/data/chars74k"):
        self.bank: Dict[str, List[Path]] = {}
        base_path = Path(chars74k_dir) / "English" / "Hnd" / "Img"

        if base_path.exists():
            for idx in range(1, 63):
                sample_name = f"Sample{idx:03d}"
                if 1 <= idx <= 10:
                    ch = str(idx - 1)
                elif 11 <= idx <= 36:
                    ch = chr(ord('A') + idx - 11)
                else:
                    ch = chr(ord('a') + idx - 37)

                dir_p = base_path / sample_name
                if dir_p.exists():
                    files = list(dir_p.glob("*.png"))
                    if files:
                        self.bank[ch] = files

        print(f"[RealCharBank] Đã nạp kho ký tự viết tay của người thật: {len(self.bank)} lớp ký tự.")

    def has_char(self, ch: str) -> bool:
        return ch in self.bank and len(self.bank[ch]) > 0

    def get_char_image(self, ch: str) -> Optional[np.ndarray]:
        if not self.has_char(ch):
            return None
        img_p = random.choice(self.bank[ch])
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        if np.mean(img) > 127:
            img = 255 - img
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = img[max(0, y-1):y+h+1, max(0, x-1):x+w+1]
        return img


class ComprehensiveWordDataset(Dataset):
    """
    Tập dữ liệu huấn luyện đa dạng toàn diện cho CRNN:
      1. Ký tự đơn lẻ ('h', 'i', 'a', 'b', '7', '5'...) để chống ảo giác với từ ngắn.
      2. Từ ngắn 2-3 ký tự ('hi', 'he', 'to', 'in', 'is', 'AI', '20'...).
      3. Hơn 10,000 từ vựng phong phú từ kho Google Vocabulary.
      4. Ghép từ ký tự viết tay thật của người thật (Chars74k).
      5. Chuẩn hóa CĂN LỀ TRÁI (Left-Aligned) triệt tiêu khoảng đen ảo giác.
    """

    SHORT_WORDS = [
        "hi", "he", "is", "it", "to", "in", "on", "at", "we", "an", "my", "go", "no", "so",
        "do", "me", "by", "up", "am", "as", "or", "if", "us", "ok", "ai", "ml", "id", "cv",
        "the", "and", "for", "cat", "dog", "car", "you", "all", "day", "get", "hot", "man",
        "new", "now", "old", "one", "out", "red", "run", "say", "see", "she", "sun", "use",
        "way", "who", "boy", "bad", "box", "key", "law", "map", "pie", "tax", "war", "yes",
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "10", "20", "26", "2026", "100"
    ]

    SINGLE_CHARS = list(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
    )

    HANDWRITING_FONTS = [
        "segoesc.ttf",   # Segoe Script (chữ thảo uốn lượn nối nét)
        "segoepr.ttf",   # Segoe Print (chữ viết tay tự nhiên)
        "inkfree.ttf",   # Ink Free (bút máy nét mực nhòe)
        "mvboli.ttf",    # MV Boli (nét chữ tròn tay)
        "comic.ttf",     # Comic Sans (chữ viết tự do)
        "arial.ttf",
        "calibri.ttf",
        "consola.ttf",
        "times.ttf"
    ]

    def __init__(
        self,
        num_samples: int = 35000,
        img_h: int = 32,
        img_w: int = 160,
        vocab_path: str = "C:/HocC/HandwritingAI/data/vocab_10k.txt"
    ):
        self.num_samples = num_samples
        self.img_h = img_h
        self.img_w = img_w
        self.converter = CTCLabelConverter()
        self.elastic = ElasticDistortion(alpha=8.0, sigma=2.5)
        self.char_bank = RealHandwrittenCharBank()

        # Nạp từ điển 10,000 từ
        p = Path(vocab_path)
        if p.exists():
            self.long_words = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if 3 <= len(line.strip()) <= 10]
        else:
            self.long_words = ["account", "balance", "payment", "machine", "learning", "system", "computer"]

        # Phông chữ hệ thống
        fonts_dir = Path("C:/Windows/Fonts")
        self.fonts = []
        for f in self.HANDWRITING_FONTS:
            fp = fonts_dir / f
            if fp.exists():
                self.fonts.append(str(fp))
        if not self.fonts:
            self.fonts = [str(f) for f in fonts_dir.glob("*.ttf")][:5]

        print(f"[Dataset] Đã khởi tạo {num_samples} mẫu. Kho từ: {len(self.long_words)} từ, {len(self.SHORT_WORDS)} từ ngắn, {len(self.SINGLE_CHARS)} ký tự đơn.")

    def __len__(self) -> int:
        return self.num_samples

    def generate_from_font(self, text: str) -> np.ndarray:
        """Sinh ảnh từ phông chữ có biến dạng ngẫu nhiên."""
        font_path = random.choice(self.fonts)
        font_size = random.randint(22, 32)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        temp_w, temp_h = 360, 80
        img = Image.new("L", (temp_w, temp_h), color=0)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), text, font=font)
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        draw_x = 10
        draw_y = (temp_h - bh) // 2 - bbox[1]
        draw.text((draw_x, draw_y), text, fill=255, font=font)

        arr = np.array(img)
        coords = cv2.findNonZero(arr)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            char_crop = arr[max(0, y-2):min(temp_h, y+h+2), max(0, x-2):min(temp_w, x+w+2)]
        else:
            char_crop = np.zeros((20, 20), dtype=np.uint8)

        # Biến dạng góc nghiêng ngẫu nhiên
        ch_h, ch_w = char_crop.shape
        shear = random.uniform(-0.25, 0.25)
        m = np.float32([[1, shear, 0], [0, 1, 0]])
        nw = int(ch_w + abs(shear * ch_h))
        if shear < 0:
            m[0, 2] = abs(shear * ch_h)
        sheared = cv2.warpAffine(char_crop, m, (nw, ch_h))

        # Biến dạng độ dày nét
        if random.random() > 0.4:
            k = random.choice([2, 3])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            if random.random() > 0.5:
                sheared = cv2.dilate(sheared, kernel, iterations=1)
            else:
                sheared = cv2.erode(sheared, kernel, iterations=1)

        # Biến dạng đàn hồi
        if random.random() > 0.3:
            sheared = self.elastic(sheared)

        return sheared

    def generate_from_real_chars(self, text: str) -> Optional[np.ndarray]:
        """Ghép các ảnh chữ cái viết tay của người thật từ Chars74k."""
        crops = []
        for ch in text:
            char_img = self.char_bank.get_char_image(ch)
            if char_img is None:
                return None
            # Chuẩn hóa chiều cao các chữ cái
            scale = 26.0 / max(1, char_img.shape[0])
            w_scaled = max(4, int(round(char_img.shape[1] * scale)))
            resized_char = cv2.resize(char_img, (w_scaled, 26))
            crops.append(resized_char)

        # Ghép liền kề có thể chồng lấn nhẹ (nối nét)
        total_w = sum(c.shape[1] for c in crops) + len(crops) * 2
        stitched = np.zeros((30, max(20, total_w)), dtype=np.uint8)
        cur_x = 2
        for c in crops:
            ch_h, ch_w = c.shape
            stitched[2:2+ch_h, cur_x:cur_x+ch_w] = np.maximum(stitched[2:2+ch_h, cur_x:cur_x+ch_w], c)
            cur_x += ch_w + random.randint(-1, 2)  # Nối nét chồng lấn nhẹ

        coords = cv2.findNonZero(stitched)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            return stitched[y:y+h, x:x+w]
        return stitched

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        # Phân bổ xác suất:
        # 25% là ký tự đơn lẻ ('h', 'i', '7'...)
        # 35% là từ ngắn ('hi', 'he', 'to', 'cat'...)
        # 40% là từ dài ('account', 'balance'...)
        dice = random.random()
        if dice < 0.25:
            word = random.choice(self.SINGLE_CHARS)
        elif dice < 0.60:
            word = random.choice(self.SHORT_WORDS)
        else:
            word = random.choice(self.long_words)

        if random.random() < 0.2:
            word = word.upper()
        elif random.random() < 0.4:
            word = word.capitalize()

        # Thử ghép từ ảnh người thật (30% xác suất nếu có đủ chữ)
        cropped_img = None
        if random.random() < 0.3:
            cropped_img = self.generate_from_real_chars(word)

        if cropped_img is None:
            cropped_img = self.generate_from_font(word)

        # ── QUY CHUẨN CĂN LỀ TRÁI (LEFT-ALIGNED) BẤT BIẾN ──
        ch_h, ch_w = cropped_img.shape
        scale = 26.0 / max(1, ch_h)
        scaled_w = min(self.img_w - 8, max(4, int(round(ch_w * scale))))
        resized = cv2.resize(cropped_img, (scaled_w, 26), interpolation=cv2.INTER_AREA)

        # Căn lề trái tại offset_x = 4 (Phần còn lại bên phải là nền đen padding)
        canvas = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        offset_y = 3
        offset_x = 4
        canvas[offset_y:offset_y+26, offset_x:offset_x+scaled_w] = resized

        # Nhiễu hạt nhẹ
        if random.random() > 0.5:
            noise = np.random.normal(0, random.uniform(2, 6), canvas.shape).astype(np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        norm_tensor = torch.tensor(canvas / 255.0, dtype=torch.float32).unsqueeze(0)
        return norm_tensor, word


def pad_collate_fn(batch):
    """Gom nhóm batch và đóng gói targets cho CTC Loss."""
    images, texts = zip(*batch)
    images = torch.stack(images, 0)
    converter = CTCLabelConverter()
    targets, target_lengths = converter.encode(list(texts))
    return images, targets, target_lengths, list(texts)
