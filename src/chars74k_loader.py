import os
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor

try:
    from nist_preprocessor import NISTPreprocessor
except ImportError:
    from src.nist_preprocessor import NISTPreprocessor


class Chars74kDataset(Dataset):
    """
    Bộ nạp dữ liệu Chars74K (Handwritten Hnd và Synthesized Fonts Fnt).
    Tự động chuẩn hóa ảnh về ma trận 28x28 căn chuẩn trọng tâm NIST Moments và lưu trữ trong RAM (In-Memory).
    """

    BALANCED_MERGE_MAP = {
        'c': 'C', 'i': 'I', 'j': 'J', 'k': 'K', 'l': 'L',
        'm': 'M', 'o': 'O', 'p': 'P', 's': 'S', 'u': 'U',
        'v': 'V', 'w': 'W', 'x': 'X', 'y': 'Y', 'z': 'Z'
    }

    # Bảng 47 lớp EMNIST Balanced
    BALANCED_CLASSES = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
    ]

    def __init__(
        self,
        base_dir: str = "C:/HocC/HandwritingAI/data/chars74k",
        mode: str = "digits",       # "digits" (10 lớp), "letters" (26 lớp), "balanced" (47 lớp)
        include_hnd: bool = True,
        include_fnt: bool = True,
        max_samples_per_class: int = 400,
        transform=None
    ):
        self.mode = mode
        self.transform = transform
        self.preprocessor = NISTPreprocessor()

        base_path = Path(base_dir)
        hnd_path = base_path / "English" / "Hnd" / "Img"
        fnt_path = base_path / "English" / "Fnt"

        raw_items = []

        # 1. Quét danh sách file
        for idx in range(1, 63):
            label = self._get_label_for_index(idx)
            if label is None:
                continue

            sample_name = f"Sample{idx:03d}"
            count = 0

            # Lấy từ Handwritten
            if include_hnd and (hnd_path / sample_name).exists():
                for p in (hnd_path / sample_name).glob("*.png"):
                    raw_items.append((str(p), label))
                    count += 1

            # Lấy từ Fnt
            if include_fnt and (fnt_path / sample_name).exists():
                for p in (fnt_path / sample_name).glob("*.png"):
                    if count >= max_samples_per_class:
                        break
                    raw_items.append((str(p), label))
                    count += 1

        print(f"[Chars74k] Đang xử lý tiền xử lý NIST cho {len(raw_items)} mẫu ({mode})...")

        # 2. Xử lý song song bằng ThreadPoolExecutor và lưu vào RAM
        processed_tensors = []
        labels = []

        def process_one(item):
            path_str, lbl = item
            img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            if np.mean(img) > 127:
                img = 255 - img
            norm_28 = self.preprocessor.preprocess_canvas_image(img)
            return norm_28, lbl

        with ThreadPoolExecutor(max_workers=8) as ex:
            results = ex.map(process_one, raw_items)

        for res in results:
            if res is not None:
                norm_28, lbl = res
                processed_tensors.append(norm_28)
                labels.append(lbl)

        # Chuyển đổi thành tensor bộ nhớ RAM
        self.data = torch.tensor(np.array(processed_tensors, dtype=np.float32)).unsqueeze(1)
        self.targets = torch.tensor(labels, dtype=torch.long)

        print(f"[Chars74k] Đã nạp thành công vào RAM: {len(self.data)} mẫu tensor 28x28")

    def _get_label_for_index(self, idx: int) -> Optional[int]:
        if 1 <= idx <= 10:
            char = str(idx - 1)
        elif 11 <= idx <= 36:
            char = chr(ord('A') + idx - 11)
        else:
            char = chr(ord('a') + idx - 37)

        if self.mode == "digits":
            return (idx - 1) if 1 <= idx <= 10 else None
        elif self.mode == "letters":
            if 11 <= idx <= 36:
                return idx - 11
            elif 37 <= idx <= 62:
                return idx - 37
            return None
        elif self.mode == "balanced":
            norm_char = self.BALANCED_MERGE_MAP.get(char, char)
            if norm_char in self.BALANCED_CLASSES:
                return self.BALANCED_CLASSES.index(norm_char)
            return None
        return None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = self.data[idx]
        y = int(self.targets[idx].item())
        if self.transform:
            x = self.transform(x)
        return x, y
