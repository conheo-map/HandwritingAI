
import string
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any

# Bản đồ 62 ký tự EMNIST ByClass chuẩn
EMNIST_BYCLASS_CHARS = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)

# Bản đồ 47 ký tự EMNIST Balanced chuẩn (NIST Benchmark)
EMNIST_BALANCED_CHARS = list(string.digits + string.ascii_uppercase) + [
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]


class SimpleEMNISTCNN(nn.Module):
    """Mạng CNN tối ưu hóa cho EMNIST ByClass 62 lớp."""
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class InferenceEngine:
    """Bộ máy suy luận tốc độ cao cho mô hình nhận diện chữ viết tay."""

    def __init__(self, model_path: str = None, use_onnx: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_model = None
        self.class_names = EMNIST_BYCLASS_CHARS

        # Tự động tìm trọng số khả dụng trong models/
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "models"
        
        target_path = None
        if model_path and Path(model_path).exists():
            target_path = Path(model_path)
        else:
            candidates = [
                models_dir / "best_emnist_resnet.pt",
                models_dir / "best_emnist_cnn.pt"
            ]
            for c in candidates:
                if c.exists():
                    target_path = c
                    break

        if target_path and target_path.exists():
            try:
                sd = torch.load(str(target_path), map_location=self.device)
                num_classes = 62
                if isinstance(sd, dict):
                    if "classes" in sd:
                        num_classes = sd["classes"]
                    if "model_state_dict" in sd:
                        sd = sd["model_state_dict"]
                    elif "state_dict" in sd:
                        sd = sd["state_dict"]

                # Cập nhật danh sách nhãn tương ứng
                if num_classes == 47:
                    self.class_names = EMNIST_BALANCED_CHARS
                else:
                    self.class_names = EMNIST_BYCLASS_CHARS

                # Kiểm tra nạp vào ResNet hay SimpleCNN
                if "resnet" in target_path.name.lower() or "stage1.conv1.weight" in sd:
                    from modern_resnet import create_model
                    model = create_model(num_classes=num_classes)
                    model.load_state_dict(sd, strict=False)
                    self.needs_transpose = False
                else:
                    model = SimpleEMNISTCNN(num_classes=num_classes)
                    model.load_state_dict(sd, strict=True)
                    self.needs_transpose = True

                model.to(self.device).eval()
                self.torch_model = model
                print(f"[✓] Đã khởi tạo thành công Model AI ({num_classes} lớp) trên {self.device} từ {target_path.name}")
            except Exception as e:
                print(f"[!] Lỗi nạp trọng số: {e}")
                self.needs_transpose = False

    def predict_single(
        self, 
        input_tensor: np.ndarray, 
        top_k: int = 5, 
        temperature: float = 0.5,
        filter_mode: str = "all"
    ) -> Dict[str, Any]:
        """
        Suy luận 1 ký tự từ ma trận đầu vào (1, 1, 28, 28).
        Hỗ trợ:
          - Temperature Scaling: Làm sắc nét phân bố xác suất Softmax.
          - Context Filter: Lọc theo ngữ cảnh ('all', 'digits', 'letters').
        """
        t0 = time.perf_counter()

        if self.torch_model is not None:
            # EMNIST chuẩn được lưu theo row-major đảo trục: hoán vị (height <-> width)
            norm_2d = input_tensor[0, 0]
            if getattr(self, "needs_transpose", False):
                norm_2d = np.transpose(norm_2d)
            model_in = np.expand_dims(np.expand_dims(norm_2d, axis=0), axis=0)

            with torch.no_grad():
                tensor = torch.from_numpy(model_in).float().to(self.device)
                logits = self.torch_model(tensor)[0]  # (62,)

                # 1. Bộ lọc ngữ cảnh (Context Masking)
                if filter_mode == "digits":
                    # Chỉ giữ lại 0-9 (index 0 đến 9), triệt tiêu chữ cái
                    logits[10:] = -1e9
                elif filter_mode == "letters":
                    # Chỉ giữ lại chữ cái (index 10 đến 61), triệt tiêu chữ số
                    logits[:10] = -1e9

                # 2. Temperature Scaling để làm sắc nét phân bố xác suất
                T = max(0.1, min(temperature, 2.0))
                scaled_logits = logits / T
                probs = torch.softmax(scaled_logits, dim=0).cpu().numpy()
        else:
            probs = np.zeros(len(self.class_names))
            probs[0] = 1.0

        latency_ms = (time.perf_counter() - t0) * 1000

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_results = [
            {
                "character": self.class_names[idx],
                "confidence": float(probs[idx]),
                "index": int(idx),
                "is_digit": self.class_names[idx].isdigit(),
                "is_upper": self.class_names[idx].isupper(),
                "is_lower": self.class_names[idx].islower()
            }
            for idx in top_indices
        ]

        best = top_k_results[0]
        return {
            "best_character": best["character"],
            "best_confidence": best["confidence"],
            "latency_ms": latency_ms,
            "top_k": top_k_results,
            "all_probs": probs
        }

    def predict_word(self, segmented_chars: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Suy luận cả chuỗi ký tự từ danh sách các ký tự đã bóc tách."""
        word_result = ""
        total_time = 0.0
        char_details = []

        for item in segmented_chars:
            res = self.predict_single(item["tensor_input"], top_k=3)
            word_result += res["best_character"]
            total_time += res["latency_ms"]
            char_details.append({
                "char": res["best_character"],
                "confidence": res["best_confidence"],
                "bbox": item["bbox"],
                "top_k": res["top_k"]
            })

        return {
            "predicted_text": word_result,
            "total_latency_ms": total_time,
            "num_characters": len(segmented_chars),
            "characters": char_details
        }
