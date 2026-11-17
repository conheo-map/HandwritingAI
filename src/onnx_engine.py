
import string
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Tuple

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
                    if "num_classes" in sd:
                        num_classes = sd["num_classes"]
                    elif "classes" in sd:
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

        # Tải cụm chuyên gia (Mixture of Experts)
        self.expert_digits = None
        self.digits_classes = list(string.digits)

        self.expert_letters = None
        self.letters_classes = list(string.ascii_uppercase)

        # Chuyên gia Chữ số (10 lớp)
        dig_path = models_dir / "resnet_digits.pt"
        if dig_path.exists():
            try:
                from modern_resnet import create_model
                sd = torch.load(str(dig_path), map_location=self.device)
                m = create_model(num_classes=10)
                m.load_state_dict(sd["model_state_dict"])
                m.to(self.device).eval()
                self.expert_digits = m
                print(f"[✓] Đã kích hoạt Chuyên Gia Chữ Số (10 lớp) trên {self.device}")
            except Exception as e:
                pass

        # Chuyên gia Chữ cái (26 lớp)
        let_path = models_dir / "resnet_letters.pt"
        if let_path.exists():
            try:
                from modern_resnet import create_model
                sd = torch.load(str(let_path), map_location=self.device)
                m = create_model(num_classes=26)
                m.load_state_dict(sd["model_state_dict"])
                m.to(self.device).eval()
                self.expert_letters = m
                print(f"[✓] Đã kích hoạt Chuyên Gia Chữ Cái (26 lớp) trên {self.device}")
            except Exception as e:
                pass

        # ── NẠP MÔ HÌNH CRNN + BiLSTM + CTC (CHO TỪ LIỀN NÉT) ──
        self.crnn_model = None
        self.crnn_converter = None
        crnn_path = models_dir / "crnn_word_model.pt"
        if crnn_path.exists():
            try:
                from crnn_model import CRNNHandwritingRecognizer, CTCLabelConverter
                sd = torch.load(str(crnn_path), map_location=self.device)
                self.crnn_converter = CTCLabelConverter(sd.get("charset"))
                m = CRNNHandwritingRecognizer(
                    img_h=32,
                    in_channels=1,
                    num_classes=sd["num_classes"],
                    hidden_size=sd.get("hidden_size", 256)
                )
                m.load_state_dict(sd["model_state_dict"])
                m.to(self.device).eval()
                self.crnn_model = m
                print(f"[✓] Đã kích hoạt Mô hình CRNN (BiLSTM + CTC) trên {self.device}")
            except Exception as e:
                print(f"[!] Lỗi nạp mô hình CRNN: {e}")

    def predict_single(
        self, 
        input_tensor: np.ndarray, 
        top_k: int = 5, 
        temperature: float = 0.5,
        filter_mode: str = "all"
    ) -> Dict[str, Any]:
        """
        Suy luận 1 ký tự từ ma trận đầu vào (1, 1, 28, 28).
        Tự động định tuyến (Dynamic Routing) đến mô hình chuyên gia:
          - 'digits' -> Chuyên gia Chữ số (10 lớp)
          - 'letters' -> Chuyên gia Chữ cái (26 lớp)
          - 'all' -> Mô hình đa năng EMNIST (47 lớp)
        """
        t0 = time.perf_counter()

        active_model = self.torch_model
        active_classes = self.class_names
        is_expert = False

        if filter_mode == "digits" and self.expert_digits is not None:
            active_model = self.expert_digits
            active_classes = self.digits_classes
            is_expert = True
        elif filter_mode == "letters" and self.expert_letters is not None:
            active_model = self.expert_letters
            active_classes = self.letters_classes
            is_expert = True

        if active_model is not None:
            norm_2d = input_tensor[0, 0]
            if getattr(self, "needs_transpose", False) and not is_expert:
                norm_2d = np.transpose(norm_2d)
            model_in = np.expand_dims(np.expand_dims(norm_2d, axis=0), axis=0)

            with torch.no_grad():
                tensor = torch.from_numpy(model_in).float().to(self.device)
                logits = active_model(tensor)[0]

                # Nếu là model tổng quát mà người dùng chọn filter, thực hiện mask
                if not is_expert:
                    if filter_mode == "digits":
                        logits[10:] = -1e9
                    elif filter_mode == "letters":
                        logits[:10] = -1e9

                T = max(0.1, min(temperature, 2.0))
                scaled_logits = logits / T
                probs = torch.softmax(scaled_logits, dim=0).cpu().numpy()
        else:
            probs = np.zeros(len(active_classes))
            probs[0] = 1.0

        latency_ms = (time.perf_counter() - t0) * 1000

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_results = [
            {
                "character": active_classes[idx],
                "confidence": float(probs[idx]),
                "index": int(idx),
                "is_digit": active_classes[idx].isdigit(),
                "is_upper": active_classes[idx].isupper(),
                "is_lower": active_classes[idx].islower()
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

    def predict_word(
        self, 
        segmented_chars: List[Dict[str, Any]], 
        temperature: float = 0.5, 
        filter_mode: str = "all"
    ) -> Dict[str, Any]:
        """Suy luận cả chuỗi ký tự từ danh sách các ký tự đã bóc tách."""
        word_result = ""
        total_time = 0.0
        char_details = []

        for item in segmented_chars:
            res = self.predict_single(
                item["tensor_input"], 
                top_k=3, 
                temperature=temperature, 
                filter_mode=filter_mode
            )
            ch = res["best_character"]
            if "char_crop" in item and ch in ['I', 'J', 'l', '1']:
                from nist_preprocessor import NISTPreprocessor
                if not hasattr(self, "_nist_prep"):
                    self._nist_prep = NISTPreprocessor()
                ch, _ = self._nist_prep.refine_dotted_character(item["char_crop"], ch)

            word_result += ch
            total_time += res["latency_ms"]
            char_details.append({
                "char": ch,
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

    def preprocess_word_for_crnn(self, image_input) -> Tuple[np.ndarray, np.ndarray]:
        """
        Chuẩn hóa ảnh nguyên cả từ viết tay thành ma trận [32, 160] cho mô hình CRNN.
        Bảo toàn toàn bộ nét nối uốn lượn (Ligatures) mà không cắt ký tự.
        """
        import cv2
        from PIL import Image

        if isinstance(image_input, Image.Image):
            arr = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            arr = image_input.copy()
        else:
            arr = cv2.imread(str(image_input))

        # Chuyển xám
        if len(arr.shape) == 3:
            if arr.shape[2] == 4:
                alpha = arr[:, :, 3]
                rgb = arr[:, :, :3]
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                if np.max(alpha) > 0 and np.mean(gray) > 200:
                    gray = 255 - gray
            else:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr.copy()

        # Nền đen chữ trắng
        if np.mean(gray) > 127:
            gray = 255 - gray

        # Lọc nhiễu nhẹ
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        coords = cv2.findNonZero(blurred)

        if coords is None:
            empty = np.zeros((32, 160), dtype=np.uint8)
            return empty, empty

        x, y, w, h = cv2.boundingRect(coords)
        pad = 2
        y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        crop = gray[y1:y2, x1:x2]

        ch_h, ch_w = crop.shape
        # Co giãn chiều cao về 26px để đệm viền 3px trên dưới thành 32px
        scale = 26.0 / max(1, ch_h)
        scaled_w = min(152, max(4, int(round(ch_w * scale))))
        resized = cv2.resize(crop, (scaled_w, 26), interpolation=cv2.INTER_AREA)

        # CĂN LỀ TRÁI TẠI OFFSET_X = 4 (Khớp 100% với nguyên tắc giải mã CTC)
        canvas = np.zeros((32, 160), dtype=np.uint8)
        offset_y = 3
        offset_x = 4
        canvas[offset_y:offset_y+26, offset_x:offset_x+scaled_w] = resized

        return canvas, crop

    def predict_cursive_word(self, image_input) -> Dict[str, Any]:
        """
        Nhận diện cả từ viết tay liền nét (Cursive Word) bằng kiến trúc CRNN + BiLSTM + CTC.
        Hoàn toàn Segmentation-Free (không cắt chữ).
        """
        if self.crnn_model is None:
            return {
                "predicted_text": "(Chưa nạp mô hình CRNN)",
                "latency_ms": 0.0,
                "preprocessed_img": np.zeros((32, 160), dtype=np.uint8)
            }

        t0 = time.perf_counter()
        canvas_32x160, raw_crop = self.preprocess_word_for_crnn(image_input)

        norm_tensor = torch.tensor(
            canvas_32x160 / 255.0, 
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            log_probs = self.crnn_model(norm_tensor)
            decoded_texts = self.crnn_converter.decode_greedy(log_probs)

        latency_ms = (time.perf_counter() - t0) * 1000
        pred_text = decoded_texts[0] if decoded_texts else ""

        return {
            "predicted_text": pred_text,
            "latency_ms": latency_ms,
            "preprocessed_img": canvas_32x160,
            "crop_img": raw_crop
        }
