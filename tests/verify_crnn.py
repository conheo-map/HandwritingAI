import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from crnn_model import CRNNHandwritingRecognizer, CTCLabelConverter

def verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = BASE_DIR / "models" / "crnn_word_model.pt"
    if not ckpt_path.exists():
        print("Chưa tìm thấy checkpoint crnn_word_model.pt")
        return

    ckpt = torch.load(str(ckpt_path), map_location=device)
    converter = CTCLabelConverter(ckpt.get("charset"))
    model = CRNNHandwritingRecognizer(
        img_h=32,
        in_channels=1,
        num_classes=ckpt["num_classes"],
        hidden_size=ckpt.get("hidden_size", 256)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    test_words = ["h", "hi", "account", "balance", "hello", "system", "cat", "2026", "money", "AI"]
    print("=" * 65)
    print("KIỂM TRA THỰC NGHIỆM CRNN (LEFT-ALIGNED & ĐA DẠNG ĐỘ DÀI):")
    print("=" * 65)

    font_path = "C:/Windows/Fonts/segoesc.ttf"
    if not Path(font_path).exists():
        font_path = "C:/Windows/Fonts/comic.ttf"

    correct = 0
    for w in test_words:
        font = ImageFont.truetype(font_path, 28)
        img = Image.new("L", (320, 60), color=0)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), w, fill=255, font=font)
        arr = np.array(img)

        coords = cv2.findNonZero(arr)
        if coords is not None:
            x, y, bw, bh = cv2.boundingRect(coords)
            crop = arr[max(0, y-2):y+bh+2, max(0, x-2):x+bw+2]
        else:
            crop = arr

        h, bw = crop.shape
        scale = 26.0 / max(1, h)
        scaled_w = min(152, max(4, int(round(bw * scale))))
        resized = cv2.resize(crop, (scaled_w, 26))

        # CĂN LỀ TRÁI CHUẨN XÁC TẠI OFFSET_X = 4
        canvas = np.zeros((32, 160), dtype=np.uint8)
        canvas[3:29, 4:4+scaled_w] = resized

        tensor = torch.tensor(canvas / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode():
            log_probs = model(tensor)
            pred = converter.decode_greedy(log_probs)[0]

        is_match = pred.lower() == w.lower()
        if is_match:
            correct += 1
        status = "DUNG" if is_match else "SAI"
        print(f"Từ gốc: {w:10s} -> CRNN nhận diện: {pred:10s} | [{status}]")

    print(f"Kết quả: {correct}/{len(test_words)} ({correct/len(test_words)*100:.1f}%) chính xác!")
    print("=" * 65)

if __name__ == "__main__":
    verify()
