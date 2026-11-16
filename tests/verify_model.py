"""
verify_model.py — Script Kiểm Thử So Sánh Transpose vs Non-Transpose
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from nist_preprocessor import NISTPreprocessor
from onnx_engine import InferenceEngine

p = NISTPreprocessor()
e = InferenceEngine()

test_cases = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'J', 'M', 'R', 'j'
]

print("=" * 60)
print("   KIEM TRA MO HINH KHONG TRANSPOSE (NORMAL ORIENTATION)")
print("=" * 60)

correct = 0
for ch in test_cases:
    img = np.zeros((280, 280), dtype=np.uint8)
    cv2.putText(img, ch, (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 7.0, 255, 20)
    prep = p.preprocess_canvas_image(img)
    
    # Input WITHOUT transpose:
    model_in = np.expand_dims(np.expand_dims(prep, axis=0), axis=0)
    with torch.no_grad():
        tensor = torch.from_numpy(model_in).float().to(e.device)
        logits = e.torch_model(tensor)[0]
        probs = torch.softmax(logits / 0.4, dim=0).cpu().numpy()
        pred_idx = np.argmax(probs)
        pred = e.class_names[pred_idx]
        conf = probs[pred_idx] * 100

    is_match = (pred.upper() == ch.upper())
    if is_match:
        correct += 1
    status = "✓ DUNG" if is_match else "✗ SAI"
    print(f"Ký tự '{ch}' -> AI đoán: '{pred}' (Độ tin cậy: {conf:5.1f}%) | {status}")

accuracy = correct / len(test_cases) * 100
print("=" * 60)
print(f"===> KET QUA: {correct}/{len(test_cases)} ({accuracy:.1f}%) KÝ TỰ KHỚP CHÍNH XÁC!")
print("=" * 60)
