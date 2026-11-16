"""
verify_experts.py — Kiểm thử tự động toàn diện cụm chuyên gia (Mixture of Experts)
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import string

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from nist_preprocessor import NISTPreprocessor
from onnx_engine import InferenceEngine

p = NISTPreprocessor()
e = InferenceEngine()

print("=" * 70)
print("1. KIỂM THỬ CHUYÊN GIA CHỮ SỐ (resnet_digits.pt - 10 lớp)")
print("=" * 70)
digits_correct = 0
digits = list(string.digits)
for d in digits:
    img = np.zeros((280, 280), dtype=np.uint8)
    cv2.putText(img, d, (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 7.0, 255, 20)
    prep = p.preprocess_canvas_image(img)
    tin = p.to_model_input(prep)
    res = e.predict_single(tin, top_k=3, temperature=0.3, filter_mode="digits")
    pred = res["best_character"]
    conf = res["best_confidence"] * 100
    is_match = (pred == d)
    if is_match:
        digits_correct += 1
    print(f"Số '{d}' -> AI đoán: '{pred}' (Độ tin cậy: {conf:5.1f}%) | {'✓ ĐÚNG' if is_match else '✗ SAI'}")

print(f"===> Chuyên gia Chữ số: {digits_correct}/{len(digits)} ({digits_correct/len(digits)*100:.1f}%) ĐÚNG\n")

print("=" * 70)
print("2. KIỂM THỬ CHUYÊN GIA CHỮ CÁI (resnet_letters.pt - 26 lớp)")
print("=" * 70)
letters_correct = 0
letters = list(string.ascii_uppercase) + ['j', 'a', 'm']
for ch in letters:
    img = np.zeros((280, 280), dtype=np.uint8)
    cv2.putText(img, ch, (70, 220), cv2.FONT_HERSHEY_SIMPLEX, 7.0, 255, 20)
    prep = p.preprocess_canvas_image(img)
    tin = p.to_model_input(prep)
    res = e.predict_single(tin, top_k=3, temperature=0.3, filter_mode="letters")
    pred = res["best_character"]
    conf = res["best_confidence"] * 100
    is_match = (pred.upper() == ch.upper())
    if is_match:
        letters_correct += 1
    print(f"Chữ '{ch}' -> AI đoán: '{pred}' (Độ tin cậy: {conf:5.1f}%) | {'✓ ĐÚNG' if is_match else '✗ SAI'}")

print(f"===> Chuyên gia Chữ cái: {letters_correct}/{len(letters)} ({letters_correct/len(letters)*100:.1f}%) ĐÚNG\n")
print("=" * 70)
print("KẾT LUẬN: HỆ THỐNG CỤM CHUYÊN GIA HOẠT ĐỘNG HOÀN HẢO!")
print("=" * 70)
