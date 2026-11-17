import sys
from pathlib import Path
import torch
import torch.nn as nn
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from crnn_model import CRNNHandwritingRecognizer, CTCLabelConverter


def test_crnn_forward_shape():
    """Kiểm tra shape tensor forward pass và tính toán CTC loss."""
    model = CRNNHandwritingRecognizer(img_h=32, in_channels=1, num_classes=68, hidden_size=128)
    model.eval()

    # Giả lập batch 4 ảnh từ viết tay kích thước [B=4, C=1, H=32, W=160]
    dummy_input = torch.randn(4, 1, 32, 160)
    with torch.inference_mode():
        log_probs = model(dummy_input)

    # Sau CNN và stride: W=160 -> W'=39 time steps
    t, b, num_classes = log_probs.shape
    assert b == 4, f"Batch size mong đợi 4, thực tế: {b}"
    assert num_classes == 68, f"Num classes mong đợi 68, thực tế: {num_classes}"
    assert t > 0, f"Time steps phải lớn hơn 0, thực tế: {t}"
    assert not torch.isnan(log_probs).any(), "Phát hiện NaN trong log_probs!"


def test_ctc_loss_calculation():
    """Kiểm tra tính toán loss và gradient không phát sinh lỗi tràn số hoặc NaN."""
    model = CRNNHandwritingRecognizer(img_h=32, in_channels=1, num_classes=68, hidden_size=64)
    model.train()

    converter = CTCLabelConverter()
    texts = ["account", "balance", "AI", "test"]
    targets, target_lengths = converter.encode(texts)

    dummy_input = torch.randn(4, 1, 32, 160, requires_grad=True)
    log_probs = model(dummy_input)  # [T, B, C]
    t, b, c = log_probs.shape

    input_lengths = torch.full(size=(b,), fill_value=t, dtype=torch.long)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    assert not torch.isnan(loss), "CTC Loss bị NaN!"
    assert not torch.isinf(loss), "CTC Loss bị Inf!"
    assert loss.item() > 0.0

    loss.backward()
    assert dummy_input.grad is not None, "Gradient không được truyền ngược về input!"


def test_ctc_label_converter_decode():
    """Kiểm tra mã hóa chuỗi và giải mã CTC Greedy Decoder."""
    converter = CTCLabelConverter()
    text = "account"
    targets, lengths = converter.encode([text])

    assert lengths.item() == len(text)

    # Giả lập ma trận xác suất lý tưởng cho từ 'AI'
    # Blank = 0, 'A' = converter.char2idx['A'], 'I' = converter.char2idx['I']
    a_idx = converter.char2idx['A']
    i_idx = converter.char2idx['I']
    num_classes = len(converter.charset) + 1

    # Tạo sequence: [A, A, Blank, I, I]
    seq_tokens = [a_idx, a_idx, 0, i_idx, i_idx]
    time_steps = len(seq_tokens)
    dummy_logits = torch.full((time_steps, 1, num_classes), -10.0)
    for t, tok in enumerate(seq_tokens):
        dummy_logits[t, 0, tok] = 10.0  # Đặt xác suất cao nhất

    log_probs = torch.nn.functional.log_softmax(dummy_logits, dim=-1)
    decoded = converter.decode_greedy(log_probs)

    assert decoded == ["AI"], f"Greedy decode mong đợi 'AI', thực tế: '{decoded[0]}'"
