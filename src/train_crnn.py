import os
import sys
import time
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from crnn_model import CRNNHandwritingRecognizer, CTCLabelConverter
from cursive_word_generator import SyntheticCursiveWordDataset, pad_collate_fn


def set_seed(seed: int = 42):
    """Cố định seed bảo đảm tính tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_crnn(
    num_samples: int = 15000,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 0.0008,
    model_name: str = "crnn_word_model.pt"
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*75}")
    print(f"KHỞI TẠO HUẤN LUYỆN MÔ HÌNH CRNN + BiLSTM + CTC TRÊN: {device}")
    print(f"{'='*75}")

    # 1. Khởi tạo dữ liệu từ ngữ viết tay đa dạng nét
    full_dataset = SyntheticCursiveWordDataset(num_samples=num_samples, img_h=32, img_w=160)
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    converter = CTCLabelConverter()
    num_classes = len(converter.charset) + 1

    # 2. Khởi tạo mạng CRNN
    model = CRNNHandwritingRecognizer(
        img_h=32,
        in_channels=1,
        num_classes=num_classes,
        hidden_size=256,
        num_lstm_layers=2,
        dropout=0.2
    )
    model.to(device)

    # 3. CTC Loss & Optimizer
    # blank=0 là quy chuẩn CTC của chúng ta
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / model_name

    best_val_acc = 0.0

    print(f"Tổng số tham số mô hình: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Số lượng mẫu huấn luyện: {train_size:,} | Mẫu kiểm định: {val_size:,}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        total_train_words = 0
        t0 = time.time()

        for batch_idx, (images, targets, target_lengths, texts) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass với Automatic Mixed Precision (AMP)
            if scaler:
                with torch.amp.autocast('cuda'):
                    log_probs = model(images)  # [T, B, Num_Classes]
                    t, b, _ = log_probs.shape
                    input_lengths = torch.full(size=(b,), fill_value=t, dtype=torch.long, device=device)
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs = model(images)
                t, b, _ = log_probs.shape
                input_lengths = torch.full(size=(b,), fill_value=t, dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            train_loss += loss.item() * len(texts)
            total_train_words += len(texts)

        scheduler.step()
        avg_train_loss = train_loss / max(1, total_train_words)

        # ── ĐÁNH GIÁ TRÊN TẬP VALIDATION BẰNG INFERENCE MODE ──
        model.eval()
        val_loss = 0.0
        val_correct_words = 0
        total_val_words = 0

        with torch.inference_mode():
            for images, targets, target_lengths, texts in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                target_lengths = target_lengths.to(device, non_blocking=True)

                log_probs = model(images)
                t, b, _ = log_probs.shape
                input_lengths = torch.full(size=(b,), fill_value=t, dtype=torch.long, device=device)
                v_loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss += v_loss.item() * len(texts)

                # Giải mã Greedy CTC
                decoded = converter.decode_greedy(log_probs)
                for pred, gt in zip(decoded, texts):
                    # So khớp không phân biệt hoa thường để đánh giá độ chính xác từ vựng
                    if pred.strip().lower() == gt.strip().lower():
                        val_correct_words += 1
                total_val_words += len(texts)

        avg_val_loss = val_loss / max(1, total_val_words)
        val_word_acc = val_correct_words / max(1, total_val_words)
        dt = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Word Acc: {val_word_acc*100:.2f}% | Time: {dt:.1f}s")

        if val_word_acc > best_val_acc or epoch == epochs:
            best_val_acc = val_word_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_word_acc,
                "epoch": epoch,
                "charset": converter.charset,
                "num_classes": num_classes,
                "hidden_size": 256
            }, str(out_path))
            print(f"  [+] Đã lưu trọng số tối ưu (Word Acc: {val_word_acc*100:.2f}%) vào {model_name}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[✓] Huấn luyện thành công CRNN! Best Word Accuracy: {best_val_acc*100:.2f}%\n")


if __name__ == "__main__":
    train_crnn(num_samples=18000, epochs=10, batch_size=64, lr=0.0008)
