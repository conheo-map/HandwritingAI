import os
import sys
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from modern_resnet import create_model
from chars74k_loader import Chars74kDataset
from font_synthesizer import SyntheticFontDataset


def emnist_transform():
    return transforms.Compose([
        lambda img: transforms.functional.rotate(img, -90),
        lambda img: transforms.functional.hflip(img),
        transforms.ToTensor(),
    ])


def train_expert(
    mode: str = "digits",      # "digits", "letters", "balanced"
    num_classes: int = 10,
    model_name: str = "resnet_digits.pt",
    epochs: int = 4,
    batch_size: int = 256,
    lr: float = 0.0003,
    load_checkpoint: Optional[str] = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}")
    print(f"BẮT ĐẦU HUẤN LUYỆN CHUYÊN GIA: {mode.upper()} ({num_classes} lớp) TRÊN {device}")
    print(f"{'='*75}")

    data_dir = BASE_DIR / "data"

    # 1. Nạp EMNIST Dataset
    emnist_split = "digits" if mode == "digits" else ("letters" if mode == "letters" else "balanced")
    emnist_train = datasets.EMNIST(
        root=str(data_dir),
        split=emnist_split,
        train=True,
        download=False,
        transform=emnist_transform()
    )

    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, ds, offset=1):
            self.ds = ds
            self.offset = offset
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            x, y = self.ds[idx]
            return x, y - self.offset

    if mode == "letters":
        emnist_train = RemappedDataset(emnist_train, offset=1)

    # 2. Nạp Chars74k Dataset
    max_samples = 400 if mode == "balanced" else 500
    chars74k_ds = Chars74kDataset(
        base_dir=str(data_dir / "chars74k"),
        mode=mode,
        include_hnd=True,
        include_fnt=True,
        max_samples_per_class=max_samples
    )

    # 3. Nạp Synthetic Font Dataset
    font_samples = 150 if mode == "digits" else (100 if mode == "letters" else 60)
    synth_ds = SyntheticFontDataset(mode=mode, samples_per_class=font_samples)

    # Kết hợp toàn bộ các nguồn dữ liệu
    full_dataset = ConcatDataset([emnist_train, chars74k_ds, synth_ds])
    print(f"Tổng số mẫu huấn luyện đa nguồn (EMNIST + Chars74k + Synthetic Fonts): {len(full_dataset):,} mẫu")

    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Khởi tạo mô hình ResNet
    model = create_model(num_classes=num_classes)
    if load_checkpoint:
        ckpt_path = BASE_DIR / "models" / load_checkpoint
        if ckpt_path.exists():
            print(f"Đang nạp trọng số nền tảng từ {load_checkpoint}...")
            sd = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(sd["model_state_dict"])

    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_val_acc = 0.0
    out_path = BASE_DIR / "models" / model_name

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        # Đánh giá trên tập Validation
        model.eval()
        val_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                v_correct += preds.eq(labels).sum().item()
                v_total += labels.size(0)

        val_acc = v_correct / v_total
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Thời gian: {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
                "num_classes": num_classes
            }, str(out_path))
            print(f"  [+] Đã cập nhật trọng số tối ưu (Val Acc: {val_acc*100:.2f}%) vào {model_name}")

    print(f"\n[✓] Hoàn thành huấn luyện {mode.upper()}. Best Val Acc: {best_val_acc*100:.2f}%\n")


if __name__ == "__main__":
    # 1. Chữ số (10 lớp)
    train_expert("digits", 10, "resnet_digits.pt", epochs=3, lr=0.0003, load_checkpoint="resnet_digits.pt")
    # 2. Chữ cái (26 lớp)
    train_expert("letters", 26, "resnet_letters.pt", epochs=3, lr=0.0003, load_checkpoint="resnet_letters.pt")
    # 3. Toàn bộ (47 lớp Balanced)
    train_expert("balanced", 47, "best_emnist_resnet.pt", epochs=3, lr=0.0003, load_checkpoint="best_emnist_resnet.pt")
