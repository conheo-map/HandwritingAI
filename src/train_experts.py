"""
train_experts.py — Huấn luyện cụm mô hình chuyên gia (Mixture of Experts: Digits & Letters)
Kiến trúc: Deep ResNet-18 với Squeeze-and-Excitation (SE) Attention & PyTorch AMP
"""

import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.modern_resnet import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = BASE_DIR / "models"
save_dir.mkdir(parents=True, exist_ok=True)
data_dir = BASE_DIR / "data"

transform_train = transforms.Compose([
    lambda img: transforms.functional.rotate(img, -90),
    lambda img: transforms.functional.hflip(img),
    transforms.RandomAffine(degrees=8, translate=(0.06, 0.06), scale=(0.96, 1.04)),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    lambda img: transforms.functional.rotate(img, -90),
    lambda img: transforms.functional.hflip(img),
    transforms.ToTensor(),
])


def train_single_expert(expert_name: str, split: str, num_classes: int, epochs: int = 4, batch_size: int = 256):
    print("=" * 70)
    print(f"[*] BAT DAU HUAN LUYEN CHUYEN GIA: {expert_name.upper()} ({num_classes} LOP)")
    print("=" * 70)
    print(f"Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load dataset
    full_train = datasets.EMNIST(root=str(data_dir), split=split, train=True, download=True, transform=transform_train)
    val_set_raw = datasets.EMNIST(root=str(data_dir), split=split, train=False, download=True, transform=transform_val)

    # Nếu là letters thì label từ 1..26 -> cần shift về 0..25
    is_letters = (split == "letters")

    class ShiftedDataset(torch.utils.data.Dataset):
        def __init__(self, ds, shift_labels=False):
            self.ds = ds
            self.shift = shift_labels

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            img, label = self.ds[idx]
            if self.shift:
                label = label - 1
            return img, label

    train_data = ShiftedDataset(full_train, shift_labels=is_letters)
    val_data = ShiftedDataset(val_set_raw, shift_labels=is_letters)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Dataset: {len(train_data):,} ảnh Train | {len(val_data):,} ảnh Validation")

    # Khởi tạo Deep ResNet-18
    model = create_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    save_path = save_dir / f"resnet_{expert_name}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"[{expert_name.upper()}] Epoch {epoch}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total*100:.2f}%"})

        scheduler.step()
        train_acc = correct / total * 100
        epoch_time = time.perf_counter() - t0

        # Đánh giá Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        print(f"[*] Epoch {epoch} ({epoch_time:.1f}s) -> Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": num_classes,
                "expert_type": expert_name
            }, save_path)
            print(f"    [🏆] Đã lưu model tốt nhất vào: {save_path.name} (Val Acc: {val_acc:.2f}%)")

    print(f"\n[✓] Hoàn thành chuyên gia {expert_name.upper()}! Best Val Acc: {best_val_acc:.2f}%\n")
    return best_val_acc


def main():
    print("KHỞI ĐỘNG TIẾN TRÌNH HUẤN LUYỆN CỤM CHUYÊN GIA RESNET (GPU RTX 2050)")
    t_start = time.perf_counter()

    # 1. Chuyên gia chữ cái (26 classes, 124,800 mẫu)
    acc_letters = train_single_expert("letters", split="letters", num_classes=26, epochs=5, batch_size=256)

    # 2. Chuyên gia chữ số (10 classes, 240,000 mẫu)
    acc_digits = train_single_expert("digits", split="digits", num_classes=10, epochs=4, batch_size=256)

    total_time = (time.perf_counter() - t_start) / 60
    print("=" * 70)
    print("HOÀN TẤT HUẤN LUYỆN TOÀN BỘ CỤM CHUYÊN GIA:")
    print(f"  1. resnet_letters.pt (26 lớp A-Z): {acc_letters:.2f}% Val Accuracy")
    print(f"  2. resnet_digits.pt  (10 lớp 0-9): {acc_digits:.2f}% Val Accuracy")
    print(f"Tổng thời gian: {total_time:.2f} phút")
    print("=" * 70)


if __name__ == "__main__":
    main()
