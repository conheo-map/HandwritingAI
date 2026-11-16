import os
import sys
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from src.modern_resnet import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình HandwritingAI")
    parser.add_argument("--epochs", type=int, default=8, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch-size", type=int, default=256, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tốc độ học ban đầu")
    parser.add_argument("--split", type=str, default="balanced", choices=["byclass", "bymerge", "balanced"], help="Biến thể EMNIST")
    parser.add_argument("--save-dir", type=str, default="models", help="Thư mục lưu trọng số")
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Khởi động quá trình huấn luyện trên thiết bị: {device}")

    num_classes = 62 if args.split == "byclass" else 47
    save_dir = BASE_DIR / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_emnist_resnet.pt"

    # Data Augmentation phong phú chuẩn EMNIST
    # Lưu ý: EMNIST gốc trong torchvision cần xoay & lật trục
    train_transform = transforms.Compose([
        lambda img: transforms.functional.rotate(img, -90),
        lambda img: transforms.functional.hflip(img),
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        lambda img: transforms.functional.rotate(img, -90),
        lambda img: transforms.functional.hflip(img),
        transforms.ToTensor(),
    ])

    print(f"[*] Đang tải tập dữ liệu EMNIST ({args.split})...")
    data_dir = BASE_DIR / "data"
    full_train = datasets.EMNIST(root=str(data_dir), split=args.split, train=True, download=True, transform=train_transform)
    test_set = datasets.EMNIST(root=str(data_dir), split=args.split, train=False, download=True, transform=val_transform)

    # Chia Train / Val (90% / 10%)
    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"[✓] Train: {len(train_set):,} ảnh | Val: {len(val_set):,} ảnh | Test: {len(test_set):,} ảnh")

    # Khởi tạo mô hình
    model = create_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total*100:.2f}%"})

        scheduler.step()
        train_acc = correct / total * 100

        # Đánh giá trên tập Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        print(f"[*] Epoch {epoch} hoàn thành -> Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Lưu checkpoint tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": num_classes
            }, best_model_path)
            print(f"    [🏆] Đã lưu mô hình mới tốt nhất vào {best_model_path} (Val Acc: {val_acc:.2f}%)")

    print(f"\n[✓] Huấn luyện thành công! Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train()
