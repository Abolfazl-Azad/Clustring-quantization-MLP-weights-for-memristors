"""Evaluate a saved VGG8 checkpoint on the CIFAR-10 test set."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from data import CIFAR10_CLASSES, create_dataloaders
from model import VGG8
from utils import load_checkpoint, select_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate VGG8 on CIFAR-10 test set")
    parser.add_argument("--checkpoint", default="outputs/vgg8_cifar10/best.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_size=5000,
        num_workers=args.num_workers,
        download=True,
    )

    checkpoint = load_checkpoint(args.checkpoint, device)
    model = VGG8(num_classes=len(checkpoint.get("classes", CIFAR10_CLASSES))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for target, pred in zip(labels.cpu(), preds.cpu()):
                confusion[int(target), int(pred)] += 1

    print(f"Test loss: {total_loss / total:.4f}")
    print(f"Test accuracy: {100.0 * correct / total:.2f}% ({correct}/{total})")
    print("\nPer-class accuracy:")
    classes = checkpoint.get("classes", CIFAR10_CLASSES)
    for i, class_name in enumerate(classes):
        denom = int(confusion[i].sum())
        acc = 100.0 * int(confusion[i, i]) / denom if denom else 0.0
        print(f"  {class_name:12s}: {acc:6.2f}% ({int(confusion[i, i])}/{denom})")


if __name__ == "__main__":
    main()
