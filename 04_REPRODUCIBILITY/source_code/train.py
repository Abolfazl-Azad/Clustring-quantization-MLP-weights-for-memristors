"""Train the VGG8-style model on CIFAR-10."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, create_dataloaders
from model import VGG8, count_trainable_parameters
from utils import select_device, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += batch_n

    if total_seen == 0:
        raise RuntimeError("DataLoader produced zero samples.")
    return total_loss / total_seen, 100.0 * total_correct / total_seen


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG8 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/vgg8_cifar10")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive")

    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = VGG8(num_classes=len(CIFAR10_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(val_loader.dataset):,}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    history_path = output_dir / "history.csv"
    best_val_accuracy = -1.0
    best_val_loss = float("inf")

    with history_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, device, optimizer=optimizer
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, device, optimizer=None
            )
            scheduler.step()
            elapsed = time.perf_counter() - start
            current_lr = optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_accuracy": f"{train_acc:.3f}",
                "val_loss": f"{val_loss:.6f}",
                "val_accuracy": f"{val_acc:.3f}",
                "learning_rate": f"{current_lr:.8g}",
            }
            writer.writerow(row)
            csv_file.flush()

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train loss {train_loss:.4f}, acc {train_acc:.2f}% | "
                f"val loss {val_loss:.4f}, acc {val_acc:.2f}% | "
                f"lr {current_lr:.2e} | {elapsed:.1f}s"
            )

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "classes": CIFAR10_CLASSES,
                "mean": CIFAR10_MEAN,
                "std": CIFAR10_STD,
                "architecture": "VGG8_CIFAR10_5conv_3fc_bn",
            }
            torch.save(checkpoint, output_dir / "last.pt")

            if val_acc > best_val_accuracy or (
                val_acc == best_val_accuracy and val_loss < best_val_loss
            ):
                best_val_accuracy = val_acc
                best_val_loss = val_loss
                torch.save(checkpoint, output_dir / "best.pt")
                print("  Saved new best checkpoint.")

    print(f"\nFinished. Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
