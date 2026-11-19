"""CIFAR-10 datasets and reproducible train/validation/test DataLoaders."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Common CIFAR-10 channel statistics.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, eval_transform


def create_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    val_size: int = 5000,
    num_workers: int = 0,
    seed: int = 42,
    download: bool = True,
):
    """Return train, validation and test loaders.

    Validation indices are a deterministic subset of the original 50k training
    images. It uses evaluation transforms (no random augmentation).
    num_workers=0 is the safest default for Windows.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not 0 < val_size < 50000:
        raise ValueError("val_size must be between 1 and 49999")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    data_dir = Path(data_dir)
    train_transform, eval_transform = build_transforms()

    train_augmented = datasets.CIFAR10(
        root=str(data_dir), train=True, download=download, transform=train_transform
    )
    train_plain = datasets.CIFAR10(
        root=str(data_dir), train=True, download=download, transform=eval_transform
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_dir), train=False, download=download, transform=eval_transform
    )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(train_augmented), generator=generator).tolist()
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]

    train_dataset = Subset(train_augmented, train_indices)
    val_dataset = Subset(train_plain, val_indices)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
