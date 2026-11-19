"""Shared helpers for training and evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def load_checkpoint(path: str | Path, device: torch.device):
    """Load a checkpoint produced by this project (trusted local file)."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # compatibility with older PyTorch releases
        return torch.load(path, map_location=device)
