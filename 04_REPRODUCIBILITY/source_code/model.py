"""VGG8-style CNN for CIFAR-10 (32x32 RGB images).

This project defines one explicit VGG8 variant: 5 convolution layers + 3
fully-connected layers. BatchNorm is used after each convolution.
"""

from __future__ import annotations

import torch
from torch import nn


class VGG8(nn.Module):
    """VGG8-like classifier for 10 CIFAR-10 classes.

    Input:  [N, 3, 32, 32]
    Output: [N, 10] logits (do NOT apply Softmax before CrossEntropyLoss).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.features = nn.Sequential(
            # Block 1: 32x32x3 -> 16x16x64
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16x16x64 -> 8x8x192
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 8x8x192 -> 8x8x384 -> 8x8x256 -> 4x4x256
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    net = VGG8()
    sample = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        features = net.forward_features(sample)
        logits = net(sample)
    print(f"Feature shape: {tuple(features.shape)}")
    print(f"Logits shape:  {tuple(logits.shape)}")
    print(f"Trainable parameters: {count_trainable_parameters(net):,}")
    assert tuple(features.shape) == (2, 256, 4, 4)
    assert tuple(logits.shape) == (2, 10)
