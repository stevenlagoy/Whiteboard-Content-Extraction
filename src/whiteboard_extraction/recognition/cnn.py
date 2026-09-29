"""Custom CNN recognition backend."""

import numpy as np
import torch
import torch.nn as nn

from whiteboard_extraction.recognition.base import RecognizedBlock


class LineCNN(nn.Module):
    """A CNN over a cropped line/region image rather than a single character."""

    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class CustomCNNRecognizer:
    """Recognition backend wrapping a trained LineCNN checkpoint."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.model = None  # load checkpoint here once training exists

    def recognize(self, timestamp: float, image: np.ndarray) -> list[RecognizedBlock]:
        raise NotImplementedError("Load the trained checkpoint and run inference here.")