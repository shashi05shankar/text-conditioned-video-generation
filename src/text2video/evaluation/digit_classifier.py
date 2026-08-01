"""A small MNIST CNN used as an *independent* judge of generated digit identity.

This is deliberately not part of the generative models and shares no weights with them.
It is trained once on real MNIST and then only ever used in evaluation, which is what
makes the structured-grounding metric independent of CLIP and of the generators.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """Small conv net over 28x28 grayscale digits. ~200k params, seconds to train."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 14 -> 7
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def crop_digit(frame: np.ndarray, size: int = 28) -> np.ndarray:
    """Crop a `size` x `size` window centred on the frame's brightest mass.

    Generated frames are 64x64 with the digit somewhere inside; the classifier expects a
    centred 28x28 MNIST-style crop. Falls back to a centre crop when the frame is empty.
    """
    height, width = frame.shape
    total = float(frame.sum())
    if total <= 0:
        cy, cx = height // 2, width // 2
    else:
        ys, xs = np.nonzero(frame)
        weights = frame[ys, xs].astype(np.float64)
        cx = int(round((xs * weights).sum() / total))
        cy = int(round((ys * weights).sum() / total))

    half = size // 2
    x0 = int(np.clip(cx - half, 0, max(0, width - size)))
    y0 = int(np.clip(cy - half, 0, max(0, height - size)))
    crop = frame[y0 : y0 + size, x0 : x0 + size]

    if crop.shape != (size, size):  # frame smaller than the crop window
        padded = np.zeros((size, size), dtype=frame.dtype)
        padded[: crop.shape[0], : crop.shape[1]] = crop
        crop = padded
    return crop


@torch.no_grad()
def classify_frames(
    model: DigitCNN, frames: np.ndarray, device: torch.device | str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Classify a stack of (N, 64, 64) uint8 frames.

    Returns (predicted_labels, confidences).
    """
    model.eval()
    crops = np.stack([crop_digit(frame) for frame in frames]).astype(np.float32)
    # MNIST normalisation: the classifier is trained on [0, 1] images.
    tensor = torch.from_numpy(crops / 255.0).unsqueeze(1).to(device)
    probabilities = F.softmax(model(tensor), dim=1)
    confidences, predictions = probabilities.max(dim=1)
    return predictions.cpu().numpy(), confidences.cpu().numpy()


def load_digit_classifier(
    path: str | Path, device: torch.device | str = "cpu"
) -> DigitCNN:
    model = DigitCNN().to(device)
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model
