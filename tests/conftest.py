"""Shared pytest fixtures.

All fixtures here are CPU-only, offline, and fast. Nothing in the default test run
downloads weights or touches a GPU -- tests that do are marked `slow`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from text2video.data.moving_mnist import BouncingMNISTGenerator, generate_dataset


@pytest.fixture(scope="session")
def fake_sprites() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic stand-in for MNIST: one distinct shape per class, no download.

    Each class gets a filled square of a different size, so class identity is
    recoverable from pixels without needing the real MNIST files.
    """
    sprites = np.zeros((100, 28, 28), dtype=np.uint8)
    labels = np.arange(100, dtype=np.int64) % 10
    for i, label in enumerate(labels):
        size = 6 + label * 2  # 6..24 px squares
        offset = (28 - size) // 2
        sprites[i, offset : offset + size, offset : offset + size] = 255
    return sprites, labels


@pytest.fixture
def generator(fake_sprites) -> BouncingMNISTGenerator:
    sprites, labels = fake_sprites
    return BouncingMNISTGenerator(
        sprites=sprites, labels=labels, canvas_size=64, num_frames=16
    )


@pytest.fixture
def tiny_dataset_dir(tmp_path: Path, generator: BouncingMNISTGenerator) -> Path:
    """A small on-disk dataset split, laid out exactly like the real one."""
    frames, captions, metas = generate_dataset(generator, num_clips=12, seed=7)
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True)
    np.save(split_dir / "frames.npy", frames)
    (split_dir / "captions.json").write_text(json.dumps(captions), encoding="utf-8")
    (split_dir / "metadata.json").write_text(json.dumps(metas), encoding="utf-8")
    return split_dir
