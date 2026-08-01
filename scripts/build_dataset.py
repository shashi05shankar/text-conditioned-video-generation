"""Generate the Bouncing MNIST + captions dataset and cache it to disk.

Run once before training:

    python scripts/build_dataset.py --config configs/dataset.yaml

Writes, for each split, into <output_root>/<split>/:
    frames.npy      (N, T, H, W) uint8
    captions.json   list[str]
    metadata.json   list[dict]  -- exact ground truth per clip
    stats.json      summary counts used for the dataset section of the README
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text2video.core.config import load_config  # noqa: E402
from text2video.data.moving_mnist import (  # noqa: E402
    BouncingMNISTGenerator,
    generate_dataset,
    load_mnist_sprites,
)


def build_split(
    name: str,
    split_cfg,
    canvas_size: int,
    num_frames: int,
    two_digit_prob: float,
    bounce_prob: float,
    mnist_root: Path,
    output_root: Path,
) -> dict:
    print(f"\n[{name}] generating {split_cfg.num_clips} clips "
          f"(sprites from MNIST {'train' if split_cfg.mnist_train else 'test'} split)")

    sprites, labels = load_mnist_sprites(mnist_root, train=split_cfg.mnist_train)
    generator = BouncingMNISTGenerator(
        sprites=sprites,
        labels=labels,
        canvas_size=canvas_size,
        num_frames=num_frames,
        two_digit_prob=two_digit_prob,
        bounce_prob=bounce_prob,
    )

    frames, captions, metas = generate_dataset(
        generator, num_clips=split_cfg.num_clips, seed=split_cfg.seed
    )

    split_dir = output_root / name
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "frames.npy", frames)
    (split_dir / "captions.json").write_text(json.dumps(captions), encoding="utf-8")
    (split_dir / "metadata.json").write_text(json.dumps(metas), encoding="utf-8")

    stats = summarise(captions, metas)
    (split_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    size_mb = frames.nbytes / 1024 / 1024
    print(f"[{name}] frames {frames.shape} uint8 = {size_mb:.1f} MB -> {split_dir}")
    print(f"[{name}] unique captions: {stats['unique_captions']}/{stats['num_clips']} "
          f"({stats['unique_caption_ratio']:.1%})")
    print(f"[{name}] bounced: {stats['clips_with_bounce']} | "
          f"two-digit: {stats['two_digit_clips']}")
    print(f"[{name}] example: {captions[0]}")
    return stats


def summarise(captions: list[str], metas: list[dict]) -> dict:
    """Dataset statistics -- also the evidence that captions are not degenerate."""
    directions: Counter[str] = Counter()
    speeds: Counter[str] = Counter()
    digits: Counter[int] = Counter()
    bounced = 0
    two_digit = 0

    for meta in metas:
        directions.update(meta["directions"])
        speeds.update(meta["speeds"])
        digits.update(meta["digits"])
        if any(meta["bounced"]):
            bounced += 1
        if meta["num_digits"] == 2:
            two_digit += 1

    vocabulary = {word for caption in captions for word in caption.lower().split()}

    return {
        "num_clips": len(captions),
        "unique_captions": len(set(captions)),
        "unique_caption_ratio": len(set(captions)) / max(1, len(captions)),
        "vocabulary_size": len(vocabulary),
        "mean_caption_words": sum(len(c.split()) for c in captions) / max(1, len(captions)),
        "clips_with_bounce": bounced,
        "two_digit_clips": two_digit,
        "direction_counts": dict(sorted(directions.items())),
        "speed_counts": dict(sorted(speeds.items())),
        "digit_counts": {str(k): v for k, v in sorted(digits.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument(
        "--splits", nargs="*", default=None, help="subset of splits to build (default: all)"
    )
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    mnist_root = PROJECT_ROOT / cfg.paths.mnist_root
    output_root = PROJECT_ROOT / cfg.paths.output_root
    mnist_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    split_names = args.splits or list(cfg.splits.to_dict().keys())
    all_stats = {}
    for name in split_names:
        all_stats[name] = build_split(
            name=name,
            split_cfg=cfg.splits[name],
            canvas_size=cfg.canvas_size,
            num_frames=cfg.num_frames,
            two_digit_prob=cfg.two_digit_prob,
            bounce_prob=cfg.bounce_prob,
            mnist_root=mnist_root,
            output_root=output_root,
        )

    (output_root / "dataset_stats.json").write_text(
        json.dumps(all_stats, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Stats written to {output_root / 'dataset_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
