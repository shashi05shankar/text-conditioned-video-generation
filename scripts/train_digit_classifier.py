"""Train the independent MNIST digit classifier used by the grounding metric.

    python scripts/train_digit_classifier.py

Trains in ~2 minutes on CPU. Saved to outputs/digit_classifier.pt.

This classifier is the "judge" for whether a generated clip shows the digit the caption
asked for. It shares no weights with the generative models and never sees generated
data during training, which is what keeps the metric independent.

It is trained with the same random-placement augmentation the crop function will face at
evaluation time, so its reported accuracy reflects the conditions it is actually used in.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text2video.core.seed import seed_everything  # noqa: E402
from text2video.data.moving_mnist import load_mnist_sprites  # noqa: E402
from text2video.evaluation.digit_classifier import DigitCNN, crop_digit  # noqa: E402


def make_canvas_batch(
    sprites: np.ndarray, indices: np.ndarray, rng: np.random.Generator, canvas: int = 64
) -> np.ndarray:
    """Paste each sprite at a random position on a 64x64 canvas, then crop it back out.

    This mirrors exactly what happens at evaluation time: the digit sits somewhere in a
    64x64 frame and `crop_digit` has to recover it from the intensity centroid. Training
    through the same path means the reported accuracy is the accuracy we actually get.
    """
    size = sprites.shape[1]
    max_pos = canvas - size
    crops = np.empty((len(indices), size, size), dtype=np.uint8)
    for i, index in enumerate(indices):
        frame = np.zeros((canvas, canvas), dtype=np.uint8)
        x = int(rng.integers(0, max_pos + 1))
        y = int(rng.integers(0, max_pos + 1))
        frame[y : y + size, x : x + size] = sprites[index]
        crops[i] = crop_digit(frame, size=size)
    return crops


def evaluate(
    model: DigitCNN, sprites, labels, rng, batch_size: int = 512,
    device: torch.device | None = None,
) -> float:
    device = device or torch.device("cpu")
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            indices = np.arange(start, min(start + batch_size, len(labels)))
            crops = make_canvas_batch(sprites, indices, rng)
            x = torch.from_numpy(crops.astype(np.float32) / 255.0).unsqueeze(1).to(device)
            predictions = model(x).argmax(dim=1).cpu().numpy()
            correct += int((predictions == labels[indices]).sum())
    return correct / len(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist-root", default="data/raw")
    parser.add_argument("--out", default="outputs/digit_classifier.pt")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = parser.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    print(f"device: {device}")

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    train_sprites, train_labels = load_mnist_sprites(PROJECT_ROOT / args.mnist_root, train=True)
    test_sprites, test_labels = load_mnist_sprites(PROJECT_ROOT / args.mnist_root, train=False)
    print(f"MNIST: {len(train_labels)} train, {len(test_labels)} test")

    model = DigitCNN().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"DigitCNN params: {num_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        indices = rng.integers(0, len(train_labels), size=args.batch_size)
        crops = make_canvas_batch(train_sprites, indices, rng)
        x = torch.from_numpy(crops.astype(np.float32) / 255.0).unsqueeze(1).to(device)
        y = torch.from_numpy(train_labels[indices]).long().to(device)

        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()

        if step % 250 == 0:
            print(f"step {step:>5}/{args.steps} loss={loss.item():.4f}")

    elapsed = time.time() - started
    accuracy = evaluate(
        model, test_sprites, test_labels, np.random.default_rng(123), device=device
    )
    print(f"\ntrained in {elapsed:.1f}s")
    print(f"held-out MNIST test accuracy (random 64x64 placement + crop): {accuracy:.2%}")

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # CPU tensors so the checkpoint loads anywhere, including a laptop with no GPU.
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "test_accuracy": accuracy,
            "steps": args.steps,
            "params": num_params,
        },
        out_path,
    )
    (out_path.with_suffix(".json")).write_text(
        json.dumps(
            {"test_accuracy": accuracy, "steps": args.steps, "params": num_params,
             "train_seconds": round(elapsed, 1)},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved -> {out_path}")

    if accuracy < 0.95:
        print("WARNING: accuracy below 95% -- grounding scores will be noisy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
