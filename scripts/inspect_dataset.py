"""Eyeball the generated dataset: save sample filmstrips, GIFs and a stats summary.

    python scripts/inspect_dataset.py --split val --num 8

Writes to outputs/dataset_preview/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text2video.data.dataset import MovingMNISTTextDataset  # noqa: E402
from text2video.data.moving_mnist import (  # noqa: E402
    measure_speed_from_frames,
    verify_direction_from_frames,
)
from text2video.utils.video import save_comparison_figure, save_gif  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--out", default="outputs/dataset_preview")
    args = parser.parse_args()

    split_dir = PROJECT_ROOT / args.data_root / args.split
    dataset = MovingMNISTTextDataset(split_dir)
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{args.split}: {len(dataset)} clips, {dataset.num_frames} frames, "
          f"{dataset.frame_size}x{dataset.frame_size}")

    clips, captions = [], []
    for i in range(min(args.num, len(dataset))):
        item = dataset[i]
        raw = dataset.frames[i]
        observed_dir = verify_direction_from_frames(raw)
        observed_speed = measure_speed_from_frames(raw)
        meta = item["meta"]

        label = (
            f"{item['caption']}\n"
            f"   truth: digits={meta['digits']} dir={meta['directions']} "
            f"speed={meta['speeds']} bounced={meta['bounced']}\n"
            f"   measured from pixels: dir={observed_dir} "
            f"speed={observed_speed:.2f}px/frame" if observed_speed else ""
        )
        clips.append(item["frames"])
        captions.append(label)
        save_gif(item["frames"], out_dir / f"{args.split}_{i:02d}.gif")

    fig_path = save_comparison_figure(
        clips, captions, out_dir / f"{args.split}_samples.png",
        every=2, title=f"{args.split} split -- every 2nd frame, left to right",
    )
    print(f"wrote {fig_path}")
    print(f"wrote {len(clips)} GIFs to {out_dir}")

    stats_path = split_dir / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print("\nsplit stats:")
        for key in ("num_clips", "unique_captions", "unique_caption_ratio",
                    "vocabulary_size", "mean_caption_words", "clips_with_bounce",
                    "two_digit_clips"):
            print(f"  {key}: {stats[key]}")
        print(f"  direction_counts: {stats['direction_counts']}")
        print(f"  speed_counts: {stats['speed_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
