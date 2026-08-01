"""Evaluate trained checkpoints and write metrics records.

    # one model
    python scripts/evaluate.py --checkpoint outputs/runs/convlstm/checkpoints/best.pt

    # every trained model plus the ceiling and control reference rows
    python scripts/evaluate.py --all

Records go to outputs/eval/<variant>.json and are consumed by scripts/build_report.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

torch.set_num_threads(os.cpu_count() or 4)

from text2video.evaluation.harness import (  # noqa: E402
    evaluate_model,
    evaluate_real_data_reference,
    evaluate_static_control,
    write_metrics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--all", action="store_true",
                        help="evaluate every run in outputs/runs plus reference rows")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-clips", type=int, default=500)
    parser.add_argument("--out-dir", default="outputs/eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--digit-classifier", default="outputs/digit_classifier.pt")
    parser.add_argument("--fid-extractor", default="clip", choices=["clip", "inception"])
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-clipsim", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    data_root = PROJECT_ROOT / args.data_root
    out_dir = PROJECT_ROOT / args.out_dir
    classifier_path = PROJECT_ROOT / args.digit_classifier

    checkpoints: list[Path] = []
    if args.checkpoint:
        checkpoints.append(Path(args.checkpoint))
    if args.all:
        runs_dir = PROJECT_ROOT / args.runs_dir
        for run in sorted(runs_dir.glob("*")):
            if run.name.endswith("_smoke"):
                continue  # smoke runs are pipeline checks, not results
            for name in ("best.pt", "last.pt"):
                candidate = run / "checkpoints" / name
                if candidate.exists():
                    checkpoints.append(candidate)
                    break

    if not checkpoints and not args.all:
        parser.error("pass --checkpoint or --all")

    common = dict(
        data_root=data_root,
        split=args.split,
        num_clips=args.num_clips,
        device=device,
        digit_classifier_path=classifier_path,
    )

    for checkpoint in checkpoints:
        print(f"\n=== {checkpoint} ===")
        metrics = evaluate_model(
            checkpoint,
            fid_extractor=args.fid_extractor,
            seed=args.seed,
            skip_fid=args.skip_fid,
            skip_clipsim=args.skip_clipsim,
            **common,
        )
        path = write_metrics(metrics, out_dir / f"{metrics['variant']}.json")
        print(f"wrote {path}")
        for key in ("grounding_score", "direction_accuracy", "digit_accuracy",
                    "temporal_score", "frame_ssim", "centroid_speed", "fid", "clipsim"):
            if key in metrics:
                print(f"  {key:24s} {metrics[key]:.4f}")

    if args.all:
        print("\n=== reference rows ===")
        ceiling = evaluate_real_data_reference(**common)
        write_metrics(ceiling, out_dir / "real_data_ceiling.json")
        print(f"real-data ceiling grounding_score = {ceiling['grounding_score']:.4f}")

        control = evaluate_static_control(**common)
        write_metrics(control, out_dir / "static_control.json")
        print(f"static control frame_ssim = {control['frame_ssim']:.4f} "
              f"(perfect) but temporal_score = {control['temporal_score']:.4f}")

    print(f"\nrecords in {out_dir}. Build the table with:\n"
          f"  python scripts/build_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
