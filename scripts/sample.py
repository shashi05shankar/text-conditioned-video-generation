"""Generate videos from text prompts using a trained checkpoint.

    # a few prompts, one clip each
    python scripts/sample.py --checkpoint outputs/runs/convlstm/checkpoints/best.pt

    # your own prompt, 4 samples showing latent diversity
    python scripts/sample.py --checkpoint ... --prompt "The digit 5 moves to the right." -n 4

    # side-by-side baseline vs ConvLSTM on identical prompts
    python scripts/sample.py --compare

Writes GIFs plus a labelled filmstrip figure to outputs/samples/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from text2video.inference.generator import VideoGenerator  # noqa: E402
from text2video.utils.video import save_comparison_figure, save_gif  # noqa: E402

# Fixed prompt set used for every qualitative figure, so results are comparable across
# models and across training runs. Covers each attribute the model must learn.
DEFAULT_PROMPTS = [
    "The digit 3 moves to the right.",
    "The digit 7 moves to the left.",
    "A handwritten 1 travels upward.",
    "The number 5 moves downward.",
    "The digit 8 glides diagonally up and to the right.",
    "A 2 moves quickly towards the bottom left corner.",
    "The digit 4 moves slowly to the right.",
    "The digit 6 moves rightward and bounces off the right edge.",
]


def summarise(score: dict) -> str:
    """One-line human-readable verdict for a generated clip."""
    bits = []
    if score["observed_digit"] is not None:
        mark = "OK" if score["digit_matches"] else "MISS"
        bits.append(f"digit {score['observed_digit']} [{mark}]")
    if score["observed_direction"] is not None:
        mark = "OK" if score["direction_matches"] else "MISS"
        bits.append(f"dir {score['observed_direction']} [{mark}]")
    else:
        bits.append("dir n/a")
    if score["observed_speed_px_per_frame"] is not None:
        bits.append(f"{score['observed_speed_px_per_frame']:.2f}px/f")
    if score["is_static"]:
        bits.append("STATIC")
    return "  ".join(bits)


def run_one(generator: VideoGenerator, prompts: list[str], args, tag: str) -> list[dict]:
    videos = generator.generate(
        prompts,
        num_samples=args.num_samples,
        temperature=args.temperature,
        use_prior_mean=args.prior_mean,
        seed=args.seed,
    )
    out_dir = PROJECT_ROOT / args.out / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    clips, labels, records = [], [], []
    for i, prompt in enumerate(prompts):
        for s in range(args.num_samples):
            index = i * args.num_samples + s
            video = videos[index]
            score = generator.score(video, prompt)
            suffix = f"_s{s}" if args.num_samples > 1 else ""
            save_gif(video, out_dir / f"{i:02d}{suffix}.gif")
            clips.append(video)
            labels.append(f"{prompt}\n   -> {summarise(score)}")
            records.append({"prompt": prompt, "sample": s, **score})
            print(f"  [{tag}] {prompt}\n        {summarise(score)}")

    save_comparison_figure(
        clips, labels, out_dir / "samples.png", every=2,
        title=f"{tag} -- generated from text (every 2nd frame)",
    )
    (out_dir / "scores.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"  wrote {out_dir}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--compare", action="store_true",
                        help="run baseline and convlstm on the same prompts")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--prompt", default=None, help="a single prompt instead of the default set")
    parser.add_argument("-n", "--num-samples", type=int, default=1,
                        help="clips per prompt; >1 shows latent diversity")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prior-mean", action="store_true",
                        help="deterministic: use the prior mean instead of sampling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="outputs/samples")
    parser.add_argument("--digit-classifier", default="outputs/digit_classifier.pt")
    args = parser.parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    classifier = PROJECT_ROOT / args.digit_classifier

    targets: list[tuple[str, Path]] = []
    if args.compare:
        for variant in ("baseline", "convlstm"):
            for name in ("best.pt", "last.pt"):
                candidate = PROJECT_ROOT / args.runs_dir / variant / "checkpoints" / name
                if candidate.exists():
                    targets.append((variant, candidate))
                    break
        if not targets:
            parser.error(f"no trained checkpoints found under {args.runs_dir}")
    elif args.checkpoint:
        targets.append(("model", Path(args.checkpoint)))
    else:
        parser.error("pass --checkpoint or --compare")

    for tag, checkpoint in targets:
        print(f"\n=== {tag}: {checkpoint} ===")
        generator = VideoGenerator(checkpoint, device=device, digit_classifier_path=classifier)
        print(f"  {generator.describe()}")
        run_one(generator, prompts, args, tag)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
