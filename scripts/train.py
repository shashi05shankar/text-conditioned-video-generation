"""Train a text-conditioned video VAE.

    python scripts/train.py --config configs/train_convlstm.yaml
    python scripts/train.py --config configs/train_baseline.yaml --resume auto
    python scripts/train.py --config configs/train_convlstm.yaml --smoke   # 30-step check

Free GPU sessions end without warning. --mirror-dir copies every checkpoint to a
second location so a dropped session costs minutes rather than the whole run:

    Kaggle:  --mirror-dir /kaggle/working/checkpoints
    Colab:   --mirror-dir /content/drive/MyDrive/text2video_ckpt

Re-running the same command with --resume auto continues from the latest checkpoint,
restoring optimizer, scheduler and RNG state -- not just the weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text2video.core.config import load_config, merge_overrides  # noqa: E402
from text2video.core.seed import seed_everything, worker_init_fn  # noqa: E402
from text2video.data.dataset import collate_fn, load_split  # noqa: E402
from text2video.models.cvae import build_model  # noqa: E402
from text2video.training.checkpoint import find_latest_checkpoint  # noqa: E402
from text2video.training.trainer import Trainer  # noqa: E402


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None,
                        help="'auto' to continue the latest checkpoint, or a path")
    parser.add_argument("--mirror-dir", default=None,
                        help="second location for checkpoints, e.g. a mounted Drive folder")
    parser.add_argument("--max-steps", type=int, default=None, help="override config")
    parser.add_argument("--batch-size", type=int, default=None, help="override config")
    parser.add_argument("--smoke", action="store_true",
                        help="30 steps on a tiny subset: proves the pipeline runs end to end")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    overrides: dict = {}
    if args.max_steps is not None:
        overrides["train.max_steps"] = args.max_steps
    if args.batch_size is not None:
        overrides["train.batch_size"] = args.batch_size
    if args.smoke:
        overrides.update({
            "train.max_steps": 30,
            "train.batch_size": 4,
            "train.log_every": 10,
            "train.eval_every": 20,
            "train.max_eval_batches": 2,
            "train.save_every_steps": 20,
            "data.num_workers": 0,
            "run_name": f"{cfg.run_name}_smoke",
            "output.run_dir": f"outputs/runs/{cfg.run_name}_smoke",
        })
    if overrides:
        cfg = merge_overrides(cfg, overrides)

    config_dict = cfg.to_dict()
    device = resolve_device(args.device)
    seed_everything(int(cfg.seed))

    data_root = PROJECT_ROOT / cfg.data.root
    max_items = 64 if args.smoke else None
    train_set = load_split(data_root, cfg.data.train_split, max_items=max_items)
    val_set = load_split(data_root, cfg.data.val_split, max_items=max_items)

    if train_set.text_embeddings is None:
        raise SystemExit(
            "No cached text embeddings found. Run:\n"
            "  python scripts/build_text_embeddings.py"
        )

    print(f"train: {len(train_set)} clips | val: {len(val_set)} clips | "
          f"{train_set.num_frames} frames @ {train_set.frame_size}x{train_set.frame_size}")

    loader_kwargs = dict(
        batch_size=int(cfg.train.batch_size),
        collate_fn=collate_fn,
        num_workers=int(cfg.data.num_workers),
        worker_init_fn=worker_init_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=int(cfg.data.num_workers) > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)

    model_kwargs = {k: v for k, v in config_dict["model"].items() if k != "name"}
    model = build_model(cfg.model.name, **model_kwargs)
    print(f"model: {model.describe()}")

    run_dir = PROJECT_ROOT / cfg.output.run_dir
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config_dict,
        device=device,
        run_dir=run_dir,
        mirror_dir=args.mirror_dir,
    )

    if args.resume:
        checkpoint = (
            find_latest_checkpoint(run_dir / "checkpoints", pattern="last.pt")
            or find_latest_checkpoint(run_dir / "checkpoints")
            if args.resume == "auto"
            else Path(args.resume)
        )
        if checkpoint is None:
            print("--resume auto: no checkpoint found, starting from scratch")
        else:
            trainer.resume(checkpoint)

    trainer.train(max_steps=int(cfg.train.max_steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
