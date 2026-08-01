"""Evaluation harness: checkpoint in, complete metrics record out.

Runs every model through an identical procedure -- same held-out split, same fixed
sampling seed, same metrics -- and writes a JSON record. The comparison table is built
by reading those records back, so no number in the final report is ever typed by hand.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from text2video.data.dataset import load_split
from text2video.evaluation.clipsim import (
    clipsim_discrimination,
    compute_clipsim,
    opposite_direction_caption,
)
from text2video.evaluation.digit_classifier import DigitCNN, load_digit_classifier
from text2video.evaluation.fid import build_feature_extractor, compute_fid
from text2video.evaluation.grounding import compute_grounding
from text2video.evaluation.temporal import temporal_metrics
from text2video.models.cvae import TextConditionedVAE, build_model
from text2video.training.checkpoint import load_checkpoint


def load_model_from_checkpoint(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> tuple[TextConditionedVAE, dict[str, Any]]:
    """Rebuild a model from its checkpoint, using the config stored inside it.

    The config travels with the checkpoint so evaluation can never silently instantiate
    a differently-shaped model than the one that was trained.
    """
    payload = load_checkpoint(checkpoint_path, map_location=device, restore_rng=False)
    config = payload.get("config", {})
    model_cfg = dict(config.get("model", {}))
    name = model_cfg.pop("name", None)
    if name is None:
        raise ValueError(f"checkpoint {checkpoint_path} has no model.name in its config")

    model = build_model(name, **model_cfg)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload


@torch.no_grad()
def generate_videos(
    model: TextConditionedVAE,
    text_embeddings: torch.Tensor,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
    seed: int = 1234,
    use_prior_mean: bool = False,
) -> tuple[torch.Tensor, float]:
    """Generate one clip per caption. Returns (videos on CPU, seconds per clip).

    The seed is fixed so a re-run reproduces the same samples, and timing is measured
    here because inference cost is one of the reported comparison axes.
    """
    torch.manual_seed(seed)
    outputs: list[torch.Tensor] = []
    started = time.time()

    for start in range(0, len(text_embeddings), batch_size):
        chunk = text_embeddings[start : start + batch_size].to(device)
        outputs.append(model.generate(chunk, use_prior_mean=use_prior_mean).cpu())

    elapsed = time.time() - started
    videos = torch.cat(outputs, dim=0)
    return videos, elapsed / max(1, len(videos))


def evaluate_model(
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str = "test",
    num_clips: int = 500,
    device: torch.device | str = "cpu",
    digit_classifier_path: str | Path | None = None,
    fid_extractor: str = "clip",
    seed: int = 1234,
    skip_fid: bool = False,
    skip_clipsim: bool = False,
) -> dict[str, Any]:
    """Full evaluation of one checkpoint on one split."""
    device = torch.device(device)
    model, payload = load_model_from_checkpoint(checkpoint_path, device)
    description = model.describe()

    dataset = load_split(data_root, split)
    if dataset.text_embeddings is None:
        raise ValueError(f"split {split} has no cached text embeddings")

    count = min(num_clips, len(dataset))
    real_videos = torch.stack([dataset[i]["frames"] for i in range(count)])
    metadata = [dataset[i]["meta"] for i in range(count)]
    captions = [dataset[i]["caption"] for i in range(count)]
    text_embeddings = torch.from_numpy(
        np.asarray(dataset.text_embeddings[:count], dtype=np.float32)
    )

    print(f"generating {count} clips from {Path(checkpoint_path).name}...")
    generated, seconds_per_clip = generate_videos(
        model, text_embeddings, device=device, seed=seed
    )

    metrics: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "variant": description["variant"],
        "split": split,
        "num_clips": count,
        "seed": seed,
        # -- computational cost ------------------------------------------------
        "total_params": description["total_params"],
        "inference_seconds_per_clip": round(seconds_per_clip, 5),
        "train_step": payload.get("global_step"),
    }

    run_record_path = Path(checkpoint_path).parent.parent / "run_record.json"
    if run_record_path.exists():
        record = json.loads(run_record_path.read_text(encoding="utf-8"))
        metrics["train_minutes"] = record.get("train_minutes")
        metrics["train_steps"] = record.get("global_step")
        metrics["peak_gpu_memory_mb"] = record.get("peak_gpu_memory_mb")

    # -- temporal consistency (real data is the reference point) ---------------
    print("  temporal metrics...")
    metrics.update(temporal_metrics(generated, reference_videos=real_videos))

    # -- structured grounding (the independent alignment measure) --------------
    print("  structured grounding...")
    classifier: DigitCNN | None = None
    if digit_classifier_path and Path(digit_classifier_path).exists():
        classifier = load_digit_classifier(digit_classifier_path, device=device)
    metrics.update(compute_grounding(generated, metadata, classifier, device=device))

    # -- visual quality --------------------------------------------------------
    if not skip_fid:
        print(f"  FID ({fid_extractor} features)...")
        extractor = build_feature_extractor(fid_extractor, device=device)
        metrics.update(
            compute_fid(real_videos, generated, extractor, max_frames_per_clip=8)
        )
        metrics["fid_extractor"] = fid_extractor

    # -- CLIP text-video alignment --------------------------------------------
    if not skip_clipsim:
        print("  CLIPSIM...")
        from text2video.evaluation.fid import CLIPImageFeatures
        from text2video.text_encoder.clip_encoder import CLIPTextEncoder

        image_encoder = CLIPImageFeatures(device=device)
        metrics.update(compute_clipsim(generated, text_embeddings, image_encoder))

        # Quantify how much CLIPSIM can be trusted, rather than only warning about it.
        text_encoder = CLIPTextEncoder(device=device)
        flipped = [opposite_direction_caption(c) for c in captions]
        mismatched = text_encoder.encode(flipped)
        metrics.update(
            clipsim_discrimination(generated, text_embeddings, mismatched, image_encoder)
        )

    return metrics


def evaluate_real_data_reference(
    data_root: str | Path,
    split: str = "test",
    num_clips: int = 500,
    device: torch.device | str = "cpu",
    digit_classifier_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score the *real* clips with the same metrics.

    This is the ceiling row of the comparison table. Metrics have measurement error --
    direction recovery tops out near 95% even on ground-truth video -- so a generated
    score has to be read against what is actually achievable, not against 100%.
    """
    dataset = load_split(data_root, split)
    count = min(num_clips, len(dataset))
    real = torch.stack([dataset[i]["frames"] for i in range(count)])
    metadata = [dataset[i]["meta"] for i in range(count)]

    classifier = (
        load_digit_classifier(digit_classifier_path, device=device)
        if digit_classifier_path and Path(digit_classifier_path).exists()
        else None
    )

    metrics: dict[str, Any] = {
        "variant": "real_data_ceiling",
        "split": split,
        "num_clips": count,
    }
    metrics.update(temporal_metrics(real, reference_videos=real))
    metrics.update(compute_grounding(real, metadata, classifier, device=device))
    return metrics


def evaluate_static_control(
    data_root: str | Path,
    split: str = "test",
    num_clips: int = 500,
    device: torch.device | str = "cpu",
    digit_classifier_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score a deliberately degenerate 'model' that freezes frame 0 for the whole clip.

    It generates no motion whatsoever, yet it scores a near-perfect frame-SSIM. That is
    the concrete demonstration of why temporal consistency is never reported as
    similarity alone -- this control would top a naive leaderboard.
    """
    dataset = load_split(data_root, split)
    count = min(num_clips, len(dataset))
    real = torch.stack([dataset[i]["frames"] for i in range(count)])
    metadata = [dataset[i]["meta"] for i in range(count)]

    frozen = real[:, :1].repeat(1, real.shape[1], 1, 1, 1)

    classifier = (
        load_digit_classifier(digit_classifier_path, device=device)
        if digit_classifier_path and Path(digit_classifier_path).exists()
        else None
    )

    metrics: dict[str, Any] = {
        "variant": "static_control",
        "split": split,
        "num_clips": count,
        "note": "frame 0 repeated 16x -- no motion at all",
    }
    metrics.update(temporal_metrics(frozen, reference_videos=real))
    metrics.update(compute_grounding(frozen, metadata, classifier, device=device))
    return metrics


def write_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path
