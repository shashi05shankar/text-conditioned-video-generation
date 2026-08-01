"""CLIPSIM: text-video alignment via CLIP.

Encode each generated frame with CLIP's image tower, take the cosine similarity against
the caption's CLIP text embedding, and average over frames and clips. Higher is better.

**Read this number with care.** Two limitations, both measured rather than assumed:

1. *Not independent.* The same frozen CLIP model conditions our generators. A model that
   learned to satisfy CLIP would score well here by construction.

2. *Probably insensitive to the attributes we care about.* The linear-probe experiment in
   `scripts/build_text_embeddings.py` found that CLIP text embeddings for "moves to the
   right" and "moves to the left" have cosine similarity 0.98, even though a linear
   classifier recovers direction from them at ~93%. The directional information exists
   but sits in a low-variance subspace that whole-vector cosine similarity does not
   surface -- and CLIPSIM *is* whole-vector cosine similarity.

So CLIPSIM is reported for comparability with the literature, while structured grounding
is our primary alignment measure. `clipsim_discrimination` below quantifies limitation 2
directly instead of leaving it as a caveat.
"""

from __future__ import annotations

import numpy as np
import torch

from text2video.evaluation.fid import CLIPImageFeatures


@torch.no_grad()
def compute_clipsim(
    videos: torch.Tensor,
    text_embeddings: torch.Tensor,
    image_encoder: CLIPImageFeatures,
    batch_size: int = 64,
) -> dict[str, float]:
    """Mean cosine similarity between each caption and its generated frames.

    Args:
        videos: (N, T, 1, H, W) in [-1, 1]
        text_embeddings: (N, 512) L2-normalised CLIP text embeddings
        image_encoder: frozen CLIP image tower
    """
    num_clips, num_frames = videos.shape[0], videos.shape[1]
    flat = videos.reshape(num_clips * num_frames, *videos.shape[2:])

    features = []
    for start in range(0, flat.shape[0], batch_size):
        features.append(image_encoder(flat[start : start + batch_size]))
    image_features = torch.cat(features, dim=0)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    text = text_embeddings.float()
    text = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # Repeat each caption across its own frames so the dot product lines up.
    text_per_frame = text.repeat_interleave(num_frames, dim=0)

    similarities = (image_features * text_per_frame).sum(dim=-1)
    per_clip = similarities.reshape(num_clips, num_frames).mean(dim=1)

    return {
        "clipsim": float(per_clip.mean()),
        "clipsim_std": float(per_clip.std()),
        "clipsim_min": float(per_clip.min()),
        "clipsim_max": float(per_clip.max()),
    }


@torch.no_grad()
def clipsim_discrimination(
    videos: torch.Tensor,
    matched_text: torch.Tensor,
    mismatched_text: torch.Tensor,
    image_encoder: CLIPImageFeatures,
    batch_size: int = 64,
) -> dict[str, float]:
    """Does CLIPSIM actually prefer the *correct* caption?

    Scores each video against its own caption and against a deliberately wrong one
    (here: a caption describing the opposite direction). If the metric is working, the
    matched score should be clearly higher and `clipsim_accuracy` well above 50%.

    A near-50% result does not mean the generator failed -- it means CLIPSIM cannot tell
    these captions apart, which is exactly the limitation documented above. Measuring it
    turns a caveat into evidence.
    """
    matched = compute_clipsim(videos, matched_text, image_encoder, batch_size)
    mismatched = compute_clipsim(videos, mismatched_text, image_encoder, batch_size)

    num_clips, num_frames = videos.shape[0], videos.shape[1]
    flat = videos.reshape(num_clips * num_frames, *videos.shape[2:])
    features = []
    for start in range(0, flat.shape[0], batch_size):
        features.append(image_encoder(flat[start : start + batch_size]))
    image_features = torch.cat(features, dim=0)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def per_clip_scores(text: torch.Tensor) -> torch.Tensor:
        normed = text.float()
        normed = normed / normed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        repeated = normed.repeat_interleave(num_frames, dim=0)
        return (image_features * repeated).sum(dim=-1).reshape(num_clips, num_frames).mean(dim=1)

    correct = per_clip_scores(matched_text) > per_clip_scores(mismatched_text)

    return {
        "clipsim_matched": matched["clipsim"],
        "clipsim_mismatched": mismatched["clipsim"],
        "clipsim_margin": matched["clipsim"] - mismatched["clipsim"],
        "clipsim_accuracy": float(correct.float().mean()),
    }


def opposite_direction_caption(caption: str) -> str:
    """Flip a caption's direction words, to build the mismatched control set."""
    swaps = [
        ("to the right", "to the left"),
        ("rightward", "leftward"),
        ("towards the right side", "towards the left side"),
        ("upward", "downward"),
        ("towards the top", "towards the bottom"),
        ("up the frame", "down the frame"),
        ("up and to the right", "down and to the left"),
        ("up and to the left", "down and to the right"),
        ("towards the top right corner", "towards the bottom left corner"),
        ("towards the top left corner", "towards the bottom right corner"),
    ]
    text = caption
    for a, b in swaps:
        if a in text:
            return text.replace(a, b)
        if b in text:
            return text.replace(b, a)
    return text


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denominator)
