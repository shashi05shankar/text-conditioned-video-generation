"""Temporal-consistency metrics.

The central trap: consecutive-frame similarity alone is not a quality measure. A model
that emits the same frame 16 times scores a *perfect* SSIM of 1.0 while producing no
video at all. Any similarity number must therefore be read alongside evidence that
motion actually happened.

So we always report a pair:

    frame_ssim          -- how similar consecutive frames are (smoothness)
    motion_magnitude    -- mean optical-flow displacement (is anything moving?)

plus a combined score that only rewards a model doing both. Real data provides the
reference point: a good model should land near the ground-truth values, not maximise
either metric alone.
"""

from __future__ import annotations

import numpy as np
import torch

from text2video.data.dataset import tensor_to_frames


def _to_uint8_clips(videos: torch.Tensor) -> np.ndarray:
    """(N, T, 1, H, W) float [-1,1] -> (N, T, H, W) uint8."""
    return np.stack([tensor_to_frames(clip) for clip in videos])


def frame_ssim(videos: torch.Tensor) -> dict[str, float]:
    """Mean SSIM between consecutive frames.

    High = smooth, low = flickering. But a frozen video scores 1.0, so this must never
    be read on its own -- see `temporal_metrics`.
    """
    from skimage.metrics import structural_similarity

    clips = _to_uint8_clips(videos)
    scores: list[float] = []
    for clip in clips:
        for t in range(len(clip) - 1):
            scores.append(
                structural_similarity(clip[t], clip[t + 1], data_range=255)
            )
    return {
        "frame_ssim": float(np.mean(scores)) if scores else 0.0,
        "frame_ssim_std": float(np.std(scores)) if scores else 0.0,
    }


def motion_magnitude(videos: torch.Tensor) -> dict[str, float]:
    """Mean per-frame optical-flow displacement, in pixels.

    Farneback dense flow -- cheap, no learned model to download, and adequate for the
    single high-contrast object our clips contain. This is the term that stops a static
    video from being scored as high-quality.
    """
    import cv2

    clips = _to_uint8_clips(videos)
    per_clip: list[float] = []
    for clip in clips:
        magnitudes: list[float] = []
        for t in range(len(clip) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                clip[t], clip[t + 1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0,
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            # Average over moving pixels only. Averaging over the whole frame would be
            # dominated by the large static black background and understate real motion.
            moving = magnitude[magnitude > 0.1]
            magnitudes.append(float(moving.mean()) if moving.size else 0.0)
        per_clip.append(float(np.mean(magnitudes)) if magnitudes else 0.0)

    return {
        "motion_magnitude": float(np.mean(per_clip)),
        "motion_magnitude_std": float(np.std(per_clip)),
        "static_clip_fraction": float(np.mean([m < 0.05 for m in per_clip])),
    }


def centroid_motion(videos: torch.Tensor) -> dict[str, float]:
    """Mean per-frame movement of the intensity centroid, in pixels.

    A second, independent motion estimate that does not depend on optical flow. For our
    single-object clips it is the more direct measurement, and it cross-checks the flow
    number.
    """
    clips = _to_uint8_clips(videos)
    speeds: list[float] = []
    for clip in clips:
        centroids = []
        for frame in clip:
            total = float(frame.sum())
            if total <= 0:
                continue
            ys, xs = np.nonzero(frame)
            weights = frame[ys, xs].astype(np.float64)
            centroids.append(
                ((xs * weights).sum() / total, (ys * weights).sum() / total)
            )
        if len(centroids) < 2:
            speeds.append(0.0)
            continue
        steps = [
            float(np.hypot(centroids[i + 1][0] - centroids[i][0],
                           centroids[i + 1][1] - centroids[i][1]))
            for i in range(len(centroids) - 1)
        ]
        speeds.append(float(np.mean(steps)))

    return {
        "centroid_speed": float(np.mean(speeds)),
        "centroid_speed_std": float(np.std(speeds)),
    }


def temporal_metrics(
    videos: torch.Tensor, reference_videos: torch.Tensor | None = None
) -> dict[str, float]:
    """All temporal metrics, plus a combined score anchored on real data.

    `temporal_score` is |1 - relative motion error| x frame_ssim: it rewards a model
    whose motion is close to the reference *and* whose frames are smooth. A static model
    gets a motion error of 1.0 and therefore a score of 0, no matter how perfect its
    SSIM -- which is the whole point.

    Without a reference the score is omitted rather than guessed at.
    """
    metrics: dict[str, float] = {}
    metrics.update(frame_ssim(videos))
    metrics.update(motion_magnitude(videos))
    metrics.update(centroid_motion(videos))

    if reference_videos is not None:
        reference = {}
        reference.update(frame_ssim(reference_videos))
        reference.update(centroid_motion(reference_videos))
        metrics["reference_frame_ssim"] = reference["frame_ssim"]
        metrics["reference_centroid_speed"] = reference["centroid_speed"]

        ref_speed = reference["centroid_speed"]
        if ref_speed > 1e-6:
            relative_error = abs(metrics["centroid_speed"] - ref_speed) / ref_speed
            metrics["motion_relative_error"] = float(relative_error)
            metrics["temporal_score"] = float(
                max(0.0, 1.0 - relative_error) * metrics["frame_ssim"]
            )

    return metrics
