"""Structured grounding: did the generated video actually do what the caption asked?

This is the project's primary alignment metric, and the one CLIPSIM cannot replace.
Because we generate the dataset ourselves, every caption has exact known ground truth --
which digit, which direction, which speed -- so we can check the *content* of a generated
clip rather than its embedding similarity.

Nothing here touches CLIP. The judges are:
    digit      -- an independently trained MNIST CNN (98.7% accurate on held-out data)
    direction  -- intensity-centroid displacement over the first frames
    speed      -- mean per-frame centroid displacement, bucketed

Each has a measurable ceiling on *real* data, reported alongside the score so generated
results are compared against what is actually achievable rather than against 100%.
"""

from __future__ import annotations

import numpy as np
import torch

from text2video.data.captions import SPEED_RANGES_BOUNDS, speed_to_bucket
from text2video.data.dataset import tensor_to_frames
from text2video.data.moving_mnist import (
    measure_speed_from_frames,
    verify_direction_from_frames,
)
from text2video.evaluation.digit_classifier import DigitCNN, classify_frames


def _to_uint8_clips(videos: torch.Tensor) -> np.ndarray:
    return np.stack([tensor_to_frames(clip) for clip in videos])


def digit_grounding(
    videos: torch.Tensor,
    expected_digits: list[int],
    classifier: DigitCNN,
    device: torch.device | str = "cpu",
    frames_to_vote: int = 5,
) -> dict[str, float]:
    """Is the requested digit the one that appears?

    Classifies several frames per clip and takes a majority vote -- a single frame can be
    misread if the digit is clipped at a border, and the vote is closer to what a human
    would judge from watching the clip.

    Also reports `identity_consistency`: the fraction of frames agreeing with the clip's
    majority label. A model whose digit morphs mid-clip scores low here even if the
    majority label is right, which is a failure mode a per-frame accuracy would hide.
    """
    clips = _to_uint8_clips(videos)
    num_clips = len(clips)
    votes = min(frames_to_vote, clips.shape[1])

    correct = 0
    consistencies: list[float] = []
    confidences: list[float] = []

    for i in range(num_clips):
        predictions, confs = classify_frames(classifier, clips[i][:votes], device=device)
        labels, counts = np.unique(predictions, return_counts=True)
        majority = int(labels[counts.argmax()])
        consistencies.append(float(counts.max() / votes))
        confidences.append(float(np.mean(confs)))
        if majority == expected_digits[i]:
            correct += 1

    return {
        "digit_accuracy": correct / max(1, num_clips),
        "identity_consistency": float(np.mean(consistencies)),
        "digit_confidence": float(np.mean(confidences)),
    }


def direction_grounding(
    videos: torch.Tensor, expected_directions: list[str]
) -> dict[str, float]:
    """Does the digit move in the requested direction?

    `undetectable_fraction` counts clips with no measurable motion at all. Those are
    scored as wrong, but tracked separately because "moved the wrong way" and "did not
    move" are different failures with different causes.
    """
    clips = _to_uint8_clips(videos)
    correct = 0
    undetectable = 0

    for i, clip in enumerate(clips):
        observed = verify_direction_from_frames(clip)
        if observed is None:
            undetectable += 1
            continue
        if observed == expected_directions[i]:
            correct += 1

    total = max(1, len(clips))
    return {
        "direction_accuracy": correct / total,
        "direction_undetectable_fraction": undetectable / total,
    }


# Below this per-frame centroid displacement a clip is treated as not moving at all,
# rather than as moving slowly. Without it a completely frozen video scores ~35% on
# speed, because zero displacement falls inside the "slow" bucket -- "did not move" is
# not a correct answer to "move slowly".
STATIC_SPEED_THRESHOLD = 0.3


def speed_grounding(
    videos: torch.Tensor, expected_speeds: list[str]
) -> dict[str, float]:
    """Does the digit move at roughly the requested speed?

    Bucketed with the same thresholds the data generator used, so the comparison is
    apples to apples. Effectively static clips are scored wrong regardless of the
    requested bucket (see `STATIC_SPEED_THRESHOLD`).
    """
    clips = _to_uint8_clips(videos)
    slow_max, medium_max = SPEED_RANGES_BOUNDS

    correct = 0
    static = 0
    measured: list[float] = []
    for i, clip in enumerate(clips):
        speed = measure_speed_from_frames(clip)
        if speed is None or speed < STATIC_SPEED_THRESHOLD:
            static += 1
            continue
        measured.append(speed)
        if speed_to_bucket(speed, slow_max, medium_max) == expected_speeds[i]:
            correct += 1

    total = max(1, len(clips))
    return {
        "speed_accuracy": correct / total,
        "speed_static_fraction": static / total,
        "measured_speed_mean": float(np.mean(measured)) if measured else 0.0,
    }


def compute_grounding(
    videos: torch.Tensor,
    metadata: list[dict],
    classifier: DigitCNN | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """All grounding metrics for a batch of generated clips.

    Restricted to single-digit clips: with two digits the "requested digit" and
    "requested direction" are ambiguous for a whole-frame measurement, and scoring them
    would add noise rather than information.
    """
    single = [i for i, meta in enumerate(metadata) if meta["num_digits"] == 1]
    if not single:
        return {"grounding_clips": 0}

    subset = videos[single]
    metas = [metadata[i] for i in single]
    expected_digits = [m["digits"][0] for m in metas]
    expected_directions = [m["directions"][0] for m in metas]
    expected_speeds = [m["speeds"][0] for m in metas]

    metrics: dict[str, float] = {"grounding_clips": len(single)}
    if classifier is not None:
        metrics.update(digit_grounding(subset, expected_digits, classifier, device))
    metrics.update(direction_grounding(subset, expected_directions))
    metrics.update(speed_grounding(subset, expected_speeds))

    # A single headline number combining the attributes that were measured.
    components = [
        metrics[key]
        for key in ("digit_accuracy", "direction_accuracy", "speed_accuracy")
        if key in metrics
    ]
    if components:
        metrics["grounding_score"] = float(np.mean(components))
    return metrics
