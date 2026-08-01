"""End-to-end generation: a sentence in, a video clip out.

Wraps the three pieces the user never has to think about -- frozen CLIP text encoder,
trained VAE, and the scoring functions -- behind one object. Used by the CLI sampler and
by the demo app, so both share exactly the same code path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from text2video.data.captions import parse_caption
from text2video.data.dataset import tensor_to_frames
from text2video.data.moving_mnist import (
    measure_speed_from_frames,
    verify_direction_from_frames,
)
from text2video.evaluation.digit_classifier import classify_frames, load_digit_classifier
from text2video.evaluation.harness import load_model_from_checkpoint


class VideoGenerator:
    """Generates clips from text and, optionally, scores what it produced.

    The text encoder is loaded lazily: the demo shows its UI immediately and only pays
    the CLIP load cost when the first prompt arrives.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        digit_classifier_path: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model, self.payload = load_model_from_checkpoint(checkpoint_path, self.device)
        self.checkpoint_path = str(checkpoint_path)
        self.variant = self.model.describe()["variant"]

        self._text_encoder: Any | None = None
        self._digit_classifier = None
        if digit_classifier_path and Path(digit_classifier_path).exists():
            self._digit_classifier = load_digit_classifier(
                digit_classifier_path, device=self.device
            )

    @property
    def text_encoder(self):
        if self._text_encoder is None:
            from text2video.text_encoder.clip_encoder import CLIPTextEncoder

            self._text_encoder = CLIPTextEncoder(device=self.device)
        return self._text_encoder

    @torch.no_grad()
    def generate(
        self,
        prompts: str | list[str],
        num_samples: int = 1,
        temperature: float = 1.0,
        use_prior_mean: bool = False,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate clips. Returns (len(prompts) * num_samples, T, 1, 64, 64).

        `num_samples > 1` draws several latents for the same caption -- the visible
        payoff of using a VAE rather than a deterministic decoder, since one sentence
        legitimately describes many videos.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if seed is not None:
            torch.manual_seed(seed)

        embeddings = self.text_encoder.encode(prompts).to(self.device)
        if num_samples > 1:
            embeddings = embeddings.repeat_interleave(num_samples, dim=0)

        return self.model.generate(
            embeddings, temperature=temperature, use_prior_mean=use_prior_mean
        ).cpu()

    def score(self, video: torch.Tensor, prompt: str) -> dict[str, Any]:
        """Measure what a single generated clip actually shows, against what was asked.

        Returns the requested attributes parsed from the prompt alongside the attributes
        measured from the pixels, so the demo can show agreement rather than assert it.
        """
        frames = tensor_to_frames(video)
        requested = parse_caption(prompt)

        observed_direction = verify_direction_from_frames(frames)
        observed_speed = measure_speed_from_frames(frames)

        observed_digit: int | None = None
        digit_confidence: float | None = None
        if self._digit_classifier is not None:
            votes = min(5, len(frames))
            predictions, confidences = classify_frames(
                self._digit_classifier, frames[:votes], device=self.device
            )
            labels, counts = np.unique(predictions, return_counts=True)
            observed_digit = int(labels[counts.argmax()])
            digit_confidence = float(np.mean(confidences))

        result: dict[str, Any] = {
            "requested_digits": requested["digits"],
            "requested_direction": requested["direction"],
            "requested_speed": requested["speed"],
            "observed_digit": observed_digit,
            "digit_confidence": digit_confidence,
            "observed_direction": observed_direction,
            "observed_speed_px_per_frame": (
                round(observed_speed, 3) if observed_speed is not None else None
            ),
            "is_static": observed_speed is None or observed_speed < 0.3,
        }
        result["direction_matches"] = (
            requested["direction"] is not None
            and observed_direction == requested["direction"]
        )
        result["digit_matches"] = (
            observed_digit is not None
            and bool(requested["digits"])
            and observed_digit == requested["digits"][0]
        )
        return result

    def describe(self) -> dict[str, Any]:
        info = dict(self.model.describe())
        info["checkpoint"] = self.checkpoint_path
        info["train_step"] = self.payload.get("global_step")
        return info
