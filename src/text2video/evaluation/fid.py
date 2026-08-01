"""Frechet Distance between real and generated frame distributions (visual quality).

FID compares two sets of images by embedding them with a fixed feature extractor and
measuring the Frechet distance between the resulting Gaussians:

    FID = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2))

Lower is better. It rewards both fidelity (features look like real ones) and diversity
(feature covariance is not collapsed), which is why it catches mode collapse that a
per-image metric would miss.

**Which feature extractor.** Conventional FID uses InceptionV3 trained on ImageNet at
299x299 RGB. Our frames are 64x64 grayscale digits -- far outside that domain, so
Inception features are not obviously meaningful here. We therefore report **CLIP-FID**
(the same frozen CLIP image tower used for CLIPSIM) as the primary number, and support
Inception-FID as an option when the standard metric is wanted for comparability.

Either way, absolute values are not comparable to published numbers on natural-image
benchmarks. They are only meaningful for **relative comparison between our own models**,
which is exactly what this project needs.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F

FeatureExtractor = Callable[[torch.Tensor], torch.Tensor]


def frechet_distance(
    features_a: np.ndarray, features_b: np.ndarray, eps: float = 1e-6
) -> float:
    """Frechet distance between two sets of feature vectors, each (N, D)."""
    from scipy import linalg

    features_a = np.asarray(features_a, dtype=np.float64)
    features_b = np.asarray(features_b, dtype=np.float64)

    mu_a, mu_b = features_a.mean(axis=0), features_b.mean(axis=0)
    sigma_a = np.cov(features_a, rowvar=False)
    sigma_b = np.cov(features_b, rowvar=False)

    diff = mu_a - mu_b

    def matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
        # scipy removed the `disp` keyword in recent versions; older ones return a
        # (result, error) tuple when it is passed. Support both.
        try:
            result = linalg.sqrtm(matrix, disp=False)
        except TypeError:
            result = linalg.sqrtm(matrix)
        return result[0] if isinstance(result, tuple) else result

    covmean = matrix_sqrt(sigma_a.dot(sigma_b))

    # sqrtm of a product of PSD matrices can pick up tiny imaginary components from
    # numerical error; nudging the diagonal and taking the real part is the standard fix.
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = matrix_sqrt((sigma_a + offset).dot(sigma_b + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("FID: matrix square root has a large imaginary component")
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean))


def frames_to_rgb(frames: torch.Tensor, size: int, mean: tuple, std: tuple) -> torch.Tensor:
    """(N, 1, H, W) in [-1, 1] -> normalised 3-channel tensor at `size` x `size`.

    Our frames are grayscale and small; every pretrained extractor expects RGB at a
    larger resolution, so we replicate the channel and resize bilinearly.
    """
    if frames.shape[1] == 1:
        frames = frames.repeat(1, 3, 1, 1)
    frames = (frames.clamp(-1.0, 1.0) + 1.0) / 2.0  # -> [0, 1]
    frames = F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)
    mean_t = torch.tensor(mean, device=frames.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=frames.device).view(1, 3, 1, 1)
    return (frames - mean_t) / std_t


class CLIPImageFeatures:
    """Frozen CLIP image tower as an FID feature extractor (512-d)."""

    input_size = 224
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k",
                 device: str | torch.device = "cpu") -> None:
        import open_clip

        self.device = torch.device(device)
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        batch = frames_to_rgb(frames.to(self.device), self.input_size, self.mean, self.std)
        return self.model.encode_image(batch).float().cpu()


class InceptionFeatures:
    """Standard InceptionV3 pool3 features (2048-d), for conventional FID."""

    input_size = 299
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, device: str | torch.device = "cpu") -> None:
        from torchvision.models import Inception_V3_Weights, inception_v3

        self.device = torch.device(device)
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        model.fc = torch.nn.Identity()  # expose the 2048-d pooled features
        self.model = model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        batch = frames_to_rgb(frames.to(self.device), self.input_size, self.mean, self.std)
        return self.model(batch).float().cpu()


def extract_frame_features(
    videos: torch.Tensor,
    extractor: FeatureExtractor,
    batch_size: int = 64,
    max_frames_per_clip: int | None = None,
) -> np.ndarray:
    """Embed every frame of every clip. (N, T, 1, H, W) -> (N*T', D).

    FID is computed over frames rather than clips: with a few hundred test clips there
    are not nearly enough *videos* to estimate a stable 512x512 covariance, but there
    are thousands of frames.
    """
    if videos.ndim != 5:
        raise ValueError(f"expected (N, T, C, H, W), got {tuple(videos.shape)}")
    if max_frames_per_clip is not None:
        videos = videos[:, :max_frames_per_clip]

    flat = videos.reshape(-1, *videos.shape[2:])
    chunks = [
        extractor(flat[start : start + batch_size]).numpy()
        for start in range(0, flat.shape[0], batch_size)
    ]
    return np.concatenate(chunks, axis=0)


def compute_fid(
    real_videos: torch.Tensor,
    generated_videos: torch.Tensor,
    extractor: FeatureExtractor,
    batch_size: int = 64,
    max_frames_per_clip: int | None = None,
) -> dict[str, float]:
    """FID between real and generated frames, plus the sample counts behind it."""
    real_features = extract_frame_features(
        real_videos, extractor, batch_size, max_frames_per_clip
    )
    fake_features = extract_frame_features(
        generated_videos, extractor, batch_size, max_frames_per_clip
    )
    return {
        "fid": frechet_distance(real_features, fake_features),
        "fid_real_frames": int(real_features.shape[0]),
        "fid_generated_frames": int(fake_features.shape[0]),
        "fid_feature_dim": int(real_features.shape[1]),
    }


def build_feature_extractor(
    kind: Literal["clip", "inception"] = "clip",
    device: str | torch.device = "cpu",
) -> FeatureExtractor:
    if kind == "clip":
        return CLIPImageFeatures(device=device)
    if kind == "inception":
        return InceptionFeatures(device=device)
    raise ValueError(f"unknown FID feature extractor {kind!r}")
