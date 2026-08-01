"""Turn generated clips into things a human can look at: GIFs, MP4s, contact sheets.

Used by the data-inspection script, the evaluation sample galleries, and the demo app.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from text2video.data.dataset import tensor_to_frames


def as_uint8_clip(clip: np.ndarray | torch.Tensor) -> np.ndarray:
    """Accept a clip in any of our representations, return (T, H, W) uint8.

    Handles torch tensors in [-1, 1] with or without a channel axis, and numpy arrays
    that are already uint8.
    """
    if isinstance(clip, torch.Tensor):
        return tensor_to_frames(clip)

    array = np.asarray(clip)
    if array.dtype == np.uint8:
        return array.squeeze() if array.ndim == 4 else array
    # Float numpy: assume [-1, 1] like everything else in the project.
    tensor = torch.from_numpy(array.astype(np.float32))
    return tensor_to_frames(tensor)


def save_gif(clip: np.ndarray | torch.Tensor, path: str | Path, fps: int = 8) -> Path:
    """Write one clip as an animated GIF."""
    import imageio.v2 as imageio

    frames = as_uint8_clip(clip)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), duration=1.0 / fps, loop=0)
    return path


def save_mp4(clip: np.ndarray | torch.Tensor, path: str | Path, fps: int = 8) -> Path:
    """Write one clip as MP4. Falls back to GIF if no ffmpeg backend is available."""
    import imageio.v2 as imageio

    frames = as_uint8_clip(clip)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # MP4 encoders need even dimensions; ours are 64x64 so this is safe.
        imageio.mimsave(path, [np.stack([f] * 3, axis=-1) for f in frames], fps=fps)
        return path
    except Exception:  # noqa: BLE001 - ffmpeg is optional, GIF is an acceptable fallback
        return save_gif(frames, path.with_suffix(".gif"), fps=fps)


def filmstrip(clip: np.ndarray | torch.Tensor, every: int = 1, pad: int = 2) -> np.ndarray:
    """Lay a clip's frames out left-to-right in one image.

    Far more useful than a GIF when reviewing many samples at once, and it is what goes
    into the README: motion is visible as the digit's position shifting across the strip.
    """
    frames = as_uint8_clip(clip)[::every]
    t, h, w = frames.shape
    canvas = np.zeros((h, t * w + (t - 1) * pad), dtype=np.uint8)
    for i, frame in enumerate(frames):
        x = i * (w + pad)
        canvas[:, x : x + w] = frame
    return canvas


def grid(
    clips: list[np.ndarray | torch.Tensor], every: int = 1, pad: int = 2
) -> np.ndarray:
    """Stack several filmstrips vertically -- one row per clip."""
    strips = [filmstrip(clip, every=every, pad=pad) for clip in clips]
    width = max(strip.shape[1] for strip in strips)
    height = sum(strip.shape[0] for strip in strips) + pad * (len(strips) - 1)
    canvas = np.zeros((height, width), dtype=np.uint8)
    y = 0
    for strip in strips:
        canvas[y : y + strip.shape[0], : strip.shape[1]] = strip
        y += strip.shape[0] + pad
    return canvas


def save_comparison_figure(
    clips: list[np.ndarray | torch.Tensor],
    captions: list[str],
    path: str | Path,
    every: int = 2,
    title: str | None = None,
) -> Path:
    """Save a labelled figure: one captioned filmstrip per row.

    This is the qualitative-results artifact -- what a reader looks at to judge whether
    the model is doing anything sensible.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: never try to open a window
    import matplotlib.pyplot as plt

    rows = len(clips)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 1.5 * rows), squeeze=False)
    for i, (clip, caption) in enumerate(zip(clips, captions)):
        ax = axes[i][0]
        ax.imshow(filmstrip(clip, every=every), cmap="gray", vmin=0, vmax=255)
        ax.set_title(caption, fontsize=8, loc="left")
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
