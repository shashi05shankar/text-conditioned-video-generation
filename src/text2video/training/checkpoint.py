"""Resumable checkpointing, built for free-tier Colab/Kaggle sessions.

Free sessions disconnect without warning, often mid-run. A checkpoint that only holds
model weights is not enough to continue: resuming with a fresh optimizer resets Adam's
moment estimates and visibly disturbs training. So a checkpoint here carries everything
needed to continue as if nothing happened:

    model weights, optimizer state, LR scheduler state, global step, epoch,
    RNG states (torch / numpy / python), the config snapshot, and metric history.

Checkpoints are saved on a step interval *and* a wall-clock timer, and optionally
mirrored to a second directory (Google Drive) that outlives the session.
"""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except (RuntimeError, ValueError):
            # Resuming on a machine with a different GPU count -- not worth failing over.
            pass


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    global_step: int = 0,
    epoch: int = 0,
    config: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
    mirror_dir: str | Path | None = None,
) -> Path:
    """Write a full resumable checkpoint, atomically.

    The write goes to a temporary file first and is then moved into place, so a session
    killed mid-save cannot leave a truncated checkpoint that fails to load later.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "global_step": global_step,
        "epoch": epoch,
        "config": config or {},
        "history": history or [],
        "rng": _rng_state(),
        "saved_at": time.time(),
        **(extra or {}),
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)

    if mirror_dir is not None:
        mirror_dir = Path(mirror_dir)
        mirror_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, mirror_dir / path.name)

    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint, optionally restoring model/optimizer/scheduler in place."""
    path = Path(path)
    # weights_only=False: our payload holds numpy RNG state and the config dict, not
    # just tensors. These are checkpoints we wrote ourselves.
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        _restore_rng_state(payload.get("rng", {}))

    return payload


def find_latest_checkpoint(directory: str | Path, pattern: str = "*.pt") -> Path | None:
    """Most recently modified checkpoint in a directory, for `--resume auto`."""
    directory = Path(directory)
    if not directory.exists():
        return None
    candidates = [p for p in directory.glob(pattern) if not p.name.endswith(".tmp")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


class CheckpointManager:
    """Decides when to save, and keeps the directory from filling up.

    Two independent triggers:
      * every `save_every_steps` optimizer steps
      * every `save_every_minutes` of wall-clock time

    The timer matters more than it looks: on a free GPU an epoch can take longer than
    the disconnect window, so a step-only policy can lose an entire session's work.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_every_steps: int = 1000,
        save_every_minutes: float = 10.0,
        keep_last: int = 2,
        mirror_dir: str | Path | None = None,
        run_name: str = "run",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_steps = save_every_steps
        self.save_every_minutes = save_every_minutes
        self.keep_last = keep_last
        self.mirror_dir = Path(mirror_dir) if mirror_dir else None
        self.run_name = run_name

        self._last_save_time = time.time()
        self._last_save_step = 0
        self.best_metric: float | None = None

    def should_save(self, global_step: int) -> bool:
        by_steps = (global_step - self._last_save_step) >= self.save_every_steps
        by_time = (time.time() - self._last_save_time) >= self.save_every_minutes * 60
        return by_steps or by_time

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        epoch: int,
        config: dict[str, Any],
        history: list[dict[str, Any]],
        scheduler: Any | None = None,
        metric: float | None = None,
        metric_name: str = "val_loss",
        tag: str | None = None,
    ) -> Path:
        """Save a rolling checkpoint, plus `best.pt` when `metric` improves."""
        name = tag or f"{self.run_name}_step{global_step:07d}"
        path = save_checkpoint(
            self.checkpoint_dir / f"{name}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch=epoch,
            config=config,
            history=history,
            extra={"metric_name": metric_name, "metric": metric},
            mirror_dir=self.mirror_dir,
        )

        # `last.pt` is what --resume auto looks for; a stable name means resuming never
        # depends on parsing step numbers out of filenames.
        save_checkpoint(
            self.checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch=epoch,
            config=config,
            history=history,
            extra={"metric_name": metric_name, "metric": metric},
            mirror_dir=self.mirror_dir,
        )

        if metric is not None and (self.best_metric is None or metric < self.best_metric):
            self.best_metric = metric
            save_checkpoint(
                self.checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                epoch=epoch,
                config=config,
                history=history,
                extra={"metric_name": metric_name, "metric": metric},
                mirror_dir=self.mirror_dir,
            )

        self._last_save_time = time.time()
        self._last_save_step = global_step
        self._prune()
        return path

    def _prune(self) -> None:
        """Delete old step checkpoints, keeping `last.pt` and `best.pt` untouched."""
        step_checkpoints = sorted(
            (p for p in self.checkpoint_dir.glob(f"{self.run_name}_step*.pt")),
            key=lambda p: p.stat().st_mtime,
        )
        for old in step_checkpoints[: -self.keep_last] if self.keep_last > 0 else []:
            old.unlink(missing_ok=True)


def write_history(path: str | Path, history: list[dict[str, Any]]) -> Path:
    """Dump the metric history as JSON so plots can be rebuilt without the checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return path
