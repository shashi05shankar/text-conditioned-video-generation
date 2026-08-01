"""Training loop for both VAE variants.

One trainer serves both models -- they share an interface and a loss, which is the
whole point of the controlled comparison. Everything that could differ between runs
(model variant, learning rate, KL weight, seed) comes from config and is written into
the run record, so a result can always be traced back to what produced it.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from text2video.models.cvae import TextConditionedVAE, vae_loss
from text2video.training.checkpoint import CheckpointManager, write_history


def kl_beta_at(
    step: int, warmup_steps: int, beta_max: float, beta_start: float = 0.0
) -> float:
    """Linear KL warm-up.

    Training with the full KL weight from step 0 is the classic way to get posterior
    collapse: before the decoder can reconstruct anything, the cheapest way to cut the
    loss is to make the posterior equal the prior, after which the latent carries no
    information and never recovers. Ramping the weight in lets reconstruction get
    established first.
    """
    if warmup_steps <= 0:
        return beta_max
    progress = min(1.0, step / warmup_steps)
    return beta_start + (beta_max - beta_start) * progress


class Trainer:
    """Trains a `TextConditionedVAE` with resumable checkpointing and run logging."""

    def __init__(
        self,
        model: TextConditionedVAE,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: dict[str, Any],
        device: torch.device | str = "cpu",
        run_dir: str | Path = "outputs/runs/run",
        mirror_dir: str | Path | None = None,
        use_amp: bool | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config.get("train", {})
        self.lr = float(train_cfg.get("lr", 2e-4))
        self.beta_max = float(train_cfg.get("kl_weight", 1e-4))
        self.beta_warmup_steps = int(train_cfg.get("kl_warmup_steps", 2000))
        self.free_bits = float(train_cfg.get("free_bits", 0.02))
        self.recon_loss = str(train_cfg.get("recon_loss", "mse"))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.log_every = int(train_cfg.get("log_every", 50))
        self.eval_every = int(train_cfg.get("eval_every", 500))
        self.max_eval_batches = int(train_cfg.get("max_eval_batches", 20))

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        # Mixed precision is a real memory saving on the GPU we actually train on, and a
        # no-op on CPU where fp16 is not supported.
        self.use_amp = (self.device.type == "cuda") if use_amp is None else use_amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.checkpoints = CheckpointManager(
            checkpoint_dir=self.run_dir / "checkpoints",
            save_every_steps=int(train_cfg.get("save_every_steps", 1000)),
            save_every_minutes=float(train_cfg.get("save_every_minutes", 10.0)),
            keep_last=int(train_cfg.get("keep_last", 2)),
            mirror_dir=mirror_dir,
            run_name=str(config.get("run_name", "run")),
        )

        self.global_step = 0
        self.epoch = 0
        self.history: list[dict[str, Any]] = []
        self.train_seconds = 0.0
        self.peak_memory_bytes = 0

    # -- core loop ----------------------------------------------------------

    def _forward_loss(self, batch: dict[str, Any], beta: float) -> dict[str, torch.Tensor]:
        video = batch["frames"].to(self.device, non_blocking=True)
        text = batch["text_emb"].to(self.device, non_blocking=True)
        outputs = self.model(video, text)
        return vae_loss(
            outputs,
            video,
            beta=beta,
            free_bits=self.free_bits,
            recon_loss=self.recon_loss,
        )

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        self.model.train()
        beta = kl_beta_at(self.global_step, self.beta_warmup_steps, self.beta_max)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            losses = self._forward_loss(batch, beta)

        self.scaler.scale(losses["loss"]).backward()
        if self.grad_clip > 0:
            # Unscale before clipping, or the clip threshold is applied to scaled
            # gradients and effectively does nothing.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.global_step += 1
        return {k: float(v.item()) for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Validation loss on held-out clips (unseen MNIST digit renderings)."""
        if self.val_loader is None:
            return {}
        self.model.eval()
        beta = kl_beta_at(self.global_step, self.beta_warmup_steps, self.beta_max)

        totals: dict[str, float] = {}
        count = 0
        for i, batch in enumerate(self.val_loader):
            if i >= self.max_eval_batches:
                break
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                losses = self._forward_loss(batch, beta)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.item())
            count += 1

        return {f"val_{k}": v / max(1, count) for k, v in totals.items()}

    def train(self, max_steps: int) -> dict[str, Any]:
        """Run until `max_steps` optimizer steps have been taken."""
        print(
            f"training {self.model.variant} for {max_steps} steps on {self.device} "
            f"(amp={self.use_amp}, params={self.model.describe()['total_params']:,})"
        )
        start = time.time()
        running: dict[str, float] = {}
        running_count = 0

        while self.global_step < max_steps:
            self.epoch += 1
            for batch in self.train_loader:
                if self.global_step >= max_steps:
                    break

                metrics = self.train_step(batch)
                for key, value in metrics.items():
                    running[key] = running.get(key, 0.0) + value
                running_count += 1

                if self.global_step % self.log_every == 0:
                    averaged = {k: v / running_count for k, v in running.items()}
                    elapsed = time.time() - start
                    print(
                        f"step {self.global_step:>6}/{max_steps} "
                        f"loss={averaged['loss']:.5f} recon={averaged['recon']:.5f} "
                        f"kl={averaged['kl']:.3f} beta={averaged['beta']:.2e} "
                        f"{self.global_step / max(elapsed, 1e-9):.2f} it/s"
                    )
                    record = {
                        "step": self.global_step,
                        "epoch": self.epoch,
                        "elapsed_s": round(elapsed, 2),
                        **{k: round(v, 6) for k, v in averaged.items()},
                    }
                    self.history.append(record)
                    running, running_count = {}, 0

                if self.eval_every > 0 and self.global_step % self.eval_every == 0:
                    val_metrics = self.evaluate()
                    if val_metrics:
                        print(
                            f"  [val] loss={val_metrics['val_loss']:.5f} "
                            f"recon={val_metrics['val_recon']:.5f} "
                            f"kl={val_metrics['val_kl']:.3f}"
                        )
                        self.history.append(
                            {"step": self.global_step, **{k: round(v, 6) for k, v in val_metrics.items()}}
                        )
                        self._save(val_metrics.get("val_loss"))

                if self.checkpoints.should_save(self.global_step):
                    self._save(None)

        self.train_seconds = time.time() - start
        if self.device.type == "cuda":
            self.peak_memory_bytes = int(torch.cuda.max_memory_allocated())

        final_val = self.evaluate()
        self._save(final_val.get("val_loss"), tag="final")
        return self.write_run_record(final_val)

    def _save(self, metric: float | None, tag: str | None = None) -> None:
        self.checkpoints.save(
            model=self.model,
            optimizer=self.optimizer,
            global_step=self.global_step,
            epoch=self.epoch,
            config=self.config,
            history=self.history,
            metric=metric,
            tag=tag,
        )
        write_history(self.run_dir / "history.json", self.history)

    def resume(self, checkpoint_path: str | Path) -> int:
        """Restore a run exactly where it stopped, including RNG state."""
        from text2video.training.checkpoint import load_checkpoint

        payload = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            map_location=self.device,
        )
        self.global_step = int(payload.get("global_step", 0))
        self.epoch = int(payload.get("epoch", 0))
        self.history = list(payload.get("history", []))
        print(f"resumed from {checkpoint_path} at step {self.global_step}")
        return self.global_step

    # -- run record ---------------------------------------------------------

    def write_run_record(self, final_val: dict[str, float]) -> dict[str, Any]:
        """Write everything needed to interpret and reproduce this run.

        Nothing in the final comparison table is allowed to be typed in by hand -- it is
        read back out of these files.
        """
        record: dict[str, Any] = {
            "run_name": self.config.get("run_name"),
            "model": self.model.describe(),
            "config": self.config,
            "device": str(self.device),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "global_step": self.global_step,
            "epochs": self.epoch,
            "train_seconds": round(self.train_seconds, 2),
            "train_minutes": round(self.train_seconds / 60, 2),
            "seconds_per_step": round(self.train_seconds / max(1, self.global_step), 4),
            "peak_gpu_memory_mb": round(self.peak_memory_bytes / 1024 / 1024, 1)
            if self.peak_memory_bytes
            else None,
            "final_train_metrics": self.history[-1] if self.history else {},
            "final_val_metrics": final_val,
            "amp": self.use_amp,
        }
        path = self.run_dir / "run_record.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"wrote run record -> {path}")
        return record


def build_optimizer_free_module_report(model: nn.Module) -> dict[str, int]:
    """Parameter count per top-level submodule -- useful when explaining the model."""
    return {
        name: sum(p.numel() for p in child.parameters() if p.requires_grad)
        for name, child in model.named_children()
    }
