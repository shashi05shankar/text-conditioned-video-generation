"""Text-conditioned video VAEs: a frame-independent baseline and a ConvLSTM model.

Both models are the *same* conditional VAE. They share the encoder, the text
projection, the posterior and prior heads, the latent size, the pixel decoder and the
loss. Exactly one thing differs:

    baseline  -- each frame's bottleneck is produced independently from (z, text, t)
    convlstm  -- the bottleneck sequence is produced by a ConvLSTM whose hidden state
                 is carried from frame to frame

So any measured difference is attributable to the temporal mechanism rather than to
capacity or architecture noise. That comparison is the point of the project.

Why a VAE at all rather than a plain regressor: one caption matches many videos (any
handwriting style, any starting position). A deterministic model trained on MSE would
average them all into a blur. A VAE learns a *distribution*, and at generation time we
sample a latent from a text-conditioned prior, so one caption can yield different valid
clips.

Shapes:
    video     (B, T, 1, 64, 64)   in [-1, 1]
    text_emb  (B, 512)            frozen CLIP, L2-normalised
    z         (B, latent_dim)
    cond      (B, latent_dim + cond_dim)   -- FiLM conditioning for the pixel decoder
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from text2video.models.blocks import (
    ConvLSTMCell,
    FrameDecoder,
    VideoEncoder,
    fold_time,
    unfold_time,
)

BOTTLENECK_SIZE = 4  # decoder starts from a 4x4 spatial grid


class TextConditionedVAE(nn.Module):
    """Base class holding everything the two variants share.

    Subclasses implement `build_bottleneck`, which turns (z, text, T) into the sequence
    of (B, T, base_ch, 4, 4) feature maps the pixel decoder renders.
    """

    variant: str = "base"

    def __init__(
        self,
        text_dim: int = 512,
        cond_dim: int = 128,
        latent_dim: int = 128,
        feature_dim: int = 256,
        base_ch: int = 128,
        encoder_base_ch: int = 32,
        frame_emb_dim: int = 64,
        num_frames: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.base_ch = base_ch
        self.num_frames = num_frames
        self.frame_emb_dim = frame_emb_dim

        self.encoder = VideoEncoder(
            in_channels=1, base_ch=encoder_base_ch, feature_dim=feature_dim
        )
        # Projects frozen 512-d CLIP embeddings down to the size the model works in.
        # This is the only trainable part of the text pathway.
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # q(z | video, text): the posterior, used during training only.
        self.posterior = nn.Sequential(
            nn.Linear(feature_dim + cond_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, latent_dim * 2),
        )
        # p(z | text): the learned prior. This is what makes text-only generation
        # possible -- at inference we sample from here, with no video input at all.
        self.prior = nn.Sequential(
            nn.Linear(cond_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, latent_dim * 2),
        )

        # Tells the model which frame it is producing. The baseline needs this to have
        # any chance of generating motion; the ConvLSTM gets it too, so the two differ
        # only in the recurrence.
        self.frame_embedding = nn.Embedding(num_frames, frame_emb_dim)

        self.decoder = FrameDecoder(
            base_ch=base_ch, cond_dim=latent_dim + cond_dim, out_channels=1
        )

    # -- shared pieces ------------------------------------------------------

    def encode_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        return self.text_proj(text_emb)

    def encode_posterior(
        self, video: torch.Tensor, text_cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(video)
        mu, logvar = self.posterior(torch.cat([features, text_cond], dim=-1)).chunk(2, dim=-1)
        return mu, self._clamp_logvar(logvar)

    def encode_prior(self, text_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.prior(text_cond).chunk(2, dim=-1)
        return mu, self._clamp_logvar(logvar)

    @staticmethod
    def _clamp_logvar(logvar: torch.Tensor) -> torch.Tensor:
        """Keep log-variance in a sane range.

        Unclamped, an over-confident posterior drives logvar towards -inf, exp(logvar)
        underflows to 0 and the KL term becomes NaN. This is the single most common way
        VAE training dies, so it is cheap insurance.
        """
        return logvar.clamp(-8.0, 8.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps -- keeps sampling differentiable w.r.t. mu and sigma."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def frame_inputs(
        self, z: torch.Tensor, text_cond: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """Per-frame conditioning vectors: (B, T, latent + cond + frame_emb)."""
        batch = z.shape[0]
        indices = torch.arange(num_frames, device=z.device)
        frame_emb = self.frame_embedding(indices)                    # (T, frame_emb_dim)
        frame_emb = frame_emb.unsqueeze(0).expand(batch, -1, -1)     # (B, T, frame_emb_dim)
        static = torch.cat([z, text_cond], dim=-1)                   # (B, latent + cond)
        static = static.unsqueeze(1).expand(-1, num_frames, -1)
        return torch.cat([static, frame_emb], dim=-1)

    def build_bottleneck(
        self, z: torch.Tensor, text_cond: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        raise NotImplementedError

    def decode(
        self, z: torch.Tensor, text_cond: torch.Tensor, num_frames: int | None = None
    ) -> torch.Tensor:
        """(z, text) -> (B, T, 1, 64, 64) video."""
        num_frames = num_frames or self.num_frames
        bottleneck = self.build_bottleneck(z, text_cond, num_frames)  # (B, T, base_ch, 4, 4)
        flat, _ = fold_time(bottleneck)

        cond = torch.cat([z, text_cond], dim=-1)                      # (B, latent + cond)
        cond = cond.unsqueeze(1).expand(-1, num_frames, -1).reshape(flat.shape[0], -1)

        frames = self.decoder(flat, cond)
        return unfold_time(frames, num_frames)

    def forward(self, video: torch.Tensor, text_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        """Training forward pass: encode, sample, reconstruct."""
        num_frames = video.shape[1]
        text_cond = self.encode_text(text_emb)

        mu_q, logvar_q = self.encode_posterior(video, text_cond)
        mu_p, logvar_p = self.encode_prior(text_cond)
        z = self.reparameterize(mu_q, logvar_q)

        return {
            "recon": self.decode(z, text_cond, num_frames),
            "mu_q": mu_q,
            "logvar_q": logvar_q,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
            "z": z,
        }

    @torch.no_grad()
    def generate(
        self,
        text_emb: torch.Tensor,
        num_frames: int | None = None,
        temperature: float = 1.0,
        use_prior_mean: bool = False,
    ) -> torch.Tensor:
        """Text -> video, with no ground-truth frames involved.

        Args:
            temperature: scales the prior standard deviation. <1 gives more typical,
                less diverse clips; >1 gives more variety and more artifacts.
            use_prior_mean: take the prior mean instead of sampling -- deterministic,
                useful for reproducible qualitative figures.
        """
        text_cond = self.encode_text(text_emb)
        mu_p, logvar_p = self.encode_prior(text_cond)
        if use_prior_mean:
            z = mu_p
        else:
            std = torch.exp(0.5 * logvar_p) * temperature
            z = mu_p + std * torch.randn_like(std)
        return self.decode(z, text_cond, num_frames or self.num_frames)

    def describe(self) -> dict[str, Any]:
        """Parameter breakdown, recorded in every experiment log."""
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "variant": self.variant,
            "total_params": count(self),
            "encoder_params": count(self.encoder),
            "decoder_params": count(self.decoder),
            "latent_dim": self.latent_dim,
            "cond_dim": self.cond_dim,
            "base_ch": self.base_ch,
            "num_frames": self.num_frames,
        }


class FrameIndependentVAE(TextConditionedVAE):
    """BASELINE. Every frame's bottleneck is computed independently.

    The model is told which frame index it is generating, so it *can* in principle
    produce motion -- it simply has no mechanism that ties frame t to frame t-1. Each
    frame is a separate function call sharing only the clip-level latent.

    Giving it the frame index matters for fairness. A baseline that emitted the same
    frame 16 times would score a perfect temporal-consistency number while generating no
    motion at all, which is precisely the metric-gaming failure the evaluation is
    designed to catch. This baseline is capable of motion; what it lacks is continuity.
    """

    variant = "baseline"

    def __init__(self, hidden_dim: int = 512, **kwargs: Any) -> None:
        """
        Args:
            hidden_dim: width of the per-frame bottleneck MLP. The default (512) gives
                a natural-sized baseline at ~4.3M params against the ConvLSTM's ~6.5M.
                Setting it to ~1500 produces a capacity-matched control, which is how we
                rule out "the ConvLSTM only won because it had more parameters".
        """
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        in_dim = self.latent_dim + self.cond_dim + self.frame_emb_dim
        self.to_bottleneck = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.base_ch * BOTTLENECK_SIZE * BOTTLENECK_SIZE),
        )

    def build_bottleneck(
        self, z: torch.Tensor, text_cond: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        inputs = self.frame_inputs(z, text_cond, num_frames)     # (B, T, in_dim)
        batch = inputs.shape[0]
        # All frames go through the same MLP in parallel -- no ordering, no state.
        flat = self.to_bottleneck(inputs.reshape(batch * num_frames, -1))
        return flat.reshape(
            batch, num_frames, self.base_ch, BOTTLENECK_SIZE, BOTTLENECK_SIZE
        )


class ConvLSTMVAE(TextConditionedVAE):
    """MAIN MODEL. The bottleneck sequence is produced by a ConvLSTM.

    Identical to the baseline except that the per-frame conditioning vector is fed into
    a ConvLSTM cell whose hidden and cell states persist across the 16 steps. The state
    is a (channels, 4, 4) feature map, so "where the digit is and where it is heading"
    survives from one frame to the next as spatial information rather than being
    re-derived from a frame index every time.

    That recurrence is the entire experimental variable of this project.
    """

    variant = "convlstm"

    def __init__(self, hidden_ch: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hidden_ch = hidden_ch or self.base_ch
        in_dim = self.latent_dim + self.cond_dim + self.frame_emb_dim

        # Same input projection as the baseline, so the recurrence is the only difference.
        self.to_step_input = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.SiLU(),
            nn.Linear(512, self.base_ch * BOTTLENECK_SIZE * BOTTLENECK_SIZE),
        )
        self.cell = ConvLSTMCell(self.base_ch, self.hidden_ch, kernel_size=3)
        # Initialising the hidden state from (z, text) rather than zeros means the
        # recurrence starts already knowing what it is supposed to draw.
        self.init_state_proj = nn.Linear(
            self.latent_dim + self.cond_dim,
            2 * self.hidden_ch * BOTTLENECK_SIZE * BOTTLENECK_SIZE,
        )
        if self.hidden_ch != self.base_ch:
            self.state_to_bottleneck: nn.Module = nn.Conv2d(self.hidden_ch, self.base_ch, 1)
        else:
            self.state_to_bottleneck = nn.Identity()

    def build_bottleneck(
        self, z: torch.Tensor, text_cond: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        inputs = self.frame_inputs(z, text_cond, num_frames)
        batch = inputs.shape[0]

        step_inputs = self.to_step_input(inputs.reshape(batch * num_frames, -1))
        step_inputs = step_inputs.reshape(
            batch, num_frames, self.base_ch, BOTTLENECK_SIZE, BOTTLENECK_SIZE
        )

        h, c = self._initial_state(z, text_cond)
        outputs = []
        for t in range(num_frames):
            # The loop is the mechanism: frame t sees the state left behind by frame t-1.
            h, c = self.cell(step_inputs[:, t], (h, c))
            outputs.append(self.state_to_bottleneck(h))
        return torch.stack(outputs, dim=1)

    def _initial_state(
        self, z: torch.Tensor, text_cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = z.shape[0]
        state = self.init_state_proj(torch.cat([z, text_cond], dim=-1))
        state = state.reshape(
            batch, 2, self.hidden_ch, BOTTLENECK_SIZE, BOTTLENECK_SIZE
        )
        # tanh keeps the initial cell state in the range an LSTM cell state normally
        # occupies; an unbounded init destabilises the first few steps.
        return torch.tanh(state[:, 0]), torch.tanh(state[:, 1])


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def gaussian_kl(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """Analytic KL( q(z|x,c) || p(z|c) ) for diagonal Gaussians, per dimension.

    Both sides are learned here: q is the posterior seeing the video, p is the prior
    seeing only the text. Pulling them together is what forces the text-conditioned
    prior to become a usable generator -- at inference we sample from p, so if p and q
    disagree, generation and reconstruction are solving different problems.

    Returns (B, latent_dim) so free-bits can be applied per dimension.
    """
    return 0.5 * (
        logvar_p
        - logvar_q
        + (logvar_q - logvar_p).exp()
        + (mu_q - mu_p).pow(2) * torch.exp(-logvar_p)
        - 1.0
    )


def vae_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.0,
    recon_loss: str = "mse",
) -> dict[str, torch.Tensor]:
    """Reconstruction + beta-weighted KL.

    Args:
        beta: KL weight. Annealed from 0 during training -- with full KL from step 0 the
            model collapses the posterior to the prior and ignores the video before it
            has learned to reconstruct anything.
        free_bits: minimum nats per latent dimension that incur no KL penalty. Prevents
            posterior collapse, where unused dimensions get driven to exactly the prior
            and the latent silently stops carrying information.
        recon_loss: "mse" (Gaussian likelihood, the standard VAE choice) or "l1"
            (sharper in practice, less principled).
    """
    recon = outputs["recon"]
    if recon_loss == "l1":
        per_sample_recon = (recon - target).abs().flatten(1).mean(dim=1)
    elif recon_loss == "mse":
        per_sample_recon = (recon - target).pow(2).flatten(1).mean(dim=1)
    else:
        raise ValueError(f"unknown recon_loss {recon_loss!r}")

    kl_per_dim = gaussian_kl(
        outputs["mu_q"], outputs["logvar_q"], outputs["mu_p"], outputs["logvar_p"]
    )
    kl_raw = kl_per_dim.sum(dim=1)

    if free_bits > 0.0:
        kl_effective = kl_per_dim.clamp_min(free_bits).sum(dim=1)
    else:
        kl_effective = kl_raw

    recon_term = per_sample_recon.mean()
    kl_term = kl_effective.mean()

    return {
        "loss": recon_term + beta * kl_term,
        "recon": recon_term,
        "kl": kl_raw.mean(),          # unclamped, for honest monitoring
        "kl_effective": kl_term,
        "beta": torch.tensor(beta),
    }


def build_model(name: str, **kwargs: Any) -> TextConditionedVAE:
    """Instantiate a model by name (the string used in configs and run logs)."""
    models: dict[str, type[TextConditionedVAE]] = {
        "baseline": FrameIndependentVAE,
        "convlstm": ConvLSTMVAE,
    }
    if name not in models:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(models)}")
    return models[name](**kwargs)
