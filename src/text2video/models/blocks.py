"""Building blocks shared by the baseline and the ConvLSTM model.

Both models are deliberately built from the *same* encoder, the same conditioning
mechanism, and the same decoder-to-pixels stack. The only thing that differs is how the
16 frames are produced from the latent: independently per frame (baseline) versus a
recurrent state carried across frames (main model). Keeping everything else identical is
what makes the comparison attributable to temporal modelling rather than to incidental
architecture differences.

Shape conventions (B = batch, T = 16 frames, C = 1 channel, H = W = 64):
    video      (B, T, C, H, W)
    per-frame  (B*T, C, H, W)   -- frames folded into the batch axis for 2D convs
    bottleneck (B*T, base_ch, 4, 4)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fold_time(video: torch.Tensor) -> tuple[torch.Tensor, int]:
    """(B, T, C, H, W) -> (B*T, C, H, W), plus T so it can be unfolded again."""
    b, t = video.shape[0], video.shape[1]
    return video.reshape(b * t, *video.shape[2:]), t


def unfold_time(flat: torch.Tensor, num_frames: int) -> torch.Tensor:
    """(B*T, C, H, W) -> (B, T, C, H, W)."""
    return flat.reshape(-1, num_frames, *flat.shape[1:])


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: conditions a feature map on a vector.

    Predicts a per-channel scale and shift from the conditioning vector and applies
    `x * (1 + gamma) + beta`. This is how the text embedding reaches every resolution of
    the decoder rather than only the bottleneck -- cheap (two linear layers) and it
    leaves the spatial structure untouched.

    The `1 +` means a zero-initialised layer starts as the identity, so conditioning
    ramps in smoothly instead of destroying the signal at step 0.
    """

    def __init__(self, cond_dim: int, num_features: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, num_features * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class ConvLSTMCell(nn.Module):
    """One step of a convolutional LSTM.

    An ordinary LSTM flattens its input to a vector, which throws away spatial layout.
    A ConvLSTM replaces the gate matrix-multiplies with convolutions, so hidden and cell
    states stay (channels, height, width) feature maps. For video that matters: "the
    digit is here and moving right" is a *spatial* fact, and the cell state can carry it
    across timesteps without ever flattening.

    Gates follow the standard LSTM formulation, computed in one fused convolution over
    the concatenated input and previous hidden state:
        i = sigma(W_i * [x, h])     input gate   -- how much new information to write
        f = sigma(W_f * [x, h])     forget gate  -- how much of the cell state to keep
        o = sigma(W_o * [x, h])     output gate  -- how much of the cell state to expose
        g = tanh (W_g * [x, h])     candidate    -- the new information itself
        c' = f * c + i * g
        h' = o * tanh(c')
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # One convolution produces all four gates at once (4 * hidden channels out).
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        # Forget-gate bias starts positive so the cell remembers by default; with a zero
        # bias, sigmoid(0)=0.5 halves the cell state every step and long-range motion
        # information decays before it can be used.
        nn.init.zeros_(self.conv.bias)
        with torch.no_grad():
            self.conv.bias[hidden_channels : 2 * hidden_channels].fill_(1.0)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        gates = self.conv(torch.cat([x, h_prev], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch_size, self.hidden_channels, height, width)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )


class VideoEncoder(nn.Module):
    """Encodes a clip into a single vector summarising content *and* motion.

    Per-frame 2D convolutions extract appearance features; a GRU over time then folds
    the frame sequence into one vector, which is what lets the latent carry motion
    (direction, speed) and not just the digit's look.

    Used only during training -- at generation time the latent is sampled from the
    text-conditioned prior, so the encoder is never on the inference path. Both models
    share this encoder unchanged.

    (B, T, 1, 64, 64) -> (B, feature_dim)
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32, feature_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(in_channels, base_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 -> 4
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.to_frame_feature = nn.Linear(base_ch * 8 * 4 * 4, feature_dim)
        self.temporal = nn.GRU(feature_dim, feature_dim, batch_first=True)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        flat, num_frames = fold_time(video)
        features = self.conv(flat).flatten(1)
        features = self.to_frame_feature(features)
        sequence = unfold_time(features, num_frames)  # (B, T, feature_dim)
        _, hidden = self.temporal(sequence)
        return hidden.squeeze(0)  # (B, feature_dim)


class FrameDecoder(nn.Module):
    """Upsamples a (B*T, base_ch, 4, 4) bottleneck to (B*T, 1, 64, 64) frames.

    Shared verbatim by both models, so any quality difference between them comes from
    how the bottleneck was produced, not from the pixel decoder.

    FiLM conditioning is applied at every resolution so the caption keeps influencing
    the image as it is upsampled, rather than being diluted after the bottleneck.
    Output is `tanh`, matching the [-1, 1] frame normalisation.
    """

    def __init__(self, base_ch: int = 128, cond_dim: int = 256, out_channels: int = 1) -> None:
        super().__init__()
        channels = [base_ch, base_ch, base_ch // 2, base_ch // 4]

        # 4 -> 8 -> 16 -> 32, then a final 32 -> 64.
        self.up1 = nn.ConvTranspose2d(channels[0], channels[1], 4, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(channels[1])
        self.film1 = FiLM(cond_dim, channels[1])

        self.up2 = nn.ConvTranspose2d(channels[1], channels[2], 4, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(channels[2])
        self.film2 = FiLM(cond_dim, channels[2])

        self.up3 = nn.ConvTranspose2d(channels[2], channels[3], 4, stride=2, padding=1)
        self.norm3 = nn.BatchNorm2d(channels[3])
        self.film3 = FiLM(cond_dim, channels[3])

        self.up4 = nn.ConvTranspose2d(channels[3], channels[3], 4, stride=2, padding=1)
        self.norm4 = nn.BatchNorm2d(channels[3])
        self.film4 = FiLM(cond_dim, channels[3])

        # A plain 3x3 at full resolution cleans up the checkerboard artifacts that
        # transposed convolutions are prone to.
        self.to_pixels = nn.Conv2d(channels[3], out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.film1(self.norm1(self.up1(x)), cond))
        x = F.relu(self.film2(self.norm2(self.up2(x)), cond))
        x = F.relu(self.film3(self.norm3(self.up3(x)), cond))
        x = F.relu(self.film4(self.norm4(self.up4(x)), cond))
        return torch.tanh(self.to_pixels(x))


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Parameter count, reported in every experiment log."""
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
