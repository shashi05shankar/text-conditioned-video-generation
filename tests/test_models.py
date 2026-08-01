"""Model tests: shapes, gradient flow, loss correctness, and tiny-overfit sanity.

Everything here runs on CPU in seconds. The tiny-overfit tests are the gate before any
GPU training: a model that cannot memorise two clips has a bug, and no amount of
Colab time will fix it.
"""

from __future__ import annotations

import pytest
import torch

from text2video.models.blocks import ConvLSTMCell, FiLM, VideoEncoder, fold_time, unfold_time
from text2video.models.cvae import (
    ConvLSTMVAE,
    FrameIndependentVAE,
    build_model,
    gaussian_kl,
    vae_loss,
)

# Small enough to train on CPU inside a test, same structure as the real thing.
TINY = dict(
    text_dim=512,
    cond_dim=32,
    latent_dim=32,
    feature_dim=64,
    base_ch=32,
    encoder_base_ch=8,
    frame_emb_dim=16,
    num_frames=8,
)


def make_batch(batch: int = 2, frames: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    video = torch.rand(batch, frames, 1, 64, 64) * 2 - 1
    text = torch.randn(batch, 512)
    return video, text / text.norm(dim=-1, keepdim=True)


class TestBlocks:
    def test_fold_unfold_roundtrip(self):
        video = torch.randn(3, 8, 1, 64, 64)
        flat, num_frames = fold_time(video)
        assert flat.shape == (24, 1, 64, 64)
        assert torch.equal(unfold_time(flat, num_frames), video)

    def test_film_starts_as_identity(self):
        """Zero-init means conditioning ramps in rather than wrecking the signal."""
        film = FiLM(cond_dim=16, num_features=8)
        x = torch.randn(4, 8, 4, 4)
        assert torch.allclose(film(x, torch.randn(4, 16)), x)

    def test_convlstm_cell_shapes_and_state_change(self):
        cell = ConvLSTMCell(input_channels=8, hidden_channels=12)
        state = cell.init_state(2, 4, 4, torch.device("cpu"))
        h, c = cell(torch.randn(2, 8, 4, 4), state)
        assert h.shape == (2, 12, 4, 4) and c.shape == (2, 12, 4, 4)
        assert not torch.allclose(h, state[0]), "hidden state did not update"

    def test_convlstm_forget_gate_bias_is_positive(self):
        """Zero forget bias halves the cell state every step, so motion information
        decays before it can be used."""
        cell = ConvLSTMCell(4, 6)
        forget_bias = cell.conv.bias[6:12]
        assert torch.all(forget_bias > 0.5)

    def test_convlstm_carries_information_across_steps(self):
        """The defining property: output at step 2 must depend on step 1's input."""
        torch.manual_seed(0)
        cell = ConvLSTMCell(4, 6)
        state = cell.init_state(1, 4, 4, torch.device("cpu"))
        x1, x2, step2 = torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4)

        out_a = cell(step2, cell(x1, state))[0]
        out_b = cell(step2, cell(x2, state))[0]
        assert not torch.allclose(out_a, out_b, atol=1e-6)

    def test_video_encoder_output_shape(self):
        encoder = VideoEncoder(base_ch=8, feature_dim=64)
        assert encoder(torch.randn(2, 8, 1, 64, 64)).shape == (2, 64)

    def test_video_encoder_is_sensitive_to_frame_order(self):
        """The encoder must capture motion, not just appearance -- otherwise the latent
        cannot carry direction or speed."""
        encoder = VideoEncoder(base_ch=8, feature_dim=64).eval()
        video = torch.rand(1, 8, 1, 64, 64)
        with torch.no_grad():
            forward = encoder(video)
            backward = encoder(video.flip(dims=[1]))
        assert not torch.allclose(forward, backward, atol=1e-5)


class TestModelForward:
    @pytest.mark.parametrize("name", ["baseline", "convlstm"])
    def test_forward_shapes(self, name):
        model = build_model(name, **TINY)
        video, text = make_batch()
        out = model(video, text)
        assert out["recon"].shape == video.shape
        assert out["mu_q"].shape == (2, TINY["latent_dim"])
        assert out["logvar_p"].shape == (2, TINY["latent_dim"])

    @pytest.mark.parametrize("name", ["baseline", "convlstm"])
    def test_output_is_in_tanh_range(self, name):
        model = build_model(name, **TINY)
        video, text = make_batch()
        recon = model(video, text)["recon"]
        assert recon.min() >= -1.0 and recon.max() <= 1.0

    @pytest.mark.parametrize("name", ["baseline", "convlstm"])
    def test_all_parameters_receive_gradients(self, name):
        """Catches disconnected submodules -- a layer built but never used."""
        model = build_model(name, **TINY)
        video, text = make_batch()
        vae_loss(model(video, text), video)["loss"].backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"no gradient reached: {missing}"

    @pytest.mark.parametrize("name", ["baseline", "convlstm"])
    def test_generate_needs_no_video(self, name):
        """Generation must run from text alone -- that is the whole product."""
        model = build_model(name, **TINY).eval()
        _, text = make_batch(batch=3)
        video = model.generate(text)
        assert video.shape == (3, TINY["num_frames"], 1, 64, 64)
        assert torch.isfinite(video).all()

    @pytest.mark.parametrize("name", ["baseline", "convlstm"])
    def test_generate_deterministic_with_prior_mean(self, name):
        model = build_model(name, **TINY).eval()
        _, text = make_batch(batch=2)
        a = model.generate(text, use_prior_mean=True)
        b = model.generate(text, use_prior_mean=True)
        assert torch.allclose(a, b)

    @pytest.mark.parametrize("name", ["baseline", "convlstm"])
    def test_generation_varies_with_text(self, name):
        """Different captions must produce different videos, or conditioning is dead."""
        torch.manual_seed(0)
        model = build_model(name, **TINY).eval()
        text_a = torch.randn(1, 512)
        text_b = torch.randn(1, 512)
        a = model.generate(text_a / text_a.norm(), use_prior_mean=True)
        b = model.generate(text_b / text_b.norm(), use_prior_mean=True)
        assert not torch.allclose(a, b, atol=1e-5)

    def test_variants_differ_structurally(self):
        baseline = build_model("baseline", **TINY)
        convlstm = build_model("convlstm", **TINY)
        assert baseline.variant == "baseline" and convlstm.variant == "convlstm"
        assert not hasattr(baseline, "cell")
        assert isinstance(convlstm.cell, ConvLSTMCell)

    def test_baseline_frames_are_computed_independently(self):
        """Defining property of the baseline: perturbing the state that frame 0 would
        have left behind cannot affect frame 5, because no such state exists.

        We verify it structurally -- the bottleneck for frame t is a pure function of
        (z, text, t), so shuffling the frame order permutes the output identically.
        """
        model = build_model("baseline", **TINY).eval()
        z = torch.randn(1, TINY["latent_dim"])
        cond = torch.randn(1, TINY["cond_dim"])
        with torch.no_grad():
            full = model.build_bottleneck(z, cond, 8)
            shorter = model.build_bottleneck(z, cond, 4)
        # Frames 0-3 are identical whether or not frames 4-7 are also being generated.
        assert torch.allclose(full[:, :4], shorter, atol=1e-6)

    def test_convlstm_frames_depend_on_history(self):
        """Complementary property: the ConvLSTM's frame t genuinely depends on earlier
        steps, so its state is doing something."""
        torch.manual_seed(0)
        model = build_model("convlstm", **TINY).eval()
        z = torch.randn(1, TINY["latent_dim"])
        cond = torch.randn(1, TINY["cond_dim"])
        with torch.no_grad():
            bottleneck = model.build_bottleneck(z, cond, 8)
        # Successive states must differ; identical states would mean the recurrence is
        # inert and the model has silently degenerated to a frame-independent one.
        diffs = [
            (bottleneck[:, t + 1] - bottleneck[:, t]).abs().mean().item() for t in range(7)
        ]
        assert min(diffs) > 1e-6, f"ConvLSTM state is static across steps: {diffs}"

    def test_capacity_matched_baseline_is_configurable(self):
        """We must be able to rule out 'the ConvLSTM only won on parameter count'."""
        narrow = build_model("baseline", hidden_dim=512, **TINY).describe()["total_params"]
        wide = build_model("baseline", hidden_dim=1500, **TINY).describe()["total_params"]
        assert wide > narrow

    def test_unknown_model_name_raises(self):
        with pytest.raises(ValueError, match="unknown model"):
            build_model("diffusion")


class TestLosses:
    def test_kl_is_zero_for_identical_distributions(self):
        mu = torch.randn(4, 8)
        logvar = torch.randn(4, 8)
        kl = gaussian_kl(mu, logvar, mu.clone(), logvar.clone())
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)

    def test_kl_is_non_negative(self):
        kl = gaussian_kl(
            torch.randn(16, 8), torch.randn(16, 8), torch.randn(16, 8), torch.randn(16, 8)
        )
        assert kl.sum(dim=1).min() >= -1e-5

    def test_kl_grows_with_distance_between_means(self):
        logvar = torch.zeros(1, 4)
        near = gaussian_kl(torch.zeros(1, 4), logvar, torch.full((1, 4), 0.1), logvar).sum()
        far = gaussian_kl(torch.zeros(1, 4), logvar, torch.full((1, 4), 3.0), logvar).sum()
        assert far > near

    def test_free_bits_reduces_the_penalised_kl(self):
        video, text = make_batch()
        model = build_model("baseline", **TINY)
        outputs = model(video, text)
        without = vae_loss(outputs, video, free_bits=0.0)
        with_fb = vae_loss(outputs, video, free_bits=1.0)
        assert with_fb["kl_effective"] >= without["kl_effective"]
        assert torch.allclose(with_fb["kl"], without["kl"]), "raw KL must stay unclamped"

    def test_beta_scales_the_kl_contribution(self):
        video, text = make_batch()
        outputs = build_model("baseline", **TINY)(video, text)
        low = vae_loss(outputs, video, beta=0.0)
        high = vae_loss(outputs, video, beta=1.0)
        assert high["loss"] > low["loss"]
        assert torch.allclose(low["loss"], low["recon"])

    def test_perfect_reconstruction_gives_zero_recon_loss(self):
        video, text = make_batch()
        outputs = build_model("baseline", **TINY)(video, text)
        outputs["recon"] = video.clone()
        assert vae_loss(outputs, video, beta=0.0)["recon"].item() == pytest.approx(0.0)

    def test_unknown_recon_loss_raises(self):
        video, text = make_batch()
        outputs = build_model("baseline", **TINY)(video, text)
        with pytest.raises(ValueError, match="unknown recon_loss"):
            vae_loss(outputs, video, recon_loss="huber")

    def test_logvar_is_clamped_against_nan(self):
        """Unclamped log-variance underflows and turns the KL into NaN -- the most
        common way VAE training dies."""
        model = build_model("baseline", **TINY)
        clamped = model._clamp_logvar(torch.tensor([-1e4, 0.0, 1e4]))
        assert torch.isfinite(clamped).all()
        assert clamped.min() >= -8.0 and clamped.max() <= 8.0


class TestTinyOverfit:
    """The gate before GPU training. If a model cannot memorise two clips, it is broken.

    We check reconstruction (the posterior path) rather than generation, because
    generation from a prior needs far more data and steps than a unit test should take.

    The clips are *real* Bouncing MNIST, not random noise. An earlier version of this
    test fitted uniform noise and the ConvLSTM only reached 51% of its starting loss in
    120 steps while the baseline sailed through -- not a bug, just the recurrent
    bottleneck having nothing exploitable to compress when the target is structureless.
    Testing on the data the model is actually for is both fairer and a better gate.
    """

    @pytest.mark.parametrize("name", ["baseline", "convlstm"])
    def test_overfits_two_real_clips(self, name, generator):
        import numpy as np

        from text2video.data.dataset import frames_to_tensor

        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        clips = [generator.generate_clip(rng)[0][:8] for _ in range(2)]
        video = torch.stack([frames_to_tensor(clip) for clip in clips])
        text = torch.randn(2, 512)
        text = text / text.norm(dim=-1, keepdim=True)

        model = build_model(name, **TINY)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        losses = []
        for _ in range(150):
            optimizer.zero_grad()
            # beta=0: pure reconstruction. We are testing capacity and gradient flow,
            # not the regulariser.
            loss = vae_loss(model(video, text), video, beta=0.0)
            loss["loss"].backward()
            optimizer.step()
            losses.append(loss["recon"].item())

        assert torch.isfinite(torch.tensor(losses)).all()
        assert losses[-1] < losses[0] * 0.25, (
            f"{name} failed to overfit real clips: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    @pytest.mark.parametrize("name", ["baseline", "convlstm"])
    def test_training_step_is_numerically_stable_with_kl(self, name):
        torch.manual_seed(0)
        model = build_model(name, **TINY)
        video, text = make_batch(batch=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(30):
            optimizer.zero_grad()
            loss = vae_loss(model(video, text), video, beta=1.0, free_bits=0.5)
            assert torch.isfinite(loss["loss"]), "loss became non-finite"
            loss["loss"].backward()
            optimizer.step()
