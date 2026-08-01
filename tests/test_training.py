"""Checkpointing and training-loop tests.

Resumability is the load-bearing feature here: free Colab sessions disconnect mid-run,
and a checkpoint that silently loses optimizer or RNG state produces a visible
discontinuity in training that is very hard to diagnose after the fact.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from text2video.core.config import Config, merge_overrides
from text2video.core.seed import seed_everything
from text2video.models.cvae import build_model
from text2video.training.checkpoint import (
    CheckpointManager,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from text2video.training.trainer import kl_beta_at

TINY = dict(
    text_dim=512, cond_dim=32, latent_dim=32, feature_dim=64,
    base_ch=32, encoder_base_ch=8, frame_emb_dim=16, num_frames=8,
)


@pytest.fixture
def model_and_optimizer():
    model = build_model("convlstm", **TINY)
    return model, torch.optim.Adam(model.parameters(), lr=1e-3)


class TestCheckpointRoundTrip:
    def test_weights_survive_a_round_trip(self, tmp_path, model_and_optimizer):
        model, optimizer = model_and_optimizer
        save_checkpoint(tmp_path / "c.pt", model, optimizer, global_step=42, epoch=3)

        restored = build_model("convlstm", **TINY)
        payload = load_checkpoint(tmp_path / "c.pt", model=restored)

        assert payload["global_step"] == 42 and payload["epoch"] == 3
        for (name, a), (_, b) in zip(model.state_dict().items(), restored.state_dict().items()):
            assert torch.equal(a, b), f"{name} changed across save/load"

    def test_optimizer_state_is_preserved(self, tmp_path, model_and_optimizer):
        """Resuming with a fresh optimizer resets Adam's moment estimates, which shows
        up as a visible bump in the loss curve right after every reconnect."""
        model, optimizer = model_and_optimizer
        video = torch.rand(2, 8, 1, 64, 64) * 2 - 1
        text = torch.randn(2, 512)
        for _ in range(3):
            optimizer.zero_grad()
            model(video, text)["recon"].mean().backward()
            optimizer.step()

        save_checkpoint(tmp_path / "c.pt", model, optimizer, global_step=3)

        fresh_model = build_model("convlstm", **TINY)
        fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
        load_checkpoint(tmp_path / "c.pt", model=fresh_model, optimizer=fresh_optimizer)

        original_state = optimizer.state_dict()["state"]
        restored_state = fresh_optimizer.state_dict()["state"]
        assert set(original_state) == set(restored_state)
        assert restored_state, "optimizer state was empty -- momentum would be lost"
        for key in original_state:
            assert torch.allclose(
                original_state[key]["exp_avg"], restored_state[key]["exp_avg"]
            )

    def test_rng_state_is_restored(self, tmp_path, model_and_optimizer):
        """Without this, resuming reshuffles data differently and re-randomises latent
        sampling, so a 'resumed' run is not the run it claims to continue."""
        model, optimizer = model_and_optimizer
        seed_everything(7)
        torch.rand(5), np.random.rand(5)  # advance the RNGs
        save_checkpoint(tmp_path / "c.pt", model, optimizer)

        expected_torch = torch.rand(4)
        expected_numpy = np.random.rand(4)

        load_checkpoint(tmp_path / "c.pt", restore_rng=True)
        assert torch.allclose(torch.rand(4), expected_torch)
        assert np.allclose(np.random.rand(4), expected_numpy)

    def test_config_and_history_travel_with_the_checkpoint(self, tmp_path, model_and_optimizer):
        """Evaluation rebuilds the model from the config inside the checkpoint, so it can
        never instantiate a differently-shaped model than the one that was trained."""
        model, optimizer = model_and_optimizer
        config = {"model": {"name": "convlstm", **TINY}, "run_name": "t"}
        history = [{"step": 10, "loss": 0.5}]
        save_checkpoint(tmp_path / "c.pt", model, optimizer, config=config, history=history)

        payload = load_checkpoint(tmp_path / "c.pt")
        assert payload["config"] == config
        assert payload["history"] == history

    def test_save_is_atomic(self, tmp_path, model_and_optimizer):
        """A session killed mid-save must not leave a truncated file that fails to load."""
        model, optimizer = model_and_optimizer
        save_checkpoint(tmp_path / "c.pt", model, optimizer)
        assert (tmp_path / "c.pt").exists()
        assert not list(tmp_path.glob("*.tmp")), "temporary file was left behind"

    def test_mirror_directory_gets_a_copy(self, tmp_path, model_and_optimizer):
        """The Drive mirror is what makes a dropped Colab session survivable."""
        model, optimizer = model_and_optimizer
        mirror = tmp_path / "drive"
        save_checkpoint(tmp_path / "c.pt", model, optimizer, mirror_dir=mirror)
        assert (mirror / "c.pt").exists()

        payload = load_checkpoint(mirror / "c.pt", model=build_model("convlstm", **TINY))
        assert payload is not None


class TestCheckpointManager:
    def test_saves_on_the_step_interval(self, tmp_path):
        manager = CheckpointManager(tmp_path, save_every_steps=100, save_every_minutes=1e6)
        assert not manager.should_save(50)
        assert manager.should_save(100)

    def test_saves_on_the_wall_clock_timer(self, tmp_path):
        """On a free GPU an epoch can outlast the disconnect window, so a step-only
        policy can lose a whole session."""
        manager = CheckpointManager(tmp_path, save_every_steps=10**9, save_every_minutes=0.0)
        assert manager.should_save(1)

    def test_writes_last_and_best(self, tmp_path, model_and_optimizer):
        model, optimizer = model_and_optimizer
        manager = CheckpointManager(tmp_path, run_name="r")
        manager.save(model, optimizer, 100, 1, {}, [], metric=0.5)
        assert (tmp_path / "last.pt").exists()
        assert (tmp_path / "best.pt").exists()

    def test_best_only_updates_on_improvement(self, tmp_path, model_and_optimizer):
        model, optimizer = model_and_optimizer
        manager = CheckpointManager(tmp_path, run_name="r")
        manager.save(model, optimizer, 100, 1, {}, [], metric=0.5)
        manager.save(model, optimizer, 200, 2, {}, [], metric=0.9)  # worse
        assert manager.best_metric == 0.5
        assert load_checkpoint(tmp_path / "best.pt")["global_step"] == 100

        manager.save(model, optimizer, 300, 3, {}, [], metric=0.1)  # better
        assert manager.best_metric == 0.1
        assert load_checkpoint(tmp_path / "best.pt")["global_step"] == 300

    def test_prunes_old_checkpoints_but_keeps_last_and_best(self, tmp_path, model_and_optimizer):
        model, optimizer = model_and_optimizer
        manager = CheckpointManager(tmp_path, run_name="r", keep_last=2)
        for step in (100, 200, 300, 400):
            manager.save(model, optimizer, step, 1, {}, [], metric=1.0 / step)

        step_files = sorted(p.name for p in tmp_path.glob("r_step*.pt"))
        assert len(step_files) == 2, f"pruning failed: {step_files}"
        assert (tmp_path / "last.pt").exists() and (tmp_path / "best.pt").exists()

    def test_find_latest_checkpoint(self, tmp_path, model_and_optimizer):
        model, optimizer = model_and_optimizer
        assert find_latest_checkpoint(tmp_path) is None
        manager = CheckpointManager(tmp_path, run_name="r")
        manager.save(model, optimizer, 100, 1, {}, [])
        assert find_latest_checkpoint(tmp_path, pattern="last.pt").name == "last.pt"

    def test_find_latest_ignores_missing_directory(self, tmp_path):
        assert find_latest_checkpoint(tmp_path / "nope") is None


class TestKLSchedule:
    def test_starts_at_zero_and_reaches_the_target(self):
        """Full KL weight from step 0 causes posterior collapse: before the decoder can
        reconstruct anything, the cheapest loss reduction is to match the prior."""
        assert kl_beta_at(0, 1000, 1e-4) == pytest.approx(0.0)
        assert kl_beta_at(1000, 1000, 1e-4) == pytest.approx(1e-4)

    def test_ramps_linearly(self):
        assert kl_beta_at(500, 1000, 1e-4) == pytest.approx(0.5e-4)

    def test_stays_at_the_target_after_warmup(self):
        assert kl_beta_at(50_000, 1000, 1e-4) == pytest.approx(1e-4)

    def test_zero_warmup_is_immediately_full(self):
        assert kl_beta_at(0, 0, 1e-4) == pytest.approx(1e-4)


class TestConfig:
    def test_attribute_and_dict_access(self):
        cfg = Config({"train": {"lr": 0.001}, "seed": 42})
        assert cfg.train.lr == 0.001
        assert cfg["train"]["lr"] == 0.001
        assert cfg.seed == 42

    def test_to_dict_is_a_deep_copy(self):
        cfg = Config({"train": {"lr": 0.001}})
        data = cfg.to_dict()
        data["train"]["lr"] = 999
        assert cfg.train.lr == 0.001

    def test_dotted_overrides(self):
        cfg = merge_overrides(
            Config({"train": {"lr": 0.001, "steps": 100}}),
            {"train.lr": 0.01, "run_name": "x"},
        )
        assert cfg.train.lr == 0.01
        assert cfg.train.steps == 100  # untouched
        assert cfg.run_name == "x"

    def test_get_with_default(self):
        cfg = Config({"a": 1})
        assert cfg.get("a") == 1
        assert cfg.get("missing", "fallback") == "fallback"
        assert "a" in cfg and "missing" not in cfg


class TestSeeding:
    def test_seeding_makes_runs_reproducible(self):
        seed_everything(123)
        a = (torch.randn(4), np.random.rand(4))
        seed_everything(123)
        b = (torch.randn(4), np.random.rand(4))
        assert torch.allclose(a[0], b[0]) and np.allclose(a[1], b[1])

    def test_returns_the_seed_for_logging(self):
        assert seed_everything(99) == 99
