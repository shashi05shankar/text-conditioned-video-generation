"""Tests for clip generation, captions, and the PyTorch dataset."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from text2video.data.captions import (
    DIRECTION_PHRASES,
    generate_caption,
    parse_caption,
    velocity_to_direction,
)
from text2video.data.dataset import (
    MovingMNISTTextDataset,
    collate_fn,
    frames_to_tensor,
    tensor_to_frames,
)
from text2video.data.moving_mnist import (
    SPEED_RANGES,
    generate_dataset,
    measure_speed_from_frames,
    verify_direction_from_frames,
)


class TestClipGeneration:
    def test_clip_shape_and_dtype(self, generator):
        frames, meta = generator.generate_clip(np.random.default_rng(0))
        assert frames.shape == (16, 64, 64)
        assert frames.dtype == np.uint8
        assert meta["num_digits"] in (1, 2)
        assert len(meta["digits"]) == meta["num_digits"]

    def test_digits_stay_inside_canvas(self, generator):
        """Sprites must never be clipped at the border -- that would corrupt identity."""
        rng = np.random.default_rng(1)
        for _ in range(30):
            frames, _ = generator.generate_clip(rng)
            for frame in frames:
                assert frame.sum() > 0, "frame is empty -- digit left the canvas"

    def test_generation_is_deterministic(self, generator):
        a, meta_a = generator.generate_clip(np.random.default_rng(42))
        b, meta_b = generator.generate_clip(np.random.default_rng(42))
        assert np.array_equal(a, b)
        assert meta_a == meta_b

    def test_rendered_motion_matches_direction_label(self, generator):
        """The single most important data invariant: the video does what the label says.

        Restricted to non-bouncing clips, where the initial direction holds for the
        whole clip and recovery from pixels should be exact.
        """
        rng = np.random.default_rng(3)
        checked = 0
        for _ in range(200):
            frames, meta = generator.generate_clip(rng)
            if meta["num_digits"] != 1 or meta["bounced"][0]:
                continue
            checked += 1
            assert verify_direction_from_frames(frames) == meta["directions"][0]
        assert checked >= 20, f"only {checked} non-bouncing clips sampled -- test too weak"

    def test_measured_speed_matches_speed_bucket(self, generator):
        """Speed buckets must be visually distinguishable, not just labels."""
        rng = np.random.default_rng(5)
        by_bucket: dict[str, list[float]] = {"slow": [], "medium": [], "fast": []}
        for _ in range(300):
            frames, meta = generator.generate_clip(rng)
            if meta["num_digits"] != 1 or meta["bounced"][0]:
                continue
            speed = measure_speed_from_frames(frames)
            if speed is not None:
                by_bucket[meta["speeds"][0]].append(speed)

        means = {k: float(np.mean(v)) for k, v in by_bucket.items() if v}
        assert set(means) == {"slow", "medium", "fast"}
        assert means["slow"] < means["medium"] < means["fast"]

    def test_bounce_flag_matches_wall_list(self, generator):
        rng = np.random.default_rng(11)
        for _ in range(50):
            _, meta = generator.generate_clip(rng)
            for bounced, walls in zip(meta["bounced"], meta["bounce_walls"]):
                assert bounced == bool(walls)
                assert all(w in ("left", "right", "top", "bottom") for w in walls)

    def test_bounce_ratio_is_not_degenerate(self, generator):
        """Guards the bug we hit in development: uniform placement made ~90% of clips
        bounce, so the attribute carried almost no information."""
        rng = np.random.default_rng(13)
        metas = [generator.generate_clip(rng)[1] for _ in range(300)]
        rate = sum(any(m["bounced"]) for m in metas) / len(metas)
        assert 0.25 < rate < 0.85, f"bounce rate {rate:.1%} is degenerate"

    def test_speed_ranges_allow_avoiding_walls(self):
        """Every speed bucket must be slow enough that a no-bounce clip is possible.

        Free space is canvas - sprite = 36px, travelled over num_frames - 1 = 15 steps.
        """
        max_travel = 36 / 15
        for bucket, (_, high) in SPEED_RANGES.items():
            assert high <= max_travel + 1e-9, f"{bucket} max speed {high} forces a bounce"


class TestCaptions:
    def test_caption_vocabulary_is_not_degenerate(self, generator):
        """Many distinct surface strings, so the model must generalise over wording
        rather than memorise one sentence per motion type."""
        frames_captions = generate_dataset(generator, num_clips=400, seed=17)[1]
        unique_ratio = len(set(frames_captions)) / len(frames_captions)
        assert unique_ratio > 0.5, f"only {unique_ratio:.1%} unique captions"

        vocabulary = {w for c in frames_captions for w in c.lower().split()}
        assert len(vocabulary) > 40, f"vocabulary of {len(vocabulary)} words is too small"

    def test_caption_mentions_the_correct_digit(self, generator):
        rng = np.random.default_rng(19)
        caption_rng = random.Random(19)
        for _ in range(60):
            _, meta = generator.generate_clip(rng)
            caption = generate_caption(meta, caption_rng)
            for digit in meta["digits"]:
                assert str(digit) in caption

    def test_bounce_is_only_mentioned_when_it_happens(self, generator):
        """A caption must never assert something the video does not show."""
        rng = np.random.default_rng(23)
        caption_rng = random.Random(23)
        for _ in range(200):
            _, meta = generator.generate_clip(rng)
            if meta["num_digits"] != 1:
                continue
            caption = generate_caption(meta, caption_rng)
            if not meta["bounced"][0]:
                assert "bounc" not in caption.lower()
                assert "hits the" not in caption.lower()

    def test_named_bounce_wall_was_really_hit(self, generator):
        rng = np.random.default_rng(29)
        caption_rng = random.Random(29)
        for _ in range(200):
            _, meta = generator.generate_clip(rng)
            if meta["num_digits"] != 1 or not meta["bounced"][0]:
                continue
            caption = generate_caption(meta, caption_rng).lower()
            named = [w for w in ("left", "right", "top", "bottom") if f"the {w} " in caption]
            for wall in named:
                if wall in ("left", "right", "top", "bottom") and (
                    f"the {wall} edge" in caption or f"the {wall} wall" in caption
                ):
                    assert wall in meta["bounce_walls"][0]

    def test_velocity_to_direction_cardinal_and_diagonal(self):
        assert velocity_to_direction(1.0, 0.0) == "right"
        assert velocity_to_direction(-1.0, 0.0) == "left"
        assert velocity_to_direction(0.0, -1.0) == "up"      # +y is down in image coords
        assert velocity_to_direction(0.0, 1.0) == "down"
        assert velocity_to_direction(1.0, -1.0) == "up_right"
        assert velocity_to_direction(-1.0, 1.0) == "down_left"

    def test_parse_caption_recovers_attributes(self, generator):
        """Round-trip: attributes -> caption -> attributes."""
        rng = np.random.default_rng(31)
        caption_rng = random.Random(31)
        matched = 0
        checked = 0
        for _ in range(150):
            _, meta = generator.generate_clip(rng)
            if meta["num_digits"] != 1:
                continue
            checked += 1
            parsed = parse_caption(generate_caption(meta, caption_rng))
            assert meta["digits"][0] in parsed["digits"]
            if parsed["direction"] == meta["directions"][0]:
                matched += 1
        assert checked > 30
        assert matched / checked > 0.95, "direction parsing is unreliable"

    def test_direction_phrases_are_unambiguous(self):
        """No direction phrase may be a substring of a different direction's phrase,
        or parsing would silently pick the wrong one."""
        for direction, phrases in DIRECTION_PHRASES.items():
            for phrase in phrases:
                for other, other_phrases in DIRECTION_PHRASES.items():
                    if other == direction:
                        continue
                    for other_phrase in other_phrases:
                        if phrase in other_phrase:
                            # Longest-match-first parsing must resolve this correctly.
                            assert parse_caption(f"the digit 1 moves {other_phrase}.")[
                                "direction"
                            ] == other


class TestDataset:
    def test_frames_tensor_roundtrip(self, generator):
        frames, _ = generator.generate_clip(np.random.default_rng(37))
        tensor = frames_to_tensor(frames)
        assert tensor.shape == (16, 1, 64, 64)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= -1.0 and tensor.max() <= 1.0
        # uint8 -> [-1,1] -> uint8 must be lossless
        assert np.array_equal(tensor_to_frames(tensor), frames)

    def test_dataset_getitem(self, tiny_dataset_dir):
        dataset = MovingMNISTTextDataset(tiny_dataset_dir)
        assert len(dataset) == 12
        item = dataset[0]
        assert item["frames"].shape == (16, 1, 64, 64)
        assert isinstance(item["caption"], str) and item["caption"]
        assert "digits" in item["meta"]

    def test_dataset_rejects_mismatched_embeddings(self, tiny_dataset_dir):
        with pytest.raises(ValueError, match="text embeddings"):
            MovingMNISTTextDataset(
                tiny_dataset_dir, text_embeddings=np.zeros((5, 512), dtype=np.float32)
            )

    def test_dataset_missing_split_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build_dataset"):
            MovingMNISTTextDataset(tmp_path / "nope")

    def test_collate_batches_tensors_and_passes_through_metadata(self, tiny_dataset_dir):
        dataset = MovingMNISTTextDataset(
            tiny_dataset_dir, text_embeddings=np.zeros((12, 512), dtype=np.float32)
        )
        batch = collate_fn([dataset[i] for i in range(4)])
        assert batch["frames"].shape == (4, 16, 1, 64, 64)
        assert batch["text_emb"].shape == (4, 512)
        assert len(batch["caption"]) == 4 and isinstance(batch["caption"][0], str)
        assert len(batch["meta"]) == 4 and isinstance(batch["meta"][0], dict)

    def test_dataloader_end_to_end(self, tiny_dataset_dir):
        from torch.utils.data import DataLoader

        dataset = MovingMNISTTextDataset(tiny_dataset_dir)
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=0)
        batch = next(iter(loader))
        assert batch["frames"].shape == (4, 16, 1, 64, 64)
