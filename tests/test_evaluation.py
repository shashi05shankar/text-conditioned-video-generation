"""Evaluation-metric tests.

These matter more than usual: a broken metric produces plausible-looking numbers that
silently misrank the models, which is worse than a crash. Each test pins a property we
rely on when interpreting the comparison table.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from text2video.data.dataset import frames_to_tensor
from text2video.evaluation.clipsim import cosine_similarity, opposite_direction_caption
from text2video.evaluation.digit_classifier import DigitCNN, classify_frames, crop_digit
from text2video.evaluation.fid import frechet_distance
from text2video.evaluation.grounding import (
    STATIC_SPEED_THRESHOLD,
    direction_grounding,
    speed_grounding,
)
from text2video.evaluation.report import build_markdown_table, build_report
from text2video.evaluation.temporal import (
    centroid_motion,
    frame_ssim,
    motion_magnitude,
    temporal_metrics,
)


@pytest.fixture
def real_clips(generator):
    """A few genuine Bouncing MNIST clips as (N, T, 1, 64, 64) in [-1, 1]."""
    rng = np.random.default_rng(0)
    clips = [generator.generate_clip(rng) for _ in range(6)]
    videos = torch.stack([frames_to_tensor(frames) for frames, _ in clips])
    return videos, [meta for _, meta in clips]


def freeze(videos: torch.Tensor) -> torch.Tensor:
    """Turn clips into static ones by repeating frame 0."""
    return videos[:, :1].repeat(1, videos.shape[1], 1, 1, 1)


class TestFID:
    def test_identical_distributions_score_near_zero(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(256, 32))
        assert frechet_distance(features, features) == pytest.approx(0.0, abs=1e-4)

    def test_distance_grows_as_distributions_separate(self):
        rng = np.random.default_rng(1)
        base = rng.normal(size=(256, 16))
        near = frechet_distance(base, base + 0.1)
        far = frechet_distance(base, base + 2.0)
        assert 0 < near < far

    def test_is_symmetric(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(size=(128, 8)), rng.normal(size=(128, 8)) + 0.5
        assert frechet_distance(a, b) == pytest.approx(frechet_distance(b, a), rel=1e-6)

    def test_detects_collapsed_diversity(self):
        """A mode-collapsed generator has the right mean but no spread. FID must
        penalise that -- it is the failure a per-image metric would miss."""
        rng = np.random.default_rng(3)
        real = rng.normal(size=(512, 16))
        collapsed = np.repeat(real.mean(axis=0, keepdims=True), 512, axis=0)
        collapsed += rng.normal(scale=1e-3, size=collapsed.shape)
        assert frechet_distance(real, collapsed) > 1.0


class TestTemporalMetrics:
    def test_static_video_scores_perfect_ssim(self, real_clips):
        """The trap this project is built to avoid: a frozen clip is 'perfectly
        consistent'. If this ever stops being true, the static control stops being a
        meaningful demonstration."""
        videos, _ = real_clips
        assert frame_ssim(freeze(videos))["frame_ssim"] == pytest.approx(1.0, abs=1e-6)

    def test_static_video_has_no_motion(self, real_clips):
        videos, _ = real_clips
        frozen = freeze(videos)
        assert motion_magnitude(frozen)["motion_magnitude"] == pytest.approx(0.0, abs=1e-6)
        assert centroid_motion(frozen)["centroid_speed"] == pytest.approx(0.0, abs=1e-6)
        assert motion_magnitude(frozen)["static_clip_fraction"] == 1.0

    def test_real_clips_have_measurable_motion(self, real_clips):
        videos, _ = real_clips
        assert centroid_motion(videos)["centroid_speed"] > 0.5

    def test_temporal_score_rejects_the_static_control(self, real_clips):
        """The combined score must be 0 for a static clip despite its perfect SSIM --
        this is the whole reason the combined score exists."""
        videos, _ = real_clips
        metrics = temporal_metrics(freeze(videos), reference_videos=videos)
        assert metrics["frame_ssim"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["temporal_score"] == pytest.approx(0.0, abs=1e-6)

    def test_real_data_scores_well_against_itself(self, real_clips):
        videos, _ = real_clips
        metrics = temporal_metrics(videos, reference_videos=videos)
        assert metrics["motion_relative_error"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["temporal_score"] == pytest.approx(metrics["frame_ssim"])

    def test_score_omitted_without_a_reference(self, real_clips):
        videos, _ = real_clips
        assert "temporal_score" not in temporal_metrics(videos)

    def test_noise_flickers_more_than_real_video(self, real_clips):
        videos, _ = real_clips
        noise = torch.rand_like(videos) * 2 - 1
        assert frame_ssim(noise)["frame_ssim"] < frame_ssim(videos)["frame_ssim"]


class TestGrounding:
    def test_direction_is_recovered_from_real_clips(self, real_clips):
        """Ground-truth data should score near the ~95% ceiling, not near chance."""
        videos, metas = real_clips
        single = [i for i, m in enumerate(metas) if m["num_digits"] == 1]
        if not single:
            pytest.skip("no single-digit clips in this sample")
        subset = videos[single]
        expected = [metas[i]["directions"][0] for i in single]
        assert direction_grounding(subset, expected)["direction_accuracy"] >= 0.7

    def test_static_clips_score_zero_direction(self, real_clips):
        videos, metas = real_clips
        expected = [m["directions"][0] for m in metas]
        result = direction_grounding(freeze(videos), expected)
        assert result["direction_accuracy"] == 0.0
        assert result["direction_undetectable_fraction"] == 1.0

    def test_static_clips_score_zero_speed(self, real_clips):
        """Regression guard: 'did not move' must not count as 'moved slowly'.

        Before the STATIC_SPEED_THRESHOLD fix, a frozen clip scored ~35% on speed
        because zero displacement fell inside the slow bucket.
        """
        videos, metas = real_clips
        expected = [m["speeds"][0] for m in metas]
        result = speed_grounding(freeze(videos), expected)
        assert result["speed_accuracy"] == 0.0
        assert result["speed_static_fraction"] == 1.0

    def test_static_threshold_is_below_the_slowest_real_speed(self):
        """The threshold must not swallow genuinely slow clips (slowest is 0.8 px/frame)."""
        assert 0.0 < STATIC_SPEED_THRESHOLD < 0.8


class TestDigitClassifier:
    def test_crop_centres_on_the_digit(self):
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[40:52, 10:22] = 255  # a block away from centre
        crop = crop_digit(frame, size=28)
        assert crop.shape == (28, 28)
        assert crop.sum() == frame.sum(), "cropping lost part of the digit"

    def test_crop_handles_an_empty_frame(self):
        crop = crop_digit(np.zeros((64, 64), dtype=np.uint8), size=28)
        assert crop.shape == (28, 28) and crop.sum() == 0

    def test_crop_handles_a_digit_at_the_border(self):
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[0:10, 0:10] = 255
        assert crop_digit(frame, size=28).shape == (28, 28)

    def test_classify_frames_output_shapes(self):
        model = DigitCNN()
        frames = np.random.randint(0, 255, size=(4, 64, 64), dtype=np.uint8)
        predictions, confidences = classify_frames(model, frames)
        assert predictions.shape == (4,) and confidences.shape == (4,)
        assert set(np.unique(predictions)).issubset(set(range(10)))
        assert ((confidences >= 0) & (confidences <= 1)).all()


class TestCLIPSim:
    def test_cosine_similarity_bounds(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0)
        assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "caption,expected_missing",
        [
            ("The digit 3 moves to the right.", "to the right"),
            ("A 5 glides rightward.", "rightward"),
            ("The number 7 travels upward.", "upward"),
        ],
    )
    def test_opposite_direction_flips_the_phrase(self, caption, expected_missing):
        flipped = opposite_direction_caption(caption)
        assert flipped != caption
        assert expected_missing not in flipped

    def test_opposite_direction_leaves_unknown_captions_alone(self):
        caption = "Two digits move at the same time."
        assert opposite_direction_caption(caption) == caption


class TestReport:
    def _records(self):
        return [
            {"variant": "static_control", "frame_ssim": 1.0, "temporal_score": 0.0,
             "grounding_score": 0.30},
            {"variant": "baseline", "total_params": 4_270_273, "fid": 40.0,
             "grounding_score": 0.50, "temporal_score": 0.40, "frame_ssim": 0.95},
            {"variant": "convlstm", "total_params": 6_503_105, "fid": 25.0,
             "grounding_score": 0.70, "temporal_score": 0.65, "frame_ssim": 0.90},
            {"variant": "real_data_ceiling", "grounding_score": 0.94,
             "temporal_score": 0.89, "frame_ssim": 0.89},
        ]

    def test_table_contains_every_variant(self):
        table = build_markdown_table(self._records())
        for name in ("Baseline", "ConvLSTM", "Real data", "Static control"):
            assert name in table

    def test_best_model_value_is_bolded(self):
        table = build_markdown_table(self._records())
        assert "**70.0%**" in table, "best grounding score should be highlighted"

    def test_reference_rows_do_not_win(self):
        """The ceiling row scores highest but is a reference, not a competitor -- bolding
        it would imply a model was beaten by the data itself."""
        table = build_markdown_table(self._records())
        assert "**94.0%**" not in table

    def test_missing_metrics_render_as_na_not_a_number(self):
        table = build_markdown_table(
            [{"variant": "baseline", "grounding_score": 0.5}, {"variant": "convlstm"}]
        )
        assert "n/a" in table

    def test_report_includes_the_delta_section(self):
        report = build_report(self._records())
        assert "Does temporal modelling help?" in report
        assert "better" in report or "worse" in report

    def test_report_flags_the_parameter_gap(self):
        report = build_report(self._records())
        assert "capacity-matched" in report
