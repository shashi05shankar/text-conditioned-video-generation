"""Streamlit demo: type a sentence, get a generated video.

    streamlit run app/demo.py

The UI is deliberately thin. The point of this project is the trained model and its
evaluation; the demo exists so a human can interact with the result, and so the honest
per-clip scores (does it show the right digit, moving the right way?) are visible rather
than buried in a metrics file.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from text2video.inference.generator import VideoGenerator  # noqa: E402
from text2video.utils.video import filmstrip, save_gif  # noqa: E402

EXAMPLE_PROMPTS = [
    "The digit 3 moves to the right.",
    "The digit 7 moves to the left.",
    "A handwritten 1 travels upward.",
    "The number 5 moves downward.",
    "The digit 8 glides diagonally up and to the right.",
    "A 2 moves quickly towards the bottom left corner.",
    "The digit 6 moves rightward and bounces off the right edge.",
]

st.set_page_config(page_title="Text to Video", page_icon="film", layout="wide")


def find_checkpoints() -> dict[str, Path]:
    """Trained checkpoints, newest-looking first. Smoke runs are excluded."""
    found: dict[str, Path] = {}
    runs_dir = PROJECT_ROOT / "outputs" / "runs"
    if not runs_dir.exists():
        return found
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir() or run.name.endswith("_smoke"):
            continue
        for name in ("best.pt", "last.pt"):
            candidate = run / "checkpoints" / name
            if candidate.exists():
                found[run.name] = candidate
                break
    return found


@st.cache_resource(show_spinner="Loading model and CLIP text encoder...")
def load_generator(checkpoint: str, device: str) -> VideoGenerator:
    classifier = PROJECT_ROOT / "outputs" / "digit_classifier.pt"
    generator = VideoGenerator(checkpoint, device=device, digit_classifier_path=classifier)
    generator.text_encoder  # warm the lazy CLIP load inside the cached call
    return generator


def render_scorecard(score: dict) -> None:
    """Show what was asked for beside what was actually produced."""
    columns = st.columns(3)

    with columns[0]:
        requested = score["requested_digits"][0] if score["requested_digits"] else None
        observed = score["observed_digit"]
        if observed is None:
            st.metric("Digit", "not scored")
        else:
            st.metric(
                "Digit",
                f"{observed}",
                delta="matches" if score["digit_matches"] else f"asked for {requested}",
                delta_color="normal" if score["digit_matches"] else "inverse",
            )
            if score["digit_confidence"] is not None:
                st.caption(f"classifier confidence {score['digit_confidence']:.0%}")

    with columns[1]:
        observed = score["observed_direction"]
        requested = score["requested_direction"]
        st.metric(
            "Direction",
            observed.replace("_", "-") if observed else "no motion",
            delta="matches" if score["direction_matches"]
            else (f"asked for {requested}" if requested else "not specified"),
            delta_color="normal" if score["direction_matches"] else "inverse",
        )

    with columns[2]:
        speed = score["observed_speed_px_per_frame"]
        st.metric(
            "Speed",
            f"{speed:.2f} px/frame" if speed is not None else "0",
            delta="STATIC - no motion generated" if score["is_static"] else "moving",
            delta_color="inverse" if score["is_static"] else "normal",
        )
        if score["requested_speed"]:
            st.caption(f"asked for: {score['requested_speed']}")


def main() -> None:
    st.title("Text-Conditioned Short Video Generation")
    st.caption(
        "A conditional VAE with a ConvLSTM temporal decoder, trained from scratch on "
        "Bouncing MNIST. Frozen CLIP encodes the caption; the model generates 16 frames "
        "at 64x64."
    )

    checkpoints = find_checkpoints()
    if not checkpoints:
        st.error(
            "No trained checkpoints found under `outputs/runs/`.\n\n"
            "Train a model first:\n"
            "```\npython scripts/train.py --config configs/train_convlstm.yaml\n```"
        )
        st.stop()

    with st.sidebar:
        st.header("Model")
        choice = st.selectbox("Checkpoint", list(checkpoints), index=len(checkpoints) - 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.caption(f"device: {device}")

        st.header("Sampling")
        num_samples = st.slider("Samples per prompt", 1, 4, 2,
                                help="The model is a VAE, so one caption can produce "
                                     "several different valid clips.")
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1,
                                help="Scales the prior's standard deviation. Lower is "
                                     "more typical, higher is more varied.")
        prior_mean = st.checkbox("Deterministic (prior mean)", value=False)
        seed = st.number_input("Seed", value=0, step=1)

    generator = load_generator(str(checkpoints[choice]), device)
    info = generator.describe()

    with st.sidebar:
        st.header("This model")
        st.write(f"**variant:** {info['variant']}")
        st.write(f"**params:** {info['total_params']/1e6:.2f}M")
        st.write(f"**trained steps:** {info['train_step']}")

    prompt = st.text_input(
        "Describe a video",
        value=EXAMPLE_PROMPTS[0],
        help="Mention a digit (0-9), a direction, and optionally a speed or a bounce.",
    )
    st.caption("Try: " + "  |  ".join(f"`{p}`" for p in EXAMPLE_PROMPTS[1:5]))

    if not st.button("Generate", type="primary"):
        st.info("Enter a prompt and press Generate.")
        return

    started = time.time()
    videos = generator.generate(
        prompt,
        num_samples=int(num_samples),
        temperature=float(temperature),
        use_prior_mean=bool(prior_mean),
        seed=int(seed),
    )
    elapsed = time.time() - started
    st.success(f"Generated {len(videos)} clip(s) in {elapsed:.2f}s")

    gif_dir = PROJECT_ROOT / "outputs" / "demo"
    gif_dir.mkdir(parents=True, exist_ok=True)

    for i, video in enumerate(videos):
        st.divider()
        st.subheader(f"Sample {i + 1}")
        left, right = st.columns([1, 2])

        with left:
            gif_path = save_gif(video, gif_dir / f"demo_{i}.gif", fps=8)
            st.image(str(gif_path), caption="animated (8 fps)", width=200)

        with right:
            strip = filmstrip(video, every=2)
            st.image(
                strip.astype(np.uint8),
                caption="every 2nd frame, left to right - motion is the digit shifting across",
                use_container_width=True,
            )

        render_scorecard(generator.score(video, prompt))

    with st.expander("How the scores are computed"):
        st.markdown(
            """
- **Digit** - an independently trained MNIST CNN (98.7% held-out accuracy) classifies
  the generated frames. It shares no weights with the generator and never saw generated
  data during training.
- **Direction** - measured from the intensity centroid's displacement over the first
  frames, then bucketed into 8 compass directions.
- **Speed** - mean per-frame centroid displacement, in pixels.

None of these use CLIP. That is deliberate: CLIP conditions the generator, so scoring
with CLIP would not be an independent check. CLIP text embeddings for "moves left" and
"moves right" are ~0.98 cosine-similar, so CLIP-based similarity barely distinguishes
them anyway.
            """
        )


if __name__ == "__main__":
    main()
