# Text-Conditioned Short Video Generation

Generate a short video clip from a natural-language sentence, using a lightweight deep-learning model trained from scratch on a single free-tier GPU.

---

## 1. Business Problem

Short-form video is the dominant format in marketing, education, and product communication, but producing even a few seconds of custom video is slow and expensive. Commercial text-to-video systems solve this at high quality, but they depend on massive datasets and industrial-scale GPU clusters.

The practical question this project answers is the one an engineer would actually be asked: **how much of that capability can be reproduced with a small model, a small dataset, and a single free GPU — and how do you prove it works?**

---

## 2. Problem Statement

Given a natural-language caption, generate a short video (16 frames, 64×64) whose content matches the caption and whose frames are temporally consistent — the same subject persisting and moving coherently across time, rather than each frame being generated independently.

This requires three capabilities working together:

| Capability | What it means here |
|---|---|
| **NLP** | Convert a caption into a numeric embedding the network can condition on |
| **Computer vision / generative modeling** | Turn that embedding (plus sampled noise) into image frames |
| **Temporal modeling** | Make frame *t+1* a coherent continuation of frame *t* |

---

## 3. Objective

Build **one strong working text-conditioned video generation model**, demonstrate measurably that it beats a simple baseline, and show that the improvement comes specifically from temporal modeling.

Concretely:

1. Build a baseline that generates frames **independently** (no temporal module).
2. Measure its weakness quantitatively — flickering, identity drift, poor motion.
3. Build the main model: a **conditional VAE with a ConvLSTM temporal decoder**.
4. Evaluate both on identical held-out prompts and report the real improvement.
5. Ship a small demo app: type a sentence → get a generated clip.

**Success = a working model, a real measured improvement over the baseline, and a demo.** Not a survey of architectures.

---

## 4. Dataset

**Moving MNIST with generated captions** — built by us, not downloaded.

MNIST digits move across a 64×64 canvas with known velocity and wall bouncing. Each clip is 16 frames. Captions are generated from the clip's exact motion parameters:

> *"Digit 3 moves slowly to the right."*
> *"The number 7 drifts diagonally upward and bounces off the wall."*
> *"Two digits travel in opposite directions."*

Captions vary in phrasing (multiple templates × synonym banks for verbs, speeds, directions) so the model must generalize over wording rather than memorize fixed strings.

**Why this dataset:** because we generate it, we know the ground-truth digit, direction, and speed for every clip. That makes it possible to check whether a generated video *actually did what the caption asked* — an objective semantic check, not just "does it look plausible." Most portfolio projects can't do this. It is also small, license-free, and trains fast on a free GPU.

Train and test use **disjoint MNIST digit images**, so evaluation tests generalization rather than memorization.

---

## 5. Baseline

**Text → embedding → decoder → 16 frames, each decoded independently.**

No recurrence, no temporal state, no awareness of neighbouring frames. It will produce recognizable digits but is expected to flicker and drift in identity across the clip.

Its purpose is to be the measurable floor. Without it, "temporal modeling helps" is a claim; with it, it's a number.

---

## 6. Proposed Architecture (main model)

**Conditional VAE + ConvLSTM temporal decoder**

```
Caption
  └─> Frozen CLIP text encoder ──> text embedding (512-d)
                                        │
                Video ──> CNN encoder ──┴──> latent distribution (μ, σ)
                                        │         │ sample z
                                        ▼         ▼
                              ConvLSTM decoder (steps t = 1…16)
                                  carries hidden state across frames
                                        │
                                        ▼
                              16 × 64×64 generated frames
```

**Why each piece:**

- **Frozen CLIP text encoder** — borrows a strong pretrained language/vision-aligned representation. We don't train it: text understanding isn't the variable under test, and freezing it saves memory and training time. Embeddings are precomputed and cached so the text encoder never runs inside the training loop.
- **Conditional VAE** — learns a *distribution* over videos rather than one fixed output per caption, so generation can be sampled and varied. The KL term regularizes the latent space so sampled points decode to valid videos.
- **ConvLSTM decoder** — an LSTM whose gates are convolutions, so its memory stays a 2D feature map instead of a flattened vector. This preserves spatial structure while carrying motion information across timesteps. **This is the component that makes the video a video and not 16 unrelated images**, and it is the specific change credited with the improvement over baseline.

Target size: well under 15M parameters.

---

## 7. Evaluation Metrics

Both models are scored on the **same fixed held-out prompt set**, never seen during training.

| What it measures | Metric |
|---|---|
| **Visual quality** | FID (frame-level) |
| **Text-video alignment** | CLIPSIM — cosine similarity between the caption embedding and per-frame CLIP image embeddings |
| **Temporal consistency** | Consecutive-frame SSIM **paired with** optical-flow motion magnitude |
| **Semantic correctness** | Structured grounding — does the generated clip show the right digit, moving in the right direction, at roughly the right speed? |
| **Efficiency** | Parameter count, training time, inference time per clip |

Two notes that matter:

- **SSIM is never reported alone.** A completely frozen video scores near-perfectly on frame similarity, so a still image would look like the "most temporally consistent" model. It is always paired with a motion measurement.
- **CLIPSIM is not fully independent**, because the same CLIP model is used to condition the model and to score it. That's why the structured grounding check exists as an independent cross-check.

**Every reported number comes from an actual run logged to `experiments/`. Metrics that weren't run are reported as "not run" — never estimated.**

### Final comparison table

| Model | FID ↓ | CLIPSIM ↑ | Temporal (SSIM + motion) | Grounding ↑ | Params | Train time | Inference |
|---|---|---|---|---|---|---|---|
| Baseline (no temporal) | | | | | | | |
| cVAE + ConvLSTM | | | | | | | |

---

## 8. Constraints

- Free-tier Colab/Kaggle, single GPU, ~12–16GB VRAM; **sessions can disconnect**, so checkpoints must be resumable.
- 64×64 resolution, 16-frame clips, grayscale.
- Model under ~15M parameters; a training run must fit in a few hours.
- No model gets multi-hour GPU time until it passes a **tiny-overfit test** — if it can't overfit 4 clips on CPU, it gets debugged, not trained.

---

## 9. Tech Stack

| Layer | Choice |
|---|---|
| Framework | PyTorch |
| Text encoder | CLIP ViT-B/32 (frozen, via `open_clip`) |
| Data | NumPy/PyTorch Moving-MNIST generator + caption generator |
| Metrics | `pytorch-fid`, `scikit-image` (SSIM), OpenCV (optical flow), CLIP (CLIPSIM) |
| Demo | Streamlit |
| Training env | Google Colab / Kaggle notebooks |
| Testing | pytest (CPU-only) |
| Tracking | JSON/CSV run logs (W&B optional, never required) |

---

## 10. Implementation Phases

Each phase is explained, implemented, tested, and reviewed before the next begins.

| Phase | What gets built | Done when |
|---|---|---|
| **1. Data pipeline** | Moving MNIST generator, caption generator, PyTorch Dataset, sample visualizations | Clips look correct; captions match their motion metadata; caption vocabulary is varied |
| **2. Text encoder** | Frozen CLIP wrapper + embedding cache | Shapes/dtypes verified; embeddings reproducible; similar captions embed closer than dissimilar ones |
| **3. Baseline model** | Independent-frame decoder + trainer + first real training run | Produces real (weak) metrics — the floor to beat |
| **4. Main model** | cVAE + ConvLSTM, trainer, tiny-overfit proof, full training run | Passes tiny-overfit gate, then trains to convergence |
| **5. Evaluation** | All five metric families, comparison table, sample galleries, failure-case analysis | Real numbers for both models; improvement identified and explained |
| **6. Demo + docs** | Streamlit app (prompt → GIF + scores), README with results | Demo runs end-to-end from a trained checkpoint |

**Optional extensions (only after the above is complete):** conditional GAN variant, lightweight diffusion variant, real-world captioned dataset (e.g. a small MSR-VTT subset).

---

## 11. Final Deliverable

A GitHub repository containing:

- A trained text-conditioned video generation model that works
- A baseline it measurably beats, with the comparison table and the reason for the gap
- Full training + evaluation code, reproducible from config and seed
- Generated sample clips (successes **and** documented failure cases)
- A Streamlit demo: sentence in → generated video out, with alignment and temporal scores
- Colab/Kaggle notebooks and a README with the real results

**How this is described on a resume:**

> Built a text-conditioned short-video generation system in PyTorch: frozen CLIP text embeddings conditioning a convolutional VAE with a ConvLSTM temporal decoder, trained from scratch on a single free-tier GPU. Quantified the contribution of temporal modeling against a frame-independent baseline using FID, CLIP similarity, optical-flow-based temporal consistency, and a custom structured-grounding metric; deployed as a Streamlit demo.

No claim of novelty, no claim of competing with commercial text-to-video systems.
