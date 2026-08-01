# Build Log

A running record of architecture decisions, tensor shapes, hyperparameters, commands,
errors and fixes, and results. Written so the whole project can be studied after the
fact — including the things that went wrong.

---

## Environment

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.13.5 (venv at `.venv`) — 3.14 was the system default but PyTorch support is behind, so 3.13 was used |
| PyTorch | 2.13.0+cpu (local); GPU build comes from Colab/Kaggle |
| Key libs | numpy 2.4.4, scikit-image 0.26.0, opencv 5.0.0, open_clip 3.3.0 |

Local machine has **no CUDA GPU**, so all local work is CPU-only: shape tests, gradient
checks, and tiny-overfit runs. Real training happens on Colab/Kaggle.

```bash
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv/Scripts/python.exe -m pip install pytest pyyaml scikit-image opencv-python-headless matplotlib imageio open_clip_torch
.venv/Scripts/python.exe -m pip install -e .
```

---

## Phase 1 — Data pipeline

### What it is

Bouncing MNIST: 1–2 MNIST digit sprites moving in straight lines on a 64×64 canvas,
reflecting elastically off walls, rendered to 16 frames. Each clip carries exact
ground-truth metadata, and a caption generated *from* that metadata.

### Why generate rather than download

The metadata is the point. Because we know the true digit, direction, speed and bounce
events for every clip, we can later check whether a **generated** video actually did
what its caption asked — the structured-grounding metric. A downloaded video dataset
gives you captions but no verifiable motion ground truth.

### Tensor shapes

| Object | Shape | dtype | Range |
|---|---|---|---|
| Stored clip | `(T=16, H=64, W=64)` | uint8 | 0–255 |
| Dataset item `frames` | `(16, 1, 64, 64)` | float32 | [−1, 1] |
| Batch `frames` | `(B, 16, 1, 64, 64)` | float32 | [−1, 1] |
| Batch `text_emb` | `(B, 512)` | float32 | CLIP ViT-B/32 |

Normalisation is `x/127.5 − 1`, chosen so a `tanh` output layer covers the full range
symmetrically. `tensor_to_frames` is the exact inverse (verified lossless by test).

### Key design decisions

**1. Directions snapped to 8 exact compass headings.**
Sampling a continuous angle and bucketing it into 8 labels would mean a digit
travelling at 22° gets labelled "right" while visibly drifting upward — fuzzy ground
truth makes the grounding metric unreliable. Snapping costs a little task difficulty
and buys exactly-correct labels.

**2. Max blending when compositing two digits**, not addition. Addition saturates
overlapping digits into a white blob and destroys both identities.

**3. Sprite pools are disjoint across splits.** Train clips use MNIST *train* images;
val/test clips use MNIST *test* images. Held-out evaluation therefore tests
generalisation to unseen digit renderings, not pixel memorisation.

**4. Caption generation is template × synonym bank.**
6 single-digit templates, 4 bounce templates, 7 pair templates; 5 verbs × 3 speed
phrasings × 3 direction phrasings per direction. Result: **75.6% unique captions**
across 20k train clips, ~96% on the 2k val/test splits. If every "moves right" clip
carried an identical string, the model could memorise a handful of strings instead of
learning language.

**5. Captions never assert what the video does not show.** A bounce is only mentioned
when one actually occurred, and the named wall must be one that was really hit.

### Errors and fixes

**Problem 1 — 90% of clips bounced.**
First version placed digits uniformly at random. Over 16 frames most digits travel
further than the 36px of free space, so nearly every clip bounced. "Bounces off the
wall" became a near-constant phrase carrying almost no information.

*Fix:* solve for the valid start-position range analytically per axis
(`_choose_start_position`) and sample inside it, with `bounce_prob=0.5`. For a bounce,
force it on a single moving axis so diagonal digits hit one named wall rather than
always cornering.

**Problem 2 — still 82% bouncing after the fix.**
The generator was *asking* for no-bounce and being overruled by geometry: at speeds up
to 4.0 px/frame a digit travels 60px in 16 frames, far more than the 36px available,
so no-bounce was impossible. Worse, this confounded two attributes — "fast" implied
"bounces".

*Fix:* cap speeds at 36/15 = 2.4 px/frame so no-bounce is achievable in every bucket.
New ranges: slow 0.8–1.4, medium 1.4–1.9, fast 1.9–2.4. Bounce rate fell to 68%, and
bounce rate by speed bucket tightened from a wide spread to 51.9% / 56.4% / 70.7%.
Fast is still 3× slow (36px vs 12px of travel) — clearly visible.

**Problem 3 — direction verification only 90.7% accurate on ground-truth data.**
`verify_direction_from_frames` originally measured net start-to-end centroid
displacement. On a bouncing clip the digit reverses partway through, so net
displacement can point the wrong way — while the caption ("moves right and bounces off
the right edge") is still correct.

*Fix:* measure displacement over the first `window` frames only, since the labelled
direction is the *initial* one. Swept the window on the real val split:

| window | overall | bouncing clips | non-bouncing |
|---|---|---|---|
| 2 | 90.6% | 90.4% | 90.9% |
| **3** | **95.7%** | **92.8%** | **100.0%** |
| 4 | 93.5% | 89.2% | 100.0% |
| 5 | 90.7% | 84.6% | 100.0% |
| 8 | 83.4% | 72.5% | 100.0% |

window=3 chosen. window=2 is noisier because integer-pixel rendering quantises a slow
digit's ~1px-per-frame motion. **95.7% is the metric's ceiling on real data** —
generated-video grounding scores must be read against that, not against 100%.

### Dataset as built

```bash
python scripts/build_dataset.py --config configs/dataset.yaml
```

| Split | Clips | Sprites from | Unique captions | Bounced | Two-digit | Size |
|---|---|---|---|---|---|---|
| train | 20,000 | MNIST train | 15,130 (75.6%) | 13,621 | 5,990 | 1250 MB |
| val | 2,000 | MNIST test | 1,925 (96.2%) | 1,351 | 620 | 125 MB |
| test | 2,000 | MNIST test | 1,931 (96.5%) | 1,398 | 610 | 125 MB |

Mean caption length 12.7 words.

Measured speed by bucket (val, non-bouncing single-digit clips):
slow **1.14**, medium **1.65**, fast **2.18** px/frame — cleanly separated.

> The dataset is **not** committed to git (1.5 GB). It is fully deterministic given the
> seeds in `configs/dataset.yaml`, so Colab regenerates byte-identical data in about a
> minute. Only code needs to be uploaded.

### Verification

21 tests, all passing, CPU-only, no network:

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

The load-bearing ones:
- `test_rendered_motion_matches_direction_label` — the video really does what the label says
- `test_measured_speed_matches_speed_bucket` — slow < medium < fast, measured from pixels
- `test_bounce_is_only_mentioned_when_it_happens` — captions are never false
- `test_caption_vocabulary_is_not_degenerate` — >50% unique captions
- `test_bounce_ratio_is_not_degenerate` — regression guard for Problem 1
- `test_speed_ranges_allow_avoiding_walls` — regression guard for Problem 2
- `test_frames_tensor_roundtrip` — uint8 → [−1,1] → uint8 is lossless

**Late fix — grammar.** Captions read "A 8 moves…". Only 8 ("eight") starts with a vowel
sound, so it needs "an". Beyond reading as broken English, "a 8" tokenizes oddly for
CLIP. Added `article()` and rebuilt the dataset.

---

## Phase 2 — Frozen CLIP text encoder

### What it is

CLIP ViT-B/32's text tower converts a caption into a 512-d embedding. It is **frozen**
(`requires_grad=False`, eval mode) and its outputs are cached to `text_embeddings.npy`
before training, so CLIP never runs inside the training loop.

### Why frozen

Training a language model is not what this project is about, and freezing means the text
tower contributes no optimizer state and no gradient memory. The same frozen model is
reused at evaluation time for CLIPSIM — one model, two roles.

### Errors and fixes

**Problem 4 — encoding 24k captions took >45 minutes on CPU and had to be killed.**

Three compounding causes, all fixed:

1. **Padding waste.** CLIP's tokenizer pads every caption to 77 tokens; ours are ~20.
   About 70% of the compute went into padding. The fix is exact rather than approximate:
   CLIP's text transformer uses a **causal** attention mask and reads the sentence
   embedding out at the EOT token, so nothing after EOT can influence the output.
   Truncating to the longest EOT index in the batch is therefore mathematically
   identical. `_check_truncation_is_exact()` verifies this against the reference
   `encode_text` at construction time and silently falls back if open_clip's internals
   differ by version — a fast path with a correctness assertion, not a leap of faith.
2. **Thread underuse.** torch defaulted to 4 of 8 cores. `torch.set_num_threads(os.cpu_count())`.
3. **Duplicate captions.** ~25% of the 20k training captions repeat and CLIP is
   deterministic, so unique captions are encoded once and shared.

Result: **~10 captions/s → 113 captions/s**, and the full build went from 45+ minutes
(killed, unfinished) to **~6 minutes**.

**Problem 5 — the embedding sanity check reported FAIL, and the check itself was wrong.**

The first version compared "same direction, different wording" (0.662) against "opposite
directions" (0.759) and concluded CLIP could not encode direction. The comparison was
rigged: the same-direction pairs differed in digit, verb, article *and* phrasing all at
once, while the opposite-direction pairs differed by a single token. Cosine similarity
was measuring how many words changed, not meaning.

Replaced with two proper checks:

*Minimal pairs* — change exactly one thing:

| comparison | cosine similarity |
|---|---|
| same meaning, different wording ("moves to the right" vs "glides rightward") | 0.8058 |
| different meaning, similar wording ("to the right" vs "to the left") | **0.9828** |

*Linear probes* on real dataset captions — can a linear classifier recover the attribute
from the frozen embedding?

| attribute | accuracy | chance |
|---|---|---|
| direction | **93.0%** | 12.5% |
| digit | **99.6%** | 10.0% |
| speed | **68.3%** | 33.3% |

(A threshold bug in the check demanded `chance × 3` — i.e. 100% for the 3-class speed
probe. Fixed to `max(0.40, chance × 1.8)`.)

### The key insight this produced

These two results look contradictory and are not. **The directional information is
present and linearly decodable (93%), but it lives in a low-variance subspace that
whole-vector cosine similarity does not surface (0.98 similar).**

Two consequences, both load-bearing for the rest of the project:

1. **Conditioning will work.** A linear readout suffices, and the model has a whole
   trainable MLP on top.
2. **CLIPSIM will be a weak alignment metric here**, because CLIPSIM *is* whole-vector
   cosine similarity. This is a prediction with evidence, made before any model was
   trained — and it is precisely why structured grounding is the primary alignment
   metric.

---

## Phase 3 — Models

### Design: what is held constant

Both models are the *same* conditional VAE — same `VideoEncoder`, `text_proj`, posterior,
learned prior, latent size, `FrameDecoder`, and loss. Exactly one thing differs:

| | how the 16 bottlenecks are produced |
|---|---|
| `FrameIndependentVAE` (baseline) | an MLP applied to `(z, text, frame_index)`, independently per frame |
| `ConvLSTMVAE` (main) | a ConvLSTM cell stepped 16 times, hidden state carried forward |

Holding everything else constant is what makes the comparison attributable to the
recurrence rather than to incidental architecture differences.

### Why the baseline gets the frame index

A baseline that emitted the same frame 16 times would score a **perfect** temporal
consistency number while generating no motion — the exact metric-gaming failure the
evaluation is built to catch. Giving it the frame index means it *can* produce motion;
what it lacks is continuity. That makes it a fair control.

### Tensor shapes

| Stage | Shape |
|---|---|
| input video | `(B, 16, 1, 64, 64)` |
| folded for 2D convs | `(B·16, 1, 64, 64)` |
| encoder output | `(B, 256)` |
| latent `z` | `(B, 128)` |
| text embedding → `text_proj` | `(B, 512)` → `(B, 128)` |
| per-frame conditioning | `(B, 16, 128+128+64=320)` |
| bottleneck | `(B, 16, 128, 4, 4)` |
| FiLM conditioning vector | `(B·16, 256)` |
| decoder 4→8→16→32→64 | `(B·16, 1, 64, 64)`, `tanh` |

### Parameter counts

| model | total | encoder | decoder |
|---|---|---|---|
| baseline | **4,270,273** | 2,133,664 | 575,009 |
| convlstm | **6,503,105** | 2,133,664 | 575,009 |

The ConvLSTM carries 2.23M more (the cell is ~1.18M, the state init projection ~1.05M).
That gap invites the objection "maybe it just has more capacity", so
`configs/train_baseline_wide.yaml` widens the baseline's MLP (`hidden_dim` 512 → 1500) as
a capacity-matched control. `build_report.py` prints the parameter ratio automatically.

### Specific design decisions

**ConvLSTM forget-gate bias initialised to 1.0.** With a zero bias, `sigmoid(0)=0.5`
halves the cell state every step and motion information decays before it can be used.

**FiLM zero-initialised** so it starts as the identity — conditioning ramps in instead of
destroying the signal at step 0.

**`logvar` clamped to [−8, 8].** An over-confident posterior drives logvar toward −∞,
`exp(logvar)` underflows to 0, and the KL becomes NaN. This is the single most common way
VAE training dies.

**Learned prior `p(z|text)`, not a standard normal.** Generation samples from this prior,
so it must be text-conditioned or text-only generation is impossible.

**`tanh` output** matching the `[−1, 1]` frame normalisation.

### KL weight reasoning

Reconstruction is a per-pixel mean (~0.01–0.02 once trained); KL is summed over 128
latent dims (~20–60 nats). An unscaled `beta=1` would swamp reconstruction entirely and
collapse the posterior. `kl_weight = 1e-4` puts the KL term at roughly 10–40% of total
loss. Warm-up ramps beta from 0 over 2,000 steps, and `free_bits=0.02` nats/dim stops
unused dimensions collapsing.

### Errors and fixes

**Problem 6 — the ConvLSTM failed its tiny-overfit test (0.418 → 0.214, needed <0.209).**

Not a bug. The test was fitting **uniform random noise**, which has no temporal structure
for a recurrent bottleneck to exploit, while the baseline's per-frame MLP could memorise
it directly. Rewrote the test to use real Bouncing MNIST clips — the data the model is
actually for. Both models now pass comfortably at a stricter threshold (<25% of starting
loss in 150 steps).

---

## Phase 4 — Training pipeline

Single `Trainer` for both models. Mixed precision on CUDA (no-op on CPU), gradient
clipping at 1.0, Adam at 2e-4, batch 32, 20,000 steps.

**Checkpoints carry** model + optimizer + scheduler + global step + epoch + RNG states
(torch/numpy/python/cuda) + config snapshot + metric history. Saved on a step interval
**and** a wall-clock timer, written atomically via a `.tmp` rename, mirrored to Drive.

Why the wall-clock timer matters: on a free GPU an epoch can take longer than the
disconnect window, so a step-only policy can lose an entire session's work.

Why full state: resuming with a fresh optimizer resets Adam's moment estimates, which
shows up as a visible bump in the loss curve after every reconnect.

### Verified locally (CPU, 30-step smoke runs)

```
convlstm  step 30/30  loss=0.10630  recon=0.10622  kl=64.813  0.84 it/s
baseline  step 30/30  loss=0.10757  recon=0.10750  kl=56.478  1.18 it/s
resumed from .../last.pt at step 30
```

Reconstruction falls 0.466 → 0.106 in 30 steps; KL rises during warm-up as expected
(beta ≈ 1e-6 at that point, so the posterior is essentially unconstrained).

CPU throughput at batch 4 implies ~55 hours for a full 20k-step run at batch 32 —
confirming GPU is required, not optional.

---

## Phase 5 — Evaluation

### Metrics and why each was chosen

**FID over CLIP image features, not Inception.** Conventional FID uses InceptionV3
trained on 299×299 natural RGB images. Our frames are 64×64 grayscale digits, far outside
that domain. We already load CLIP for CLIPSIM, so CLIP-FID adds no dependency and its
features are at least as meaningful here. Inception remains available via
`--fid-extractor inception`. Either way, absolute values are not comparable to published
numbers — only relative comparison between our own models is claimed.

**Temporal consistency is never frame-similarity alone.** Always paired with optical-flow
motion magnitude and centroid speed. `temporal_score = max(0, 1 − relative motion error)
× frame_ssim`, anchored on real data, so a static model scores 0 regardless of its SSIM.

**Structured grounding is the primary alignment metric.** An independently trained MNIST
CNN judges digit identity; centroid displacement judges direction and speed. No CLIP
anywhere in this path.

### Independent digit classifier

`DigitCNN`, 458,890 params, trained 1,500 steps in 232s on CPU. Trained with the *same*
random-64×64-placement-then-crop path it faces at evaluation time, so its reported
accuracy reflects real conditions.

**Held-out MNIST test accuracy: 98.68%.**

### Measured ceilings on real data (val, n=200)

Metrics have measurement error, so generated scores must be read against what is
achievable, not against 100%:

| metric | real-data value |
|---|---|
| digit accuracy | 99.24% |
| identity consistency | 100.0% |
| direction accuracy | 94.66% |
| speed accuracy | 88.55% |
| **grounding score** | **94.15%** |
| frame SSIM | 0.8900 |
| centroid speed | 1.50 px/frame |

### The static control (n=150) — why SSIM alone is useless

Frame 0 repeated 16 times:

| metric | value |
|---|---|
| frame SSIM | **1.0000** (perfect — would top a naive leaderboard) |
| motion magnitude | 0.0000 |
| centroid speed | 0.0000 |
| **temporal score** | **0.0000** |
| direction accuracy | 0.0% |
| digit accuracy | 99.0% (frame 0 does contain the right digit) |

This control is evaluated as a row in the results table, not just described.

### Errors and fixes

**Problem 7 — `scipy.linalg.sqrtm` no longer accepts `disp=False`.** Recent scipy dropped
the keyword; older versions return a `(result, error)` tuple when it is passed. Wrapped in
a helper that handles both.

**Problem 8 — the static control scored 35% on speed accuracy.** Zero displacement falls
inside the "slow" bucket, so a frozen clip was credited whenever the caption asked for
slow motion. "Did not move" is not a correct answer to "move slowly". Added
`STATIC_SPEED_THRESHOLD = 0.3` px/frame: below it, a clip is scored wrong regardless of
the requested bucket, and `speed_static_fraction` is reported separately. Static control
speed accuracy is now 0%, and a test pins the threshold below the slowest real speed
(0.8 px/frame) so it can never swallow genuinely slow clips.

### Full pipeline verified end to end

Evaluated a 30-step smoke checkpoint — the numbers are appropriately terrible, which is
the point: the machinery reports honestly rather than flattering an untrained model.

```
grounding_score 0.0325   direction_accuracy 0.0000   digit_accuracy 0.0976
temporal_score  0.0469   frame_ssim         0.9980   centroid_speed  0.0689
fid           108.8727   clipsim            0.1645
```

Note `frame_ssim = 0.998` on an untrained model that outputs nearly-static blur — a third
independent confirmation that frame similarity alone is meaningless.

---

## Phase 6 — Inference, demo, and Colab

`VideoGenerator` wraps CLIP + model + scoring behind one object, shared by the CLI
sampler and the Streamlit demo so both take exactly the same code path. It reports what
was *asked for* beside what was *measured*, including a `STATIC` flag.

Verified on the smoke checkpoint:

```
The digit 3 moves to the right.
  -> digit 1 [MISS]  dir n/a  0.14px/f  STATIC
```

Correct behaviour for a 30-step model: wrong digit, no motion, honestly reported.

`notebooks/train_on_colab.ipynb` — 30 cells, validated as JSON, every `scripts/*.py` and
`configs/*.yaml` path it references confirmed to exist. The dataset is regenerated on
Colab rather than uploaded: it is deterministic from the seeds, and 1.5 GB is not worth
uploading when the code is a few hundred KB.

---

## Test suite

**110 tests, all passing, CPU-only, no network.**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

| file | count | covers |
|---|---|---|
| `test_data_pipeline.py` | 21 | clip physics, caption truthfulness, dataset tensors |
| `test_models.py` | 36 | shapes, gradient flow, losses, tiny-overfit gate |
| `test_evaluation.py` | 30 | FID properties, the SSIM trap, grounding, report rendering |
| `test_training.py` | 23 | checkpoint round-trips, resumability, KL schedule, config |

Several tests exist specifically as regression guards for the bugs above:
`test_bounce_ratio_is_not_degenerate` (Problem 1), `test_speed_ranges_allow_avoiding_walls`
(Problem 2), `test_static_clips_score_zero_speed` (Problem 8),
`test_static_video_scores_perfect_ssim` + `test_temporal_score_rejects_the_static_control`
(the metric trap), `test_optimizer_state_is_preserved` (silent resume corruption).

---

## Phase 7 — Kaggle GPU workflow

Training moved to Kaggle. `notebooks/train_on_kaggle.ipynb` (41 cells) adapts the Colab
notebook; the Colab one is kept as an alternative.

### What changed for Kaggle

| Colab | Kaggle |
|---|---|
| `google.colab` imports, `drive.mount()` | none — no Drive |
| `/content/...` | `/kaggle/working/...` |
| `files.download()` | Save Version → Output tab |
| checkpoints mirrored to Drive | mirrored to `/kaggle/working/checkpoints/` |

### The install trap this notebook is built to avoid

Kaggle ships a CUDA-enabled PyTorch. A plain `pip install -e .` lets pip resolve this
project's `torch` dependency and **silently replace the CUDA build with a CPU wheel** —
training then runs at CPU speed on a GPU machine, which is easy to miss because nothing
errors.

Fix: install with `--no-deps` and add only genuinely missing packages, then **assert**
`torch.cuda.is_available()` and `torch.version.cuda is not None` afterwards. The
environment check in section 1 also refuses to continue without a GPU rather than
starting a ~55-hour CPU run by accident.

### Manual training gate

Sections 1–7 (setup, dataset, embeddings, classifier, smoke test) run in ~15 minutes and
are separated from real training by an explicit banner. Every check *verifies* rather
than assumes: dataset shapes and counts, embedding dimension and L2 norm, classifier
accuracy ≥95%, and — post-smoke-run — that loss is finite, actually fell, checkpoints
reload, generation runs on CUDA, and output pixels are finite. Any failure raises
`SystemExit` before GPU hours are spent.

### Script change

`train_digit_classifier.py` gained `--device` (it was CPU-only) and now saves CPU tensors
so the checkpoint loads anywhere. Verified: `--device cuda` path works, reload succeeds.

### Errors and fixes

**Problem 9 — the training-info cell read run_record keys that do not exist.**
It expected `total_params`, `batch_size`, `lr`, `seed` at the top level; they actually
live under `model`, `config.train` and `config.seed`. Caught by executing the cell
locally against a real `run_record.json` rather than eyeballing it. Rewritten against the
true structure.

**Problem 10 — a `\n` inside a generated cell became a real newline**, splitting a
`print(` statement across two lines and leaving an unterminated string. Caught by
compiling every code cell. Fixed, and all 41 cells now compile.

### Local validation performed

No GPU here, so Kaggle GPU training itself is **not** locally tested. What *was* verified:

- notebook is valid JSON; all 41 code cells compile
- every referenced `scripts/*.py` and `configs/*.yaml` exists
- zero Colab-specific strings remain (`google.colab`, `drive.mount`, `/content/`, `files.download`, `MyDrive`)
- the pure-Python cells were **executed locally** against real artifacts with CUDA and
  `/kaggle/working` patched out: dataset verify, embedding verify, classifier check,
  smoke verify, training info, comparison, bundle packaging, bundle verify — all pass
- the baseline-vs-ConvLSTM comparison was tested with two synthetic records including a
  deliberate ConvLSTM regression, confirming correct polarity (lower FID/params/inference
  = better) and that regressions are reported honestly rather than assumed away

### Results bundle

`/kaggle/working/results_bundle.zip` collects best checkpoints, run records, loss
histories, all `logs_*.txt`, configs, evaluation records, `comparison.json`, `RESULTS.md`,
samples, and environment/timing info. The 1.5 GB regenerated dataset is excluded — it is
reproducible from the seeds.

---

## Status and what comes next

Everything is built and verified except the thing that needs a GPU: **actual training**.

The local machine has no CUDA device, and CPU throughput implies ~55 hours per run. So
the next step is running `notebooks/train_on_kaggle.ipynb` on a Kaggle T4 (~1.5–2 hours
for both models plus evaluation), then bringing back `results_bundle.zip` to populate
`RESULTS.md` with real numbers.

**No performance figure appears anywhere in this repo until it comes from a real run.**
