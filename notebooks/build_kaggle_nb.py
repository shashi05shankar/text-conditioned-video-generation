"""Generate notebooks/train_on_kaggle.ipynb from scratch.

Building it programmatically (rather than hand-editing JSON) keeps every cell's source
a plain raw string, so backslashes and newlines cannot be mangled by successive patches.
"""

import json
import pathlib

REPO = "/kaggle/working/text-conditioned-video-generation"

cells = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": text.strip("\n"),
    })


# Repeated in every in-kernel cell so each one is runnable on its own, in any order,
# after any kernel restart. See BUILD_LOG for why this is not optional.
BOOT = r'''import os, sys
REPO_DIR = '/kaggle/working/text-conditioned-video-generation'
os.chdir(REPO_DIR)                                   # a kernel restart resets cwd
if os.path.join(REPO_DIR, 'src') not in sys.path:    # and drops the editable-install path
    sys.path.insert(0, os.path.join(REPO_DIR, 'src'))
'''

# --------------------------------------------------------------------------- header
md(r'''
# Text-Conditioned Video Generation — Kaggle GPU Training

Trains two models and produces the real results for this project.

## Before you run anything

In the right-hand panel:
- **Accelerator → GPU T4 x2** (or P100)
- **Internet → On** — required for the repo clone, MNIST download and CLIP weights

## How this notebook is organised

**Sections 1–7 are setup and sanity checks (~15 minutes, no real GPU cost).**
Long training only starts after the large banner further down.

Run 1–7 first, check the output, and continue past the banner only once it looks right.

Every cell is self-contained: it re-establishes the working directory and import path
before doing anything, so cells can be re-run in any order and survive a kernel restart.

## What is being compared

Two conditional VAEs, identical except for how the 16 frames are produced — an MLP
applied independently per frame, versus a ConvLSTM carrying hidden state across frames.
Same encoder, latent, decoder, loss, seed, data and step budget, so any difference is
attributable to the recurrence.

**The experiment decides whether the ConvLSTM helps. It is not assumed.**
''')

# --------------------------------------------------------------------------- 1
md(r'''
## 1. Environment check

Fails loudly if there is no GPU, rather than silently starting a ~55-hour CPU run.
''')

code(r'''
import platform, sys

print('python           :', platform.python_version())
print('platform         :', platform.platform())

import torch
print('torch            :', torch.__version__)
print('torch CUDA build :', torch.version.cuda)
print('cuda available   :', torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit(
        'NO GPU DETECTED.\n'
        'Right-hand panel -> Accelerator -> GPU T4 x2, then restart the session.\n'
        'Refusing to continue: CPU training would take roughly 55 hours per model.'
    )

props = torch.cuda.get_device_properties(0)
print('gpu              :', props.name)
print('gpu memory       : %.1f GB' % (props.total_memory / 1024**3))
print('gpu count        :', torch.cuda.device_count())

ENV_INFO = {
    'python': platform.python_version(),
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'gpu_name': props.name,
    'gpu_memory_gb': round(props.total_memory / 1024**3, 2),
    'gpu_count': torch.cuda.device_count(),
    'platform': 'kaggle',
}
print('\nOK - GPU ready')
''')

code(r'''!nvidia-smi''')

# --------------------------------------------------------------------------- 2
md(r'''
## 2. Clone the repository

Safe to re-run — pulls instead of failing if the directory already exists.
''')

code(r'''
import os, subprocess, sys

REPO_URL = 'https://github.com/shashi05shankar/text-conditioned-video-generation.git'
REPO_DIR = '/kaggle/working/text-conditioned-video-generation'

if os.path.isdir(os.path.join(REPO_DIR, '.git')):
    print('repo already present, pulling latest')
    r = subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'],
                       capture_output=True, text=True)
else:
    print('cloning', REPO_URL)
    r = subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], capture_output=True, text=True)

# Always surface git's own output -- hiding it makes a failure here impossible to
# diagnose, and the cause is usually stated plainly in stderr.
print('git exit code:', r.returncode)
if r.stdout.strip():
    print('stdout:', r.stdout.strip())
if r.stderr.strip():
    print('stderr:', r.stderr.strip())

if r.returncode != 0 or not os.path.isdir(REPO_DIR):
    raise SystemExit(
        'Git failed -- see the stderr above for the reason.\n'
        '\n'
        'By far the most common cause is Internet being disabled. In the right-hand\n'
        'panel: Internet -> On (Kaggle may ask you to verify a phone number first),\n'
        'then re-run this cell.\n'
        '\n'
        'If the repo already exists but the pull failed, delete and re-clone with:\n'
        '    import shutil; shutil.rmtree(REPO_DIR, ignore_errors=True)'
    )

os.chdir(REPO_DIR)
os.environ['REPO_DIR'] = REPO_DIR
if os.path.join(REPO_DIR, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(REPO_DIR, 'src'))

print('cwd     :', os.getcwd())
print('contents:', sorted(os.listdir('.')))
''')

# --------------------------------------------------------------------------- 3
md(r'''
## 3. Dependencies

Kaggle already ships a CUDA-enabled PyTorch. A plain `pip install -e .` would let pip
resolve this project's `torch` dependency and **silently replace the CUDA build with a
CPU wheel** — training would then run at CPU speed on a GPU machine, with nothing
raising an error.

So we install with `--no-deps` and add only genuinely missing packages. The next cell
verifies in a fresh subprocess that CUDA survived.
''')

code(r'''
import importlib, subprocess, sys

REPO_DIR = '/kaggle/working/text-conditioned-video-generation'
os.chdir(REPO_DIR)

needed = {'open_clip': 'open_clip_torch', 'skimage': 'scikit-image',
          'cv2': 'opencv-python-headless', 'imageio': 'imageio', 'yaml': 'PyYAML'}
missing = []
for module, package in needed.items():
    try:
        importlib.import_module(module)
        print(f'  present : {module}')
    except ImportError:
        missing.append(package)
        print(f'  MISSING : {module} -> installing {package}')

if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)

# --no-deps is the important part: registers the package without touching torch.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '-e', '.'],
               check=True)
print('\ninstalled text2video (--no-deps)')
''')

code(BOOT + r'''
# Verify the CUDA build survived the install.
#
# The probe runs in a FRESH subprocess deliberately. Reloading torch in-process is not
# possible -- it registers C++ operator namespaces at import time and re-executing that
# raises "Only a single TORCH_LIBRARY can be used...". A subprocess is also the only
# honest check: it reads what is installed on disk now, not the copy this kernel
# imported before pip ran.
import json, subprocess

probe = (
    "import torch, json;"
    "print(json.dumps({"
    "'version': torch.__version__,"
    "'cuda_build': torch.version.cuda,"
    "'available': torch.cuda.is_available(),"
    "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
)
result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)
if result.returncode != 0:
    print(result.stdout); print(result.stderr)
    raise SystemExit('torch failed to import in a fresh interpreter after installing')

info = json.loads(result.stdout.strip().splitlines()[-1])
print('torch          :', info['version'])
print('torch CUDA     :', info['cuda_build'])
print('cuda available :', info['available'])
print('device         :', info['device'])

if not info['available'] or info['cuda_build'] is None:
    raise SystemExit(
        'CUDA disappeared after installing dependencies -- a CPU-only torch wheel was '
        'pulled in.\n'
        'Fix: Run -> Factory reset, then start again from section 1.'
    )

# pip registers an editable install via a .pth file in site-packages, and .pth files are
# only read at interpreter STARTUP. This kernel was already running, so it never sees
# it -- hence the explicit src/ path above.
import text2video
print('text2video     :', text2video.__version__, 'from', os.path.dirname(text2video.__file__))
print('\nOK - GPU torch intact, package importable')
''')

# --------------------------------------------------------------------------- 4
md(r'''
## 4. Generate the dataset

Takes 2–3 minutes. The dataset is **generated here, not uploaded** — it is fully
deterministic from the seeds in `configs/dataset.yaml`, and 1.5 GB is not worth
uploading when the code is a few hundred KB.

Expected: 20,000 train / 2,000 val / 2,000 test clips of 16 frames at 64×64.

Watch for the `[train] frames (20000, 16, 64, 64)` line — that is the confirmation it
actually wrote the data.
''')

code(r'''!cd "/kaggle/working/text-conditioned-video-generation" && python scripts/build_dataset.py --config configs/dataset.yaml''')

code(BOOT + r'''
# Verify the dataset rather than assuming the script worked.
import json, pathlib
import numpy as np

print('cwd:', os.getcwd())

root = pathlib.Path('data/processed')
if not root.exists():
    raise SystemExit(
        f'No {root.resolve()} -- the dataset was not generated.\n'
        'Re-run the previous cell and check it finished without an error.'
    )

EXPECTED = {'train': 20000, 'val': 2000, 'test': 2000}
problems = []

for split, expected_count in EXPECTED.items():
    d = root / split
    if not (d / 'frames.npy').exists():
        problems.append(f'{split}: missing {d / "frames.npy"}')
        continue

    frames = np.load(d / 'frames.npy', mmap_mode='r')
    captions = json.loads((d / 'captions.json').read_text())
    metadata = json.loads((d / 'metadata.json').read_text())

    print(f'{split:6s} frames={frames.shape} {frames.dtype}  '
          f'captions={len(captions)}  metadata={len(metadata)}')
    print(f'         unique captions: {len(set(captions))} '
          f'({len(set(captions)) / len(captions):.1%})')
    print(f'         example: {captions[0]}')

    if frames.shape != (expected_count, 16, 64, 64):
        problems.append(f'{split}: expected {(expected_count, 16, 64, 64)}, got {frames.shape}')
    if not len(frames) == len(captions) == len(metadata):
        problems.append(f'{split}: frames/captions/metadata length mismatch')
    if not all(k in metadata[0] for k in ('digits', 'directions', 'speeds', 'bounced')):
        problems.append(f'{split}: metadata missing required keys')

if problems:
    raise SystemExit('DATASET GENERATION FAILED:\n  ' + '\n  '.join(problems))
print('\nOK - dataset verified')
''')

# --------------------------------------------------------------------------- 5
md(r'''
## 5. CLIP text embeddings

Encodes every caption once with the frozen CLIP ViT-B/32 text tower and caches the
result, so CLIP never runs inside the training loop.

`--verify` runs a minimal-pair test and three linear probes. The probes answer the
question that actually matters: **can a linear readout recover direction / digit / speed
from the frozen embedding?** If not, conditioning could never work and there would be no
point training anything.

A local CPU run gave roughly 93% / 99.6% / 68%. Those are **sanity references, not
targets** — report whatever Kaggle produces.
''')

code(r'''!cd "/kaggle/working/text-conditioned-video-generation" && python scripts/build_text_embeddings.py --device cuda --verify 2>&1 | tee /kaggle/working/logs_embeddings.txt''')

code(BOOT + r'''
import pathlib
import numpy as np

for split in ('train', 'val', 'test'):
    path = pathlib.Path('data/processed') / split / 'text_embeddings.npy'
    if not path.exists():
        raise SystemExit(f'MISSING {path} -- the embedding build failed')
    emb = np.load(path)
    norms = np.linalg.norm(emb, axis=1)
    print(f'{split:6s} {emb.shape} {emb.dtype}  L2 norm mean={norms.mean():.4f}')
    assert emb.shape[1] == 512, 'expected 512-d CLIP embeddings'
    assert abs(norms.mean() - 1.0) < 0.01, 'embeddings should be L2-normalised'

print('\nOK - embeddings cached')
''')

# --------------------------------------------------------------------------- 6
md(r'''
## 6. Independent digit classifier

The judge for the structured-grounding metric. It shares no weights with the generators
and never sees generated data, which is what keeps the metric independent of the thing
it measures.

A local run reached 98.68% held-out. Again a sanity reference — use the real number.
''')

code(r'''!cd "/kaggle/working/text-conditioned-video-generation" && python scripts/train_digit_classifier.py --device cuda 2>&1 | tee /kaggle/working/logs_digit_classifier.txt''')

code(BOOT + r'''
import json, pathlib

path = pathlib.Path('outputs/digit_classifier.json')
if not path.exists():
    raise SystemExit('No outputs/digit_classifier.json -- the previous cell failed')

info = json.loads(path.read_text())
print(json.dumps(info, indent=2))

if info['test_accuracy'] < 0.95:
    raise SystemExit(
        f'Digit classifier only reached {info["test_accuracy"]:.2%}. Grounding scores '
        'would be too noisy to trust. Investigate before training.'
    )
print(f'\nOK - classifier at {info["test_accuracy"]:.2%}')
''')

# --------------------------------------------------------------------------- 7
md(r'''
## 7. Smoke test — the gate before spending GPU hours

30 steps per model on a tiny subset. This proves, on the actual GPU: the forward pass
runs, the backward pass runs, the loss is finite and falls, checkpoints save and reload,
generation runs on CUDA, and the output pixels are finite.

**If this fails, stop.** A failure here becomes a failure an hour into real training.
''')

code(r'''!cd "/kaggle/working/text-conditioned-video-generation" && python scripts/train.py --config configs/train_convlstm.yaml --device cuda --smoke
!cd "/kaggle/working/text-conditioned-video-generation" && python scripts/train.py --config configs/train_baseline.yaml --device cuda --smoke''')

code(BOOT + r'''
# Independently verify what the smoke runs actually produced.
import json, math, pathlib
import torch

from text2video.evaluation.harness import load_model_from_checkpoint

problems = []
for variant in ('baseline', 'convlstm'):
    run = pathlib.Path(f'outputs/runs/{variant}_smoke')
    if not (run / 'history.json').exists():
        problems.append(f'{variant}: no run at {run}')
        continue

    history = json.loads((run / 'history.json').read_text())
    losses = [h['loss'] for h in history if 'loss' in h]

    if not losses:
        problems.append(f'{variant}: no loss history')
        continue
    if not all(math.isfinite(v) for v in losses):
        problems.append(f'{variant}: non-finite loss (NaN/inf)')
    if losses[-1] >= losses[0]:
        problems.append(f'{variant}: loss did not fall ({losses[0]:.4f} -> {losses[-1]:.4f})')

    model, payload = load_model_from_checkpoint(run / 'checkpoints' / 'last.pt', 'cuda')
    with torch.no_grad():
        out = model.generate(torch.randn(2, 512, device='cuda'))

    if out.shape != (2, 16, 1, 64, 64):
        problems.append(f'{variant}: bad output shape {tuple(out.shape)}')
    if not torch.isfinite(out).all():
        problems.append(f'{variant}: generated non-finite pixels')
    if out.device.type != 'cuda':
        problems.append(f'{variant}: generation did not run on GPU')

    print(f'{variant:9s} loss {losses[0]:.4f} -> {losses[-1]:.4f}  '
          f'step={payload["global_step"]}  out={tuple(out.shape)} on {out.device}  OK')

print(f'\npeak GPU memory so far: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB')

if problems:
    raise SystemExit('SMOKE TEST FAILED:\n  ' + '\n  '.join(problems))
print('\nOK - safe to start real training')
''')

# --------------------------------------------------------------------------- banner
md(r'''
---
---

# ==============================
# LONG GPU TRAINING STARTS BELOW
# ==============================

---

**Everything above is setup. Stop here on the first pass**, review the output, and only
continue once it looks right.

Before continuing, confirm you saw:

- [ ] section 1 — your GPU name and memory
- [ ] section 4 — `OK - dataset verified`, with 20000 / 2000 / 2000
- [ ] section 5 — three probe accuracies well above chance
- [ ] section 6 — classifier accuracy ≥ 95%
- [ ] section 7 — `OK - safe to start real training`

From here the notebook spends real GPU time:

| Section | Runs | Rough time on a T4 |
|---|---|---|
| 8 | Baseline, 20,000 steps | 30–50 min |
| 9 | ConvLSTM, 20,000 steps | 40–70 min |
| 10 | Optional capacity-matched baseline | 30–50 min |
| 11–13 | Evaluation, samples, results bundle | 10–15 min |

**If the session dies mid-training, just re-run the same cell.** `--resume auto` restores
model, optimizer, scheduler and RNG state from the latest checkpoint — not only the
weights, so the run genuinely continues instead of restarting Adam's momentum.

Checkpoints are mirrored to `/kaggle/working/checkpoints/`. Section 13 explains how to
get everything off Kaggle before the session ends.
''')

code(BOOT + r'''
MIRROR = '/kaggle/working/checkpoints'
os.makedirs(MIRROR, exist_ok=True)
print('checkpoints mirrored to', MIRROR)
''')

# --------------------------------------------------------------------------- 8
md(r'''
## 8. Train the BASELINE (frame-independent)

4.27M parameters. Frames are decoded independently from `(z, text, frame_index)` — it
*can* produce motion, it simply has no continuity between frames.

Watch `recon` fall. `kl` **rising** over the first 2,000 steps is correct: beta ramps
from 0 during warm-up, so the posterior is nearly unconstrained early on.
''')

code(r'''!cd "/kaggle/working/text-conditioned-video-generation" && python scripts/train.py --config configs/train_baseline.yaml --device cuda --resume auto --mirror-dir /kaggle/working/checkpoints/baseline 2>&1 | tee /kaggle/working/logs_train_baseline.txt''')

# --------------------------------------------------------------------------- 9
md(r'''
## 9. Train the MAIN MODEL (cVAE + ConvLSTM)

6.50M parameters. Same everything, except the 16 bottlenecks come from a ConvLSTM cell
stepped 16 times, whose hidden state carries motion information forward.
''')

code(r'''!cd "/kaggle/working/text-conditioned-video-generation" && python scripts/train.py --config configs/train_convlstm.yaml --device cuda --resume auto --mirror-dir /kaggle/working/checkpoints/convlstm 2>&1 | tee /kaggle/working/logs_train_convlstm.txt''')

code(BOOT + r'''
# Record what actually happened, for the results bundle.
import json, pathlib
import torch

TRAIN_INFO = {'env': globals().get('ENV_INFO', {}), 'runs': {}}

for variant in ('baseline', 'convlstm', 'baseline_wide'):
    record_path = pathlib.Path(f'outputs/runs/{variant}/run_record.json')
    if not record_path.exists():
        print(f'{variant}: not trained (no run_record.json)')
        continue

    record = json.loads(record_path.read_text())
    TRAIN_INFO['runs'][variant] = record

    model = record.get('model', {})
    train_cfg = record.get('config', {}).get('train', {})
    val = record.get('final_val_metrics', {})

    print(f'--- {variant} ---')
    print(f'  params        : {model.get("total_params"):,}')
    print(f'  steps         : {record.get("global_step")}  (epochs {record.get("epochs")})')
    print(f'  train time    : {record.get("train_minutes")} min '
          f'({record.get("seconds_per_step")} s/step)')
    print(f'  batch / lr    : {train_cfg.get("batch_size")} / {train_cfg.get("lr")}')
    print(f'  seed          : {record.get("config", {}).get("seed")}')
    print(f'  device / amp  : {record.get("device")} / amp={record.get("amp")}')
    print(f'  peak GPU mem  : {record.get("peak_gpu_memory_mb")} MB')
    print(f'  final val loss: {val.get("val_loss")}  (recon {val.get("val_recon")})')

TRAIN_INFO['peak_gpu_memory_gb'] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
print('\nsession peak GPU memory:', TRAIN_INFO['peak_gpu_memory_gb'], 'GB')

pathlib.Path('/kaggle/working/train_info.json').write_text(json.dumps(TRAIN_INFO, indent=2))
print('wrote /kaggle/working/train_info.json')
''')

# --------------------------------------------------------------------------- 10
md(r'''
## 10. (Optional) Capacity-matched baseline

The ConvLSTM carries ~2.2M more parameters. This run widens the baseline's MLP to close
that gap while keeping it strictly frame-independent, so "it only won because it was
bigger" can be ruled out rather than argued about.

Skip it if you are short on GPU quota — the core comparison does not depend on it.
''')

code(r'''!cd "/kaggle/working/text-conditioned-video-generation" && python scripts/train.py --config configs/train_baseline_wide.yaml --device cuda --resume auto --mirror-dir /kaggle/working/checkpoints/baseline_wide 2>&1 | tee /kaggle/working/logs_train_baseline_wide.txt''')

# --------------------------------------------------------------------------- 11
md(r'''
## 11. Evaluate

Runs every trained model on the held-out **test** split with a fixed sampling seed, plus
two reference rows:

- **real-data ceiling** — the same metrics on ground-truth clips. Measurement is
  imperfect, so generated scores should be read against what is achievable, not 100%.
- **static control** — frame 0 repeated 16×, which scores a *perfect* frame SSIM while
  generating no motion at all. This is why frame similarity is never reported alone.
''')

code(r'''!cd "/kaggle/working/text-conditioned-video-generation" && python scripts/evaluate.py --all --device cuda --split test --num-clips 500 2>&1 | tee /kaggle/working/logs_evaluate.txt''')

code(r'''!cd "/kaggle/working/text-conditioned-video-generation" && python scripts/build_report.py''')

code(BOOT + r'''
import pathlib

report = pathlib.Path('RESULTS.md')
print(report.read_text() if report.exists() else 'RESULTS.md not generated')
''')

code(BOOT + r'''
# Machine-readable baseline vs ConvLSTM comparison.
# No winner is assumed -- whatever the numbers say is what gets recorded.
import json, pathlib

eval_dir = pathlib.Path('outputs/eval')
records = {p.stem: json.loads(p.read_text())
           for p in eval_dir.glob('*.json') if p.stem != 'comparison'}

comparison = {'records': records, 'delta': {}}
base, main = records.get('baseline'), records.get('convlstm')

if base and main:
    # (metric, higher_is_better)
    for metric, higher_better in [
        ('grounding_score', True), ('direction_accuracy', True), ('digit_accuracy', True),
        ('speed_accuracy', True), ('temporal_score', True), ('frame_ssim', True),
        ('centroid_speed', True), ('clipsim', True), ('fid', False),
        ('total_params', False), ('inference_seconds_per_clip', False),
    ]:
        if metric not in base or metric not in main:
            continue
        b, m = base[metric], main[metric]
        if b is None or m is None:
            continue
        absolute = m - b
        improved = (absolute > 0) if higher_better else (absolute < 0)
        comparison['delta'][metric] = {
            'baseline': b, 'convlstm': m,
            'absolute_change': absolute,
            'relative_change': (absolute / abs(b)) if b else None,
            'higher_is_better': higher_better,
            'convlstm_better': bool(improved),
        }
        print(f'{metric:28s} baseline={b:12.4f}  convlstm={m:12.4f}  '
              f'-> convlstm {"better" if improved else "WORSE"}')

    wins = sum(v['convlstm_better'] for v in comparison['delta'].values())
    comparison['summary'] = {'metrics_compared': len(comparison['delta']),
                             'convlstm_better_on': wins}
    print(f'\nConvLSTM better on {wins}/{len(comparison["delta"])} metrics')
else:
    print('WARNING: missing baseline and/or convlstm evaluation records')

(eval_dir / 'comparison.json').write_text(json.dumps(comparison, indent=2))
print('\nwrote outputs/eval/comparison.json')
''')

# --------------------------------------------------------------------------- 12
md(r'''
## 12. Generate samples

The same held-out prompts through both models, saved as GIFs plus a labelled filmstrip.
Each clip is scored for what it actually shows — which digit appeared, which direction it
moved, and whether it moved at all.
''')

code(r'''!cd "/kaggle/working/text-conditioned-video-generation" && python scripts/sample.py --compare --device cuda -n 2 2>&1 | tee /kaggle/working/logs_samples.txt''')

code(r'''!cd "/kaggle/working/text-conditioned-video-generation" && python scripts/sample.py --checkpoint outputs/runs/convlstm/checkpoints/best.pt --device cuda --prompt "Digit 7 moves from left to right." -n 4 --out outputs/samples_prompt''')

code(BOOT + r'''
from IPython.display import Image, display

for variant in ('baseline', 'convlstm'):
    path = f'outputs/samples/{variant}/samples.png'
    if os.path.exists(path):
        print(variant)
        display(Image(path))
    else:
        print(f'{variant}: no samples at {path}')
''')

# --------------------------------------------------------------------------- 13
md(r'''
## 13. Package the results

Builds `/kaggle/working/results_bundle.zip` with everything needed for the write-up:
checkpoints, logs, configs, metrics, the comparison, samples, and environment/timing
information.

The regenerated dataset is deliberately excluded — it is 1.5 GB and reproducible from the
seeds.
''')

code(BOOT + r'''
import json, pathlib, shutil

bundle = pathlib.Path('/kaggle/working/results_bundle')
if bundle.exists():
    shutil.rmtree(bundle)
bundle.mkdir(parents=True)


def copy_into(src, dest_name=None):
    src = pathlib.Path(src)
    if not src.exists():
        print(f'  skip (missing): {src}')
        return
    dest = bundle / (dest_name or src.name)
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    print(f'  added: {dest.relative_to(bundle)}')


print('evaluation + report')
copy_into('outputs/eval')
copy_into('RESULTS.md')
copy_into('outputs/digit_classifier.json')

print('samples')
copy_into('outputs/samples')
copy_into('outputs/samples_prompt')

print('configs')
copy_into('configs')

print('environment + timing')
copy_into('/kaggle/working/train_info.json')

print('logs')
for log in sorted(pathlib.Path('/kaggle/working').glob('logs_*.txt')):
    copy_into(log, f'logs/{log.name}')

print('checkpoints (best only) + run records')
runs_dir = pathlib.Path('outputs/runs')
if runs_dir.exists():
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir() or run.name.endswith('_smoke'):
            continue
        copy_into(run / 'checkpoints' / 'best.pt', f'runs/{run.name}/best.pt')
        copy_into(run / 'run_record.json', f'runs/{run.name}/run_record.json')
        copy_into(run / 'history.json', f'runs/{run.name}/history.json')

total_mb = sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file()) / 1024**2
print(f'\nbundle contents: {total_mb:.1f} MB')

archive = shutil.make_archive('/kaggle/working/results_bundle', 'zip', bundle)
print('wrote', archive, f'({pathlib.Path(archive).stat().st_size / 1024**2:.1f} MB)')
''')

code(BOOT + r'''
import zipfile

names = zipfile.ZipFile('/kaggle/working/results_bundle.zip').namelist()
print(f'{len(names)} entries\n')
for n in sorted(names)[:40]:
    print(' ', n)

required = ['RESULTS.md', 'comparison.json', 'train_info.json']
missing = [r for r in required if not any(n.endswith(r) for n in names)]
print('\nMISSING:', missing if missing else 'nothing - bundle complete')
''')

md(r'''
### Getting the bundle off Kaggle

`/kaggle/working` is wiped when the session ends, so save before closing.

**Save Version (recommended).** Click **Save Version** (top right) → **Quick Save** →
Save. Everything in `/kaggle/working` is stored as the notebook's output. When it
finishes, open the notebook's **Output** tab and download `results_bundle.zip`.

**Or download directly.** In the right-hand panel open the **Output** file browser, find
`results_bundle.zip`, and use the download button.

Then unzip it into the project root locally:

```
D:\Projects\text_video\
```

and the remaining analysis (RESULTS.md, error analysis, sample galleries) can be built
from the real numbers.
''')

# --------------------------------------------------------------------------- write
for cell in cells:
    text = cell["source"]
    lines = text.split("\n")
    cell["source"] = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

out = pathlib.Path(__file__).resolve().parent / "train_on_kaggle.ipynb"
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"wrote {out}  ({len(cells)} cells)")
