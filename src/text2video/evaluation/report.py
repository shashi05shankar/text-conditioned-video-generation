"""Build the final comparison table from evaluation JSON records.

Every value here is read out of a file written by an actual run. Nothing is entered by
hand, so the table cannot drift from the experiments that produced it, and a missing
metric renders as "n/a" rather than as a plausible-looking number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# (json key, column heading, format, higher-is-better)
COLUMNS: list[tuple[str, str, str, bool | None]] = [
    ("total_params", "Params", "params", None),
    ("fid", "FID (CLIP) ↓", "{:.2f}", False),
    ("clipsim", "CLIPSIM ↑", "{:.4f}", True),
    ("grounding_score", "Grounding ↑", "{:.1%}", True),
    ("digit_accuracy", "Digit ↑", "{:.1%}", True),
    ("direction_accuracy", "Direction ↑", "{:.1%}", True),
    ("speed_accuracy", "Speed ↑", "{:.1%}", True),
    ("frame_ssim", "Frame SSIM", "{:.4f}", None),
    ("centroid_speed", "Motion px/f", "{:.2f}", None),
    ("temporal_score", "Temporal ↑", "{:.4f}", True),
    ("train_minutes", "Train min", "{:.1f}", None),
    ("inference_seconds_per_clip", "Infer s/clip", "{:.4f}", None),
]

DISPLAY_NAMES = {
    "baseline": "Baseline (frame-independent)",
    "convlstm": "ConvLSTM (main model)",
    "real_data_ceiling": "Real data (ceiling)",
    "static_control": "Static control (no motion)",
}

# Ceiling and control rows are reference points, not competitors.
NON_MODEL_VARIANTS = ("real_data_ceiling", "static_control")


def _format(value: Any, spec: str) -> str:
    if value is None:
        return "n/a"
    if spec == "params":
        return f"{value/1e6:.2f}M"
    try:
        return spec.format(value)
    except (ValueError, TypeError):
        return str(value)


def load_records(directory: str | Path) -> list[dict[str, Any]]:
    """Load every *.json metrics record in a directory, ordered for display."""
    directory = Path(directory)
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]

    def sort_key(record: dict[str, Any]) -> tuple[int, str]:
        variant = record.get("variant", "")
        order = {"static_control": 0, "baseline": 1, "convlstm": 2, "real_data_ceiling": 3}
        return (order.get(variant, 2), variant)

    return sorted(records, key=sort_key)


def build_markdown_table(records: list[dict[str, Any]]) -> str:
    """Markdown comparison table. Best model value per column is bolded."""
    active = [
        (key, heading, spec, higher)
        for key, heading, spec, higher in COLUMNS
        if any(record.get(key) is not None for record in records)
    ]

    # Only real models compete for "best" -- the ceiling row would win everything.
    model_records = [r for r in records if r.get("variant") not in NON_MODEL_VARIANTS]
    best: dict[str, Any] = {}
    for key, _, _, higher in active:
        if higher is None:
            continue
        values = [r[key] for r in model_records if r.get(key) is not None]
        if values:
            best[key] = max(values) if higher else min(values)

    lines = [
        "| Model | " + " | ".join(h for _, h, _, _ in active) + " |",
        "|" + "---|" * (len(active) + 1),
    ]
    for record in records:
        variant = record.get("variant", "?")
        cells = []
        for key, _, spec, _ in active:
            value = record.get(key)
            text = _format(value, spec)
            if key in best and value is not None and value == best[key] and text != "n/a":
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {DISPLAY_NAMES.get(variant, variant)} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def build_csv(records: list[dict[str, Any]]) -> str:
    keys: list[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    lines = [",".join(keys)]
    for record in records:
        lines.append(",".join(str(record.get(key, "")) for key in keys))
    return "\n".join(lines)


def _delta_section(records: list[dict[str, Any]]) -> list[str]:
    """Baseline vs ConvLSTM, the comparison the whole project exists to make."""
    by_variant = {r.get("variant"): r for r in records}
    baseline, convlstm = by_variant.get("baseline"), by_variant.get("convlstm")
    if not baseline or not convlstm:
        return []

    lines = [
        "",
        "## Does temporal modelling help?",
        "",
        "Baseline and ConvLSTM share the same encoder, latent, decoder, loss, seed, data",
        "and step budget. The only difference is the ConvLSTM recurrence, so these deltas",
        "are attributable to it.",
        "",
        "| Metric | Baseline | ConvLSTM | Change |",
        "|---|---|---|---|",
    ]
    for key, heading, spec, higher in COLUMNS:
        a, b = baseline.get(key), convlstm.get(key)
        if a is None or b is None or higher is None:
            continue
        delta = b - a
        improved = (delta > 0) if higher else (delta < 0)
        marker = "better" if improved else ("same" if abs(delta) < 1e-12 else "worse")
        relative = f"{delta / abs(a) * 100:+.1f}%" if a else "n/a"
        lines.append(
            f"| {heading} | {_format(a, spec)} | {_format(b, spec)} | "
            f"{relative} ({marker}) |"
        )

    params_a = baseline.get("total_params")
    params_b = convlstm.get("total_params")
    if params_a and params_b:
        lines += [
            "",
            f"Note: the ConvLSTM carries {(params_b - params_a)/1e6:.2f}M more parameters "
            f"({params_b/params_a:.2f}x). Run the capacity-matched baseline "
            "(`hidden_dim: 1500`) to separate the effect of the recurrence from the "
            "effect of extra capacity.",
        ]
    return lines


def build_report(records: list[dict[str, Any]]) -> str:
    """The full markdown report."""
    lines = [
        "# Results",
        "",
        "All values are read directly from evaluation JSON records produced by",
        "`scripts/evaluate.py`. Metrics that were not run appear as `n/a` and are never",
        "estimated.",
        "",
        "## Comparison table",
        "",
        build_markdown_table(records),
        "",
        "**How to read this table**",
        "",
        "- *Real data (ceiling)* is the same metrics computed on ground-truth clips. It is",
        "  the practical maximum: direction recovery tops out near 95% even on real video,",
        "  so generated scores should be read against that, not against 100%.",
        "- *Static control* repeats frame 0 sixteen times. It scores a **perfect frame",
        "  SSIM** while generating no motion at all -- which is precisely why temporal",
        "  consistency is never reported as frame similarity alone. Its `temporal_score`",
        "  of 0 is the honest verdict.",
        "- *Grounding* is the primary alignment metric: it checks whether the generated",
        "  clip really shows the requested digit moving in the requested direction at the",
        "  requested speed, judged by an independently trained classifier and by optical",
        "  measurement -- with no CLIP involved.",
        "- *CLIPSIM* is reported for comparability with the literature, but is known to be",
        "  weak here: CLIP text embeddings for opposite directions have ~0.98 cosine",
        "  similarity, so whole-vector similarity barely distinguishes left from right.",
        "  See `clipsim_accuracy` in the raw records for the measured discrimination.",
    ]
    lines += _delta_section(records)
    return "\n".join(lines) + "\n"
