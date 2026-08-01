"""Assemble the final comparison table from evaluation records.

    python scripts/build_report.py

Reads outputs/eval/*.json and writes RESULTS.md plus outputs/eval/comparison.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text2video.evaluation.report import build_csv, build_report, load_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="outputs/eval")
    parser.add_argument("--out", default="RESULTS.md")
    args = parser.parse_args()

    eval_dir = PROJECT_ROOT / args.eval_dir
    records = load_records(eval_dir)
    if not records:
        print(f"No evaluation records in {eval_dir}. Run scripts/evaluate.py --all first.")
        return 1

    report = build_report(records)
    out_path = PROJECT_ROOT / args.out
    out_path.write_text(report, encoding="utf-8")

    csv_path = eval_dir / "comparison.csv"
    csv_path.write_text(build_csv(records), encoding="utf-8")

    print(report)
    print(f"\nwrote {out_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
