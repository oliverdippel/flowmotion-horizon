"""Overlays horizon-eval CSVs from multiple runs on one set of plots.

Not part of the `flowmotion` package: this is a standalone analysis script for
comparing runs against each other (e.g. a fully-trained model vs. a deliberately
undertrained one), not something the training/eval pipeline itself needs.

Usage:
    uv run python scripts/compare_runs.py \
        --run "well-trained (20K steps)=runs/real_v2/eval_report/horizon_eval.csv" \
        --run "undertrained (1.5K steps)=runs/ablation_undertrained/eval_report/horizon_eval.csv" \
        --out assets/ablation_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from flowmotion.eval.report import PLOT_METRICS, PLOT_TITLES


def _load_run(spec: str) -> tuple[str, pd.DataFrame]:
    label, _, path = spec.partition("=")
    if not path:
        raise ValueError(f"expected 'label=path/to/horizon_eval.csv', got {spec!r}")
    df = pd.read_csv(path)
    return label, df[~df["skipped"]]


def build_comparison_figure(runs: list[tuple[str, pd.DataFrame]]):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, metric in zip(axes.flat, PLOT_METRICS):
        for label, df in runs:
            lengths = sorted(df["rollout_length"].unique().tolist())
            means = [df.loc[df["rollout_length"] == length, metric].mean() for length in lengths]
            ax.plot(lengths, means, marker="o", label=label)
        ax.set_xlabel("rollout length (frames)")
        ax.set_title(PLOT_TITLES.get(metric, metric))
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        dest="runs",
        help="label=path/to/horizon_eval.csv (repeatable)",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs = [_load_run(spec) for spec in args.runs]
    fig = build_comparison_figure(runs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote comparison plot to {out_path}")


if __name__ == "__main__":
    main()
