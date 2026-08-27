"""Overlays horizon-eval CSVs from multiple runs on one set of plots, and reports
whether their 95% confidence intervals overlap at each rollout length -- the basis
for actually claiming two runs' metrics do or don't differ, rather than eyeballing
point estimates.

Not part of the `flowmotion` package: this is a standalone analysis script for
comparing runs against each other (e.g. a fully-trained model vs. a deliberately
undertrained one), not something the training/eval pipeline itself needs.

Usage:
    uv run python scripts/compare_runs.py \
        --run "well-trained=runs/real_v2/eval_report_v2/horizon_eval.csv" \
        --run "undertrained=runs/ablation_undertrained/eval_report_v2/horizon_eval.csv" \
        --out assets/ablation_comparison.png --report-out assets/ablation_comparison.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from flowmotion.eval.report import METRIC_COLS, PLOT_METRICS, PLOT_TITLES, bootstrap_subject_ci


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


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    a_lo, a_hi = a
    b_lo, b_hi = b
    if a_lo != a_lo or b_lo != b_lo:  # nan
        return True  # can't claim a difference without a defined interval
    return not (a_hi < b_lo or b_hi < a_lo)


def build_comparison_report(
    runs: list[tuple[str, pd.DataFrame]], metric_cols: list[str] | None = None
) -> str:
    """For each metric and rollout length, each run's mean + 95% bootstrap CI (over
    held-out subjects), and whether the two runs' CIs overlap -- overlap means the data
    doesn't support claiming the runs differ on that metric at that length, non-overlap
    means it does."""
    metric_cols = metric_cols or METRIC_COLS
    if len(runs) != 2:
        raise ValueError("overlap reporting is defined for exactly 2 runs")
    (label_a, df_a), (label_b, df_b) = runs

    lines = [f"# Comparison: {label_a} vs. {label_b}", ""]
    for metric in metric_cols:
        ci_a = bootstrap_subject_ci(df_a, metric)
        ci_b = bootstrap_subject_ci(df_b, metric)
        lengths = sorted(set(ci_a) & set(ci_b))
        if not lengths:
            continue
        lines.append(f"## {metric}")
        lines.append(f"| length | {label_a} | {label_b} | 95% CIs overlap |")
        lines.append("|---|---|---|---|")
        for length in lengths:
            mean_a = df_a.loc[df_a["rollout_length"] == length, metric].mean()
            mean_b = df_b.loc[df_b["rollout_length"] == length, metric].mean()
            overlap = _intervals_overlap(ci_a[length], ci_b[length])
            lines.append(
                f"| {length} | {mean_a:.4g} [{ci_a[length][0]:.4g}, {ci_a[length][1]:.4g}] "
                f"| {mean_b:.4g} [{ci_b[length][0]:.4g}, {ci_b[length][1]:.4g}] "
                f"| {'yes' if overlap else '**no**'} |"
            )
        lines.append("")
    return "\n".join(lines)


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
    parser.add_argument("--report-out", default=None)
    args = parser.parse_args()

    runs = [_load_run(spec) for spec in args.runs]
    fig = build_comparison_figure(runs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote comparison plot to {out_path}")

    if len(runs) == 2:
        report = build_comparison_report(runs)
        print(report)
        if args.report_out:
            report_path = Path(args.report_out)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report)
            print(f"wrote comparison report to {report_path}")


if __name__ == "__main__":
    main()
