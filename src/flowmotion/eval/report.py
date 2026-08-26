"""Writes the horizon-eval harness's output as CSV/JSON, and the headline plot: each
metric vs. rollout length, aggregate mean line with a cross-subject std band -- this is
the artifact for showing partners where the lines start to bend."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flowmotion.eval.harness import aggregate_by_length, aggregate_by_subject_and_length

METRIC_COLS = [
    "foot_skate_mean",
    "jerk_mean_sq",
    "drift_speed_z",
    "drift_accel_z",
    "divergence_l2_mean",
    "divergence_l2_final",
]

PLOT_METRICS = ["foot_skate_mean", "jerk_mean_sq", "divergence_l2_mean", "drift_speed_z"]
PLOT_TITLES = {
    "foot_skate_mean": "foot skate (m / contact frame)",
    "jerk_mean_sq": "mean squared jerk",
    "divergence_l2_mean": "free vs. teacher-forced divergence (L2)",
    "drift_speed_z": "speed distributional drift (z-score)",
}


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False)


def write_json_summary(
    df: pd.DataFrame, path: str | Path, metric_cols: list[str] | None = None
) -> None:
    metric_cols = metric_cols or METRIC_COLS
    evaluated = df[~df["skipped"]]
    if evaluated.empty:
        Path(path).write_text(json.dumps({}, indent=2))
        return

    agg = aggregate_by_length(df)
    per_subj = aggregate_by_subject_and_length(df)

    summary: dict = {}
    for length in sorted(evaluated["rollout_length"].unique().tolist()):
        length_key = str(int(length))
        summary[length_key] = {}
        for metric in metric_cols:
            mean_val = float(agg.loc[length, (metric, "mean")])
            std_raw = agg.loc[length, (metric, "std")]
            std_val = 0.0 if pd.isna(std_raw) else float(std_raw)
            per_subject_vals = {}
            for subject_id in per_subj.index.get_level_values("subject_id").unique():
                if (subject_id, length) in per_subj.index:
                    per_subject_vals[subject_id] = float(per_subj.loc[(subject_id, length), metric])
            summary[length_key][metric] = {
                "mean": mean_val,
                "std": std_val,
                "per_subject": per_subject_vals,
            }

    Path(path).write_text(json.dumps(summary, indent=2))


def plot_horizon_curves(df: pd.DataFrame, out_png_path: str | Path) -> None:
    evaluated = df[~df["skipped"]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    if evaluated.empty:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "no evaluated rollouts", ha="center", va="center")
        fig.savefig(out_png_path, dpi=150)
        plt.close(fig)
        return

    lengths = sorted(evaluated["rollout_length"].unique().tolist())
    for ax, metric in zip(axes.flat, PLOT_METRICS):
        means, stds = [], []
        for length in lengths:
            vals = evaluated.loc[evaluated["rollout_length"] == length, metric]
            means.append(vals.mean())
            stds.append(vals.std() if len(vals) > 1 else 0.0)
        means_arr, stds_arr = np.array(means), np.array(stds)
        ax.plot(lengths, means_arr, marker="o")
        ax.fill_between(lengths, means_arr - stds_arr, means_arr + stds_arr, alpha=0.2)
        ax.set_xlabel("rollout length (frames)")
        ax.set_title(PLOT_TITLES.get(metric, metric))

    fig.tight_layout()
    fig.savefig(out_png_path, dpi=150)
    plt.close(fig)
