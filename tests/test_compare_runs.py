"""Tests scripts/compare_runs.py's statistics -- it's not part of the installed
package, but the CI-overlap logic is real statistical reasoning worth covering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from compare_runs import _intervals_overlap, build_comparison_report  # noqa: E402


def test_intervals_overlap_true_cases():
    assert _intervals_overlap((0.0, 2.0), (1.0, 3.0))  # partial overlap
    assert _intervals_overlap((0.0, 5.0), (1.0, 2.0))  # one contains the other
    assert _intervals_overlap((0.0, 1.0), (1.0, 2.0))  # touching at the boundary


def test_intervals_overlap_false_case():
    assert not _intervals_overlap((0.0, 1.0), (2.0, 3.0))


def test_intervals_overlap_nan_defaults_to_overlap():
    nan = float("nan")
    assert _intervals_overlap((nan, nan), (0.0, 1.0))


def _fake_df(subject_means: dict[str, float], rollout_length: int = 30) -> pd.DataFrame:
    rows = []
    for subject_id, mean in subject_means.items():
        for offset in (-0.05, 0.0, 0.05):
            rows.append(
                {
                    "subject_id": subject_id,
                    "rollout_length": rollout_length,
                    "skipped": False,
                    "divergence_l2_mean": mean + offset,
                    "divergence_l2_final": mean + offset,
                    "jerk_mean_sq": 0.0,
                    "foot_skate_mean": 0.0,
                    "drift_speed_z": 0.0,
                    "drift_accel_z": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_build_comparison_report_flags_a_clear_difference():
    low = _fake_df({"s1": 1.0, "s2": 1.1, "s3": 0.9, "s4": 1.05, "s5": 0.95})
    high = _fake_df({"s1": 10.0, "s2": 10.1, "s3": 9.9, "s4": 10.05, "s5": 9.95})
    report = build_comparison_report(
        [("low", low), ("high", high)], metric_cols=["divergence_l2_mean"]
    )
    assert "**no**" in report  # CIs should not overlap given the large, tight separation


def test_build_comparison_report_does_not_flag_indistinguishable_runs():
    a = _fake_df({"s1": 1.0, "s2": 1.1, "s3": 0.9, "s4": 1.05, "s5": 0.95})
    b = _fake_df({"s1": 1.02, "s2": 1.08, "s3": 0.93, "s4": 1.0, "s5": 0.98})
    report = build_comparison_report([("a", a), ("b", b)], metric_cols=["divergence_l2_mean"])
    assert "**no**" not in report
