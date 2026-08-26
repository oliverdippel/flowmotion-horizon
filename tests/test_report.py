from __future__ import annotations

import json

import pandas as pd

from flowmotion.eval.report import bootstrap_subject_ci, write_json_summary


def _fake_eval_df(subject_means: dict[str, float], rollout_length: int = 30) -> pd.DataFrame:
    rows = []
    for subject_id, mean in subject_means.items():
        for offset in (-0.1, 0.0, 0.1):  # a few "trials" per subject, not identical
            rows.append(
                {
                    "subject_id": subject_id,
                    "seq_id": "seq0",
                    "seed_idx": 0,
                    "rollout_length": rollout_length,
                    "skipped": False,
                    "foot_skate_mean": 0.0,
                    "foot_skate_contact_frames": 0,
                    "jerk_mean_sq": 0.0,
                    "drift_speed_z": 0.0,
                    "drift_accel_z": 0.0,
                    "divergence_l2_mean": mean + offset,
                    "divergence_l2_final": mean + offset,
                }
            )
    return pd.DataFrame(rows)


def test_bootstrap_ci_contains_the_sample_mean():
    df = _fake_eval_df({"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0, "s5": 5.0})
    result = bootstrap_subject_ci(df, "divergence_l2_mean", n_boot=2000, seed=0)
    lo, hi = result[30]
    sample_mean = df["divergence_l2_mean"].groupby(df["subject_id"]).mean().mean()
    assert lo < sample_mean < hi


def test_bootstrap_ci_is_deterministic_given_seed():
    df = _fake_eval_df({"s1": 1.0, "s2": 2.0, "s3": 3.0})
    a = bootstrap_subject_ci(df, "divergence_l2_mean", n_boot=500, seed=42)
    b = bootstrap_subject_ci(df, "divergence_l2_mean", n_boot=500, seed=42)
    assert a == b


def test_bootstrap_ci_nan_for_fewer_than_two_subjects():
    df = _fake_eval_df({"s1": 1.0})
    result = bootstrap_subject_ci(df, "divergence_l2_mean")
    lo, hi = result[30]
    assert lo != lo  # nan != nan
    assert hi != hi


def test_write_json_summary_includes_ci_fields(tmp_path):
    df = _fake_eval_df({"s1": 1.0, "s2": 2.0, "s3": 3.0})
    out_path = tmp_path / "summary.json"
    write_json_summary(df, out_path)

    summary = json.loads(out_path.read_text())
    entry = summary["30"]["divergence_l2_mean"]
    assert "ci_95_low" in entry
    assert "ci_95_high" in entry
    assert entry["n_subjects"] == 3
    assert entry["ci_95_low"] < entry["mean"] < entry["ci_95_high"]
