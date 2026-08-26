"""Horizon-stability evaluation: runs matched free/teacher-forced rollouts from held-out-
subject seed windows at increasing rollout lengths, and reports foot-skate, jerk,
distributional-drift, and free-vs-teacher-forced divergence, grouped by held-out subject.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import torch

from flowmotion.config import DEFAULT_ODE_STEPS, TARGET_FPS
from flowmotion.data.dataset import lookup_id
from flowmotion.data.loader import SequenceMeta, load_sequence, resample_to_fps
from flowmotion.data.transforms import Normalizer, features_from_numpy
from flowmotion.eval.metrics import (
    compute_reference_stats,
    distributional_drift,
    foot_skate,
    mean_squared_jerk,
    rollout_divergence,
    sequence_features_to_joint_positions,
)
from flowmotion.rollout import free_rollout, teacher_forced_rollout


@dataclass
class EvalConfig:
    rollout_lengths: list[int] = field(default_factory=lambda: [30, 60, 90, 150, 300])
    seeds_per_subject: int = 2
    ode_steps: int = DEFAULT_ODE_STEPS
    trailing_window: int = 30
    fps: float = TARGET_FPS
    seed_base: int = 0


def run_horizon_eval(
    model,
    normalizer: Normalizer,
    subject_vocab: dict[str, int],
    action_vocab: dict[str, int],
    held_out_sequences: list[SequenceMeta],
    cfg: EvalConfig,
) -> pd.DataFrame:
    model.eval()
    K, H = model.K, model.H

    # Load every held-out sequence's absolute features once; also used to build the
    # real-data reference statistics that distributional_drift measures against.
    loaded: list[tuple[SequenceMeta, torch.Tensor]] = []
    for meta in held_out_sequences:
        raw = resample_to_fps(load_sequence(meta), target_fps=cfg.fps)
        feat = features_from_numpy(raw.poses, raw.trans)
        loaded.append((meta, feat))

    real_joint_pos = [sequence_features_to_joint_positions(feat) for _, feat in loaded]
    ref_stats = compute_reference_stats(
        real_joint_pos, fps=cfg.fps, trailing_window=cfg.trailing_window
    )

    rows: list[dict] = []
    seed_counter = cfg.seed_base

    for meta, feat in loaded:
        T = feat.shape[0]
        subject_id = lookup_id(meta.subject_key, subject_vocab)
        action_id = lookup_id(meta.dataset_name, action_vocab)
        max_start = T - K

        for seed_idx in range(cfg.seeds_per_subject):
            start = seed_idx * K
            if start > max_start:
                break  # sequence too short for another non-overlapping seed window

            seed_past_abs = feat[start : start + K]

            for length in cfg.rollout_lengths:
                seed_counter += 1
                needed = start + K + length
                if needed > T:
                    rows.append(
                        {
                            "subject_id": meta.subject_key,
                            "seq_id": meta.path.stem,
                            "seed_idx": seed_idx,
                            "rollout_length": length,
                            "skipped": True,
                        }
                    )
                    continue

                gen_free = torch.Generator().manual_seed(seed_counter)
                gen_tf = torch.Generator().manual_seed(seed_counter)

                free = free_rollout(
                    model,
                    normalizer,
                    seed_past_abs,
                    subject_id,
                    action_id,
                    total_frames=length,
                    H=H,
                    steps=cfg.ode_steps,
                    generator=gen_free,
                )
                tf = teacher_forced_rollout(
                    model,
                    normalizer,
                    feat,
                    start,
                    subject_id,
                    action_id,
                    total_frames=length,
                    K=K,
                    H=H,
                    steps=cfg.ode_steps,
                    generator=gen_tf,
                )

                free_jp = sequence_features_to_joint_positions(free)
                tf_jp = sequence_features_to_joint_positions(tf)

                fs = foot_skate(free_jp, fps=cfg.fps)
                jerk = mean_squared_jerk(free_jp, fps=cfg.fps)
                drift = distributional_drift(
                    free_jp, fps=cfg.fps, ref_stats=ref_stats, trailing_window=cfg.trailing_window
                )
                divergence = rollout_divergence(free_jp, tf_jp)

                rows.append(
                    {
                        "subject_id": meta.subject_key,
                        "seq_id": meta.path.stem,
                        "seed_idx": seed_idx,
                        "rollout_length": length,
                        "skipped": False,
                        "foot_skate_mean": fs["mean_skate_per_contact_frame"],
                        "foot_skate_contact_frames": fs["contact_frame_count"],
                        "jerk_mean_sq": jerk,
                        "drift_speed_z": drift["speed_z"],
                        "drift_accel_z": drift["accel_z"],
                        "divergence_l2_mean": float(divergence.mean().item())
                        if divergence.numel()
                        else float("nan"),
                        "divergence_l2_final": float(divergence[-1].item())
                        if divergence.numel()
                        else float("nan"),
                    }
                )

    return pd.DataFrame(rows)


def aggregate_by_length(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "foot_skate_mean",
        "jerk_mean_sq",
        "drift_speed_z",
        "drift_accel_z",
        "divergence_l2_mean",
        "divergence_l2_final",
    ]
    evaluated = df[~df["skipped"]]
    return evaluated.groupby("rollout_length")[metric_cols].agg(["mean", "std"])


def aggregate_by_subject_and_length(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "foot_skate_mean",
        "jerk_mean_sq",
        "drift_speed_z",
        "drift_accel_z",
        "divergence_l2_mean",
        "divergence_l2_final",
    ]
    evaluated = df[~df["skipped"]]
    return evaluated.groupby(["subject_id", "rollout_length"])[metric_cols].mean()
