"""Horizon-stability metrics: foot skate, jerk, distributional drift, free-vs-teacher-forced
divergence. Each takes joint positions (T, 22, 3) in world space, produced via forward
kinematics from network features.
"""

from __future__ import annotations

import torch

from flowmotion.data.rotation_conversions import axis_angle_to_matrix
from flowmotion.data.skeleton import LEFT_FOOT_IDX, RIGHT_FOOT_IDX, UP_AXIS_IDX, forward_kinematics
from flowmotion.data.transforms import feature_to_pose


def sequence_features_to_joint_positions(feat: torch.Tensor) -> torch.Tensor:
    """(T, D) absolute network features -> (T, 22, 3) world joint positions."""
    poses_aa, trans = feature_to_pose(feat)
    rotmats = axis_angle_to_matrix(poses_aa)
    return forward_kinematics(rotmats, trans)


def estimate_foot_floor(
    reference_joint_pos: list[torch.Tensor],
    up_axis_idx: int = UP_AXIS_IDX,
    foot_idxs: tuple[int, int] = (LEFT_FOOT_IDX, RIGHT_FOOT_IDX),
    percentile: float = 1.0,
) -> dict:
    """Per-foot floor height, estimated as a low percentile of that joint's height across
    real reference sequences.

    This codebase has no real SMPL body model (see skeleton.py), so there is no guarantee
    that world-space height 0 corresponds to actual ground contact under the approximate
    rig -- on real AMASS data it does not (verified: even ground-truth sequences never
    bring a foot below ~0.5-0.6 under this rig's FK). Calibrating "floor" from real data's
    own low-percentile height is a data-driven stand-in that doesn't require a real body
    model, and `foot_skate` falls back to an absolute floor of 0 when `floor=None` (e.g.
    for the synthetic fixture, where 0 genuinely is ground level by construction)."""
    floor = {}
    for name, idx in zip(("left", "right"), foot_idxs):
        heights = torch.cat(
            [jp[:, idx, up_axis_idx] for jp in reference_joint_pos if jp.shape[0] > 0]
        )
        floor[name] = float(torch.quantile(heights, percentile / 100.0).item())
    return floor


def foot_skate(
    joint_pos: torch.Tensor,
    fps: float,
    up_axis_idx: int = UP_AXIS_IDX,
    height_thresh: float = 0.05,
    vel_thresh: float = 0.15,
    foot_idxs: tuple[int, int] = (LEFT_FOOT_IDX, RIGHT_FOOT_IDX),
    floor: dict | None = None,
) -> dict:
    """Mean horizontal displacement of a foot joint during frames classified as
    ground-contact (low height above `floor` + low vertical velocity). `floor` is a
    per-foot world-space height from `estimate_foot_floor`; without one, height is
    measured against an absolute floor of 0. `contact_frame_count` is surfaced
    separately: a foot that never touches the ground reports 0 skate, which must not
    be read as "good" -- it means the model produced no evaluable contact."""
    horiz_idxs = [i for i in range(3) if i != up_axis_idx]
    T = joint_pos.shape[0]
    per_foot = {}
    total_skate = 0.0
    total_contact = 0

    for name, idx in zip(("left", "right"), foot_idxs):
        pos = joint_pos[:, idx, :]
        height = pos[:, up_axis_idx]
        floor_height = floor[name] if floor is not None else 0.0
        vel_up = torch.zeros(T, dtype=pos.dtype)
        vel_up[1:] = (height[1:] - height[:-1]) * fps
        disp = torch.zeros(T, dtype=pos.dtype)
        horiz = pos[:, horiz_idxs]
        disp[1:] = torch.linalg.norm(horiz[1:] - horiz[:-1], dim=-1)

        contact = (height < floor_height + height_thresh) & (vel_up.abs() < vel_thresh)
        contact[0] = False  # frame 0 has no valid velocity/displacement reference

        skate_vals = disp[contact]
        n_contact = int(contact.sum().item())
        skate_sum = float(skate_vals.sum().item())
        per_foot[name] = {
            "mean_skate": skate_sum / n_contact if n_contact > 0 else 0.0,
            "contact_frames": n_contact,
            "total_skate": skate_sum,
        }
        total_skate += skate_sum
        total_contact += n_contact

    return {
        "mean_skate_per_contact_frame": total_skate / total_contact if total_contact > 0 else 0.0,
        "total_skate": total_skate,
        "contact_frame_count": total_contact,
        "per_foot": per_foot,
    }


def mean_squared_jerk(joint_pos: torch.Tensor, fps: float) -> float:
    """Mean squared third derivative of joint position -- a smoothness/physical-
    plausibility proxy; high-frequency jitter from a collapsing rollout shows up here."""
    if joint_pos.shape[0] < 4:
        return 0.0
    vel = (joint_pos[1:] - joint_pos[:-1]) * fps
    acc = (vel[1:] - vel[:-1]) * fps
    jerk = (acc[1:] - acc[:-1]) * fps
    return float((jerk**2).mean().item())


def _speed_accel_stats(
    joint_pos: torch.Tensor, fps: float, trailing_window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    window = joint_pos[-trailing_window:] if joint_pos.shape[0] >= trailing_window else joint_pos
    if window.shape[0] < 2:
        return torch.zeros(0), torch.zeros(0)
    vel = (window[1:] - window[:-1]) * fps
    speed = torch.linalg.norm(vel, dim=-1).reshape(-1)
    if window.shape[0] < 3:
        return speed, torch.zeros(0)
    acc = (vel[1:] - vel[:-1]) * fps
    accel = torch.linalg.norm(acc, dim=-1).reshape(-1)
    return speed, accel


def compute_reference_stats(
    held_out_sequences: list[torch.Tensor], fps: float, trailing_window: int = 30
) -> dict:
    """Pooled speed/accel mean+std over real held-out sequences, for `distributional_drift`
    to measure z-score deviation against."""
    all_speed, all_accel = [], []
    for jp in held_out_sequences:
        speed, accel = _speed_accel_stats(jp, fps, trailing_window)
        if speed.numel() > 0:
            all_speed.append(speed)
        if accel.numel() > 0:
            all_accel.append(accel)
    speed_cat = torch.cat(all_speed) if all_speed else torch.zeros(0)
    accel_cat = torch.cat(all_accel) if all_accel else torch.zeros(0)
    return {
        "speed_mean": float(speed_cat.mean().item()) if speed_cat.numel() > 0 else 0.0,
        "speed_std": float(speed_cat.std().item()) if speed_cat.numel() > 1 else 1e-6,
        "accel_mean": float(accel_cat.mean().item()) if accel_cat.numel() > 0 else 0.0,
        "accel_std": float(accel_cat.std().item()) if accel_cat.numel() > 1 else 1e-6,
    }


def distributional_drift(
    joint_pos: torch.Tensor, fps: float, ref_stats: dict, trailing_window: int = 30
) -> dict:
    """Z-score deviation of the rollout's TRAILING window's speed/accel from real held-out
    reference statistics -- using the trailing window (not the whole rollout) so a late
    collapse isn't diluted by a well-behaved seed."""
    speed, accel = _speed_accel_stats(joint_pos, fps, trailing_window)
    speed_mean = float(speed.mean().item()) if speed.numel() > 0 else 0.0
    accel_mean = float(accel.mean().item()) if accel.numel() > 0 else 0.0
    speed_z = abs(speed_mean - ref_stats["speed_mean"]) / max(ref_stats["speed_std"], 1e-6)
    accel_z = abs(accel_mean - ref_stats["accel_mean"]) / max(ref_stats["accel_std"], 1e-6)
    return {
        "speed_mean": speed_mean,
        "accel_mean": accel_mean,
        "speed_z": speed_z,
        "accel_z": accel_z,
    }


def rollout_divergence(free_joint_pos: torch.Tensor, tf_joint_pos: torch.Tensor) -> torch.Tensor:
    """Per-frame L2 divergence in world joint-position space between a free rollout and a
    teacher-forced rollout from the same seed/RNG -- the primary "does it collapse" signal:
    the only difference between the two is whether the model conditions on its own output."""
    L = min(free_joint_pos.shape[0], tf_joint_pos.shape[0])
    diff = (free_joint_pos[:L] - tf_joint_pos[:L]).reshape(L, -1)
    return torch.linalg.norm(diff, dim=-1)
