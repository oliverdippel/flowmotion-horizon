"""Pose <-> network-feature encoding, and per-channel z-score normalization.

Network feature layout (D = 135): 22 joints x 6D rotation (132) + root translation xyz (3).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from flowmotion.data.rotation_conversions import (
    axis_angle_to_rotation_6d,
    matrix_to_rotation_6d,
    rotation_6d_to_axis_angle,
    rotation_6d_to_matrix,
)
from flowmotion.data.skeleton import NUM_JOINTS

FEATURE_DIM = NUM_JOINTS * 6 + 3
TRANS_START = FEATURE_DIM - 3


def pose_to_feature(poses_aa: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """poses_aa: (..., 22, 3) axis-angle, trans: (..., 3) -> (..., 135) feature."""
    d6 = axis_angle_to_rotation_6d(poses_aa)  # (..., 22, 6)
    flat = d6.reshape(*d6.shape[:-2], NUM_JOINTS * 6)
    return torch.cat([flat, trans], dim=-1)


def feature_to_pose(feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., 135) -> (poses_aa (..., 22, 3), trans (..., 3))."""
    rot_flat, trans = feature[..., : NUM_JOINTS * 6], feature[..., NUM_JOINTS * 6 :]
    d6 = rot_flat.reshape(*rot_flat.shape[:-1], NUM_JOINTS, 6)
    poses_aa = rotation_6d_to_axis_angle(d6)
    return poses_aa, trans


def recenter_xy(trans: torch.Tensor, reference_xy: torch.Tensor) -> torch.Tensor:
    """Subtract a reference (x, y) from a translation tensor's first two channels.

    trans: (..., T, 3), reference_xy: (..., 2) -- broadcasts over the T axis.
    """
    ref = torch.zeros_like(trans[..., 0, :])
    ref[..., :2] = reference_xy
    return trans - ref.unsqueeze(-2)


def _root_yaw(root_d6: torch.Tensor) -> torch.Tensor:
    """root_d6: (6,) root joint's 6D rotation -> scalar yaw (radians) about the (Z) up
    axis, taken from where the root's local +X axis points, projected onto the
    horizontal plane. Any fixed local axis works here -- this is used purely to define
    a canonical heading for alignment, not to recover a physically meaningful "facing
    direction" (this codebase has no ground truth for that without a real body model)."""
    r = rotation_6d_to_matrix(root_d6)
    local_x_world = r[:, 0]
    return torch.atan2(local_x_world[1], local_x_world[0])


def _yaw_matrix(yaw: torch.Tensor) -> torch.Tensor:
    """scalar yaw -> (3, 3) rotation matrix about the Z axis."""
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack(
        [
            torch.stack([cos, -sin, zero]),
            torch.stack([sin, cos, zero]),
            torch.stack([zero, zero, one]),
        ]
    )


def yaw_align_window(
    window: torch.Tensor, reference_frame: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """window: (T, D) feature tensor -> (aligned_window, yaw). Rotates the whole window
    about the up axis so that `reference_frame`'s root faces a canonical heading. Only
    the root's 6D block (indices [0:6]) and translation x, y change: every other
    joint's rotation is parent-relative in the kinematic chain and is unaffected by a
    rigid rotation of the whole body about its vertical axis (see skeleton.py). `yaw`
    is returned so the transform can be inverted exactly with `yaw_unalign_window`.
    """
    yaw = _root_yaw(window[reference_frame, 0:6])
    align = _yaw_matrix(-yaw)
    return _apply_yaw(window, align), yaw


def yaw_unalign_window(window: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Inverse of `yaw_align_window` given the `yaw` it returned."""
    return _apply_yaw(window, _yaw_matrix(yaw))


def _apply_yaw(window: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    out = window.clone()
    root_mats = rotation_6d_to_matrix(window[:, 0:6])  # (T, 3, 3)
    out[:, 0:6] = matrix_to_rotation_6d(rot @ root_mats)
    trans_xy = window[:, TRANS_START : TRANS_START + 2]  # (T, 2)
    out[:, TRANS_START : TRANS_START + 2] = trans_xy @ rot[:2, :2].T
    return out


@dataclass
class Normalizer:
    mean: torch.Tensor  # (D,)
    std: torch.Tensor  # (D,)

    @classmethod
    def fit(cls, features: torch.Tensor, eps: float = 1e-2) -> Normalizer:
        """features: (N, D) -- statistics computed over the leading (sample) dim only.

        `eps` floors std, not just to avoid division by exactly zero but to cap how much
        a near-frozen channel gets amplified. Verified on real AMASS data: spine joints
        barely rotate across the corpus, giving some 6D-rotation channels a genuine std
        as low as ~1e-6 -- a machine-epsilon-sized floor would let normalization divide
        by that and blow tiny floating-point noise up into huge normalized targets
        (this destabilized an early real training run into million-scale loss). eps is
        in feature units (rotation-6D components and translation meters are both O(1)),
        so 1e-2 is "don't bother resolving variation smaller than this," not a numerical
        safety valve.
        """
        mean = features.mean(dim=0)
        std = features.std(dim=0).clamp(min=eps)
        return cls(mean=mean, std=std)

    @classmethod
    def fit_streaming(cls, windows: Iterable[torch.Tensor], eps: float = 1e-2) -> Normalizer:
        """Same statistics as `fit`, but consumes an iterable of (T, D) tensors one at a
        time (e.g. `MotionWindowDataset.iter_recentered_windows()`) via running sum/sum-
        of-squares, so the whole corpus is never materialized as one in-memory tensor."""
        count = 0
        sum_: torch.Tensor | None = None
        sumsq: torch.Tensor | None = None
        for window in windows:
            flat = window.reshape(-1, window.shape[-1])
            if sum_ is None or sumsq is None:
                sum_ = flat.sum(dim=0).clone()
                sumsq = (flat**2).sum(dim=0).clone()
            else:
                sum_ += flat.sum(dim=0)
                sumsq += (flat**2).sum(dim=0)
            count += flat.shape[0]

        if count == 0 or sum_ is None or sumsq is None:
            raise ValueError("fit_streaming received no windows to fit on")

        mean = sum_ / count
        var = (sumsq / count - mean**2).clamp(min=0.0)
        std = var.sqrt().clamp(min=eps)
        return cls(mean=mean, std=std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def to_state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict) -> Normalizer:
        return cls(mean=state["mean"], std=state["std"])


def features_from_numpy(poses: np.ndarray, trans: np.ndarray) -> torch.Tensor:
    """Convenience: RawSequence-style numpy arrays -> (T, 135) feature tensor."""
    poses_t = torch.from_numpy(poses.astype(np.float32))
    trans_t = torch.from_numpy(trans.astype(np.float32))
    return pose_to_feature(poses_t, trans_t)
