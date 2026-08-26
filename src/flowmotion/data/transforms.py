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
    rotation_6d_to_axis_angle,
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


@dataclass
class Normalizer:
    mean: torch.Tensor  # (D,)
    std: torch.Tensor  # (D,)

    @classmethod
    def fit(cls, features: torch.Tensor, eps: float = 1e-6) -> Normalizer:
        """features: (N, D) -- statistics computed over the leading (sample) dim only."""
        mean = features.mean(dim=0)
        std = features.std(dim=0).clamp(min=eps)
        return cls(mean=mean, std=std)

    @classmethod
    def fit_streaming(cls, windows: Iterable[torch.Tensor], eps: float = 1e-6) -> Normalizer:
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
