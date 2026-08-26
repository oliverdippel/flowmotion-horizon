"""Rotation representation conversions: axis-angle <-> rotation matrix <-> 6D.

The 6D continuous rotation representation (Zhou et al., "On the Continuity of
Rotation Representations in Neural Networks") is used as the network's I/O
space because axis-angle has periodicity/sign discontinuities near +-pi that
make it a poor regression target for a velocity field.

All functions are batched: leading dims are arbitrary, the rotation itself
occupies the trailing dim(s).
"""

from __future__ import annotations

import torch

_EPS = 1e-8


def _skew_symmetric(axis: torch.Tensor) -> torch.Tensor:
    """axis: (..., 3) unit vectors -> (..., 3, 3) skew-symmetric matrices."""
    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    row0 = torch.stack([zeros, -z, y], dim=-1)
    row1 = torch.stack([z, zeros, -x], dim=-1)
    row2 = torch.stack([-y, x, zeros], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrix, via Rodrigues' formula."""
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)  # (..., 1)
    axis = aa / torch.clamp(theta, min=_EPS)
    k = _skew_symmetric(axis)
    theta_ = theta.unsqueeze(-1)  # (..., 1, 1)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(*aa.shape[:-1], 3, 3)
    r = eye + torch.sin(theta_) * k + (1.0 - torch.cos(theta_)) * (k @ k)
    return r


def matrix_to_axis_angle(r: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrix -> (..., 3) axis-angle."""
    trace = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)  # (...,)
    sin_theta = torch.sin(theta)

    vec = torch.stack(
        [
            r[..., 2, 1] - r[..., 1, 2],
            r[..., 0, 2] - r[..., 2, 0],
            r[..., 1, 0] - r[..., 0, 1],
        ],
        dim=-1,
    )
    # away from theta ~ 0, axis = vec / (2 sin theta); near 0, angle -> 0 so aa -> 0 regardless.
    denom = torch.clamp(2.0 * sin_theta, min=_EPS).unsqueeze(-1)
    axis = vec / denom
    aa = axis * theta.unsqueeze(-1)
    # where theta is ~0, the above is numerically unstable (0/eps) but theta~0 means aa~0 anyway.
    small = (theta.unsqueeze(-1) < 1e-6).expand_as(aa)
    aa = torch.where(small, torch.zeros_like(aa), aa)
    return aa


def matrix_to_rotation_6d(r: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrix -> (..., 6): first two columns, flattened."""
    return torch.cat([r[..., :, 0], r[..., :, 1]], dim=-1)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt orthonormalization of the first two columns."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / torch.clamp(torch.linalg.norm(a1, dim=-1, keepdim=True), min=_EPS)
    a2_proj = (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 - a2_proj
    b2 = b2 / torch.clamp(torch.linalg.norm(b2, dim=-1, keepdim=True), min=_EPS)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def axis_angle_to_rotation_6d(aa: torch.Tensor) -> torch.Tensor:
    return matrix_to_rotation_6d(axis_angle_to_matrix(aa))


def rotation_6d_to_axis_angle(d6: torch.Tensor) -> torch.Tensor:
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))
