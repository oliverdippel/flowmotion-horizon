from __future__ import annotations

import torch

from flowmotion.data.rotation_conversions import (
    axis_angle_to_matrix,
    axis_angle_to_rotation_6d,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_axis_angle,
    rotation_6d_to_matrix,
)


def _random_axis_angle(*shape, max_angle=2.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    axis = torch.randn(*shape, 3, generator=g)
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1e-8)
    angle = torch.rand(*shape, 1, generator=g) * max_angle
    return axis * angle


def test_axis_angle_matrix_round_trip():
    aa = _random_axis_angle(8, 22, seed=1)
    r = axis_angle_to_matrix(aa)
    aa2 = matrix_to_axis_angle(r)
    r2 = axis_angle_to_matrix(aa2)
    assert torch.allclose(r, r2, atol=1e-4)


def test_matrices_are_orthonormal():
    aa = _random_axis_angle(5, 22, seed=2)
    r = axis_angle_to_matrix(aa)
    should_be_identity = r.transpose(-1, -2) @ r
    eye = torch.eye(3).expand_as(should_be_identity)
    assert torch.allclose(should_be_identity, eye, atol=1e-4)
    det = torch.linalg.det(r)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-4)


def test_6d_round_trip_recovers_original_rotation():
    aa = _random_axis_angle(6, 22, seed=3)
    r = axis_angle_to_matrix(aa)
    d6 = matrix_to_rotation_6d(r)
    r2 = rotation_6d_to_matrix(d6)
    assert torch.allclose(r, r2, atol=1e-4)


def test_6d_orthonormalizes_arbitrary_input():
    # rotation_6d_to_matrix must produce a valid rotation even from an unnormalized,
    # non-orthogonal 6D input (this is the whole point of Gram-Schmidt here).
    d6 = torch.randn(4, 6) * 3.0
    r = rotation_6d_to_matrix(d6)
    should_be_identity = r.transpose(-1, -2) @ r
    eye = torch.eye(3).expand_as(should_be_identity)
    assert torch.allclose(should_be_identity, eye, atol=1e-4)


def test_axis_angle_rotation_6d_convenience_round_trip():
    aa = _random_axis_angle(3, 22, seed=4)
    d6 = axis_angle_to_rotation_6d(aa)
    aa2 = rotation_6d_to_axis_angle(d6)
    r1 = axis_angle_to_matrix(aa)
    r2 = axis_angle_to_matrix(aa2)
    assert torch.allclose(r1, r2, atol=1e-4)


def test_zero_rotation_round_trips_to_zero():
    aa = torch.zeros(3, 3)
    r = axis_angle_to_matrix(aa)
    assert torch.allclose(r, torch.eye(3).expand(3, 3, 3), atol=1e-6)
    aa2 = matrix_to_axis_angle(r)
    assert torch.allclose(aa2, torch.zeros(3, 3), atol=1e-6)
