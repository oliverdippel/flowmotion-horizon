from __future__ import annotations

import torch

from flowmotion.data.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from flowmotion.data.transforms import (
    FEATURE_DIM,
    TRANS_START,
    yaw_align_window,
    yaw_unalign_window,
)


def _random_window(seed: int, T: int = 5) -> torch.Tensor:
    torch.manual_seed(seed)
    window = torch.zeros(T, FEATURE_DIM)
    # plausible (not necessarily orthonormal before conversion) rotation data for all
    # joints, including root, plus a translation with horizontal drift and varying height.
    aa = torch.randn(T, 22, 3) * 0.3
    d6 = matrix_to_rotation_6d(axis_angle_to_matrix(aa)).reshape(T, 22 * 6)
    window[:, : 22 * 6] = d6
    window[:, TRANS_START : TRANS_START + 2] = torch.randn(T, 2) * 2.0
    window[:, TRANS_START + 2] = 0.9 + torch.randn(T) * 0.05
    return window


def test_round_trip_recovers_original_window():
    window = _random_window(seed=0)
    aligned, yaw = yaw_align_window(window)
    recovered = yaw_unalign_window(aligned, yaw)
    assert torch.allclose(recovered, window, atol=1e-5)


def test_alignment_produces_a_canonical_heading():
    window = _random_window(seed=1)
    aligned, _ = yaw_align_window(window)
    root_mat = rotation_6d_to_matrix(aligned[0, 0:6])
    local_x_world = root_mat[:, 0]
    # local +X axis, projected onto the horizontal (x, y) plane, points along world +X.
    assert local_x_world[1].abs().item() < 1e-4
    assert local_x_world[0].item() > 0


def test_alignment_is_invariant_to_the_windows_initial_heading():
    # Two windows with IDENTICAL local pose content (all non-root joints, height, and
    # horizontal displacement pattern) but a different global heading at frame 0 --
    # after alignment they must become identical. This is the actual property that
    # justifies canonicalization: heading stops being a nuisance variable.
    window_a = _random_window(seed=2)

    extra_yaw = torch.tensor(0.83)  # an arbitrary additional rotation about Z
    cos, sin = torch.cos(extra_yaw), torch.sin(extra_yaw)
    extra_rot = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])

    window_b = window_a.clone()
    root_mats_a = rotation_6d_to_matrix(window_a[:, 0:6])
    window_b[:, 0:6] = matrix_to_rotation_6d(extra_rot @ root_mats_a)
    trans_xy_a = window_a[:, TRANS_START : TRANS_START + 2]
    window_b[:, TRANS_START : TRANS_START + 2] = trans_xy_a @ extra_rot[:2, :2].T

    aligned_a, _ = yaw_align_window(window_a)
    aligned_b, _ = yaw_align_window(window_b)
    assert torch.allclose(aligned_a, aligned_b, atol=1e-4)


def test_alignment_leaves_non_root_joints_untouched():
    window = _random_window(seed=3)
    aligned, _ = yaw_align_window(window)
    assert torch.allclose(window[:, 6:TRANS_START], aligned[:, 6:TRANS_START], atol=1e-6)
    assert torch.allclose(
        window[:, TRANS_START + 2], aligned[:, TRANS_START + 2], atol=1e-6
    )  # height (z) unchanged
