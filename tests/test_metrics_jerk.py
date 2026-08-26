from __future__ import annotations

import pytest
import torch

from flowmotion.eval.metrics import mean_squared_jerk


def test_frozen_trajectory_has_zero_jerk():
    joint_pos = torch.zeros(10, 22, 3)
    joint_pos[:] = torch.rand(22, 3)  # same (arbitrary) pose every frame
    assert mean_squared_jerk(joint_pos, fps=20.0) == 0.0


def test_constant_velocity_has_zero_jerk():
    T = 10
    velocity = torch.tensor([0.05, 0.0, 0.0])
    base = torch.rand(22, 3)
    joint_pos = torch.stack([base + i * velocity for i in range(T)], dim=0)
    # exact 0 in theory; float32 accumulation over three finite differences leaves noise
    assert mean_squared_jerk(joint_pos, fps=20.0) == pytest.approx(0.0, abs=1e-6)


def test_constant_acceleration_has_zero_jerk():
    T = 10
    accel = torch.tensor([0.0, 0.02, 0.0])
    base = torch.rand(22, 3)
    joint_pos = torch.stack([base + 0.5 * accel * (i**2) for i in range(T)], dim=0)
    result = mean_squared_jerk(joint_pos, fps=20.0)
    assert result == pytest.approx(0.0, abs=1e-6)


def test_jerk_matches_manual_calculation_for_a_jolt():
    # constant position, except a single-frame jolt in the middle -> compute jerk by hand.
    fps = 10.0
    joint_pos = torch.zeros(6, 22, 3)
    joint_pos[3, 0, 0] = 1.0  # one joint jumps out and back

    vel = (joint_pos[1:] - joint_pos[:-1]) * fps
    acc = (vel[1:] - vel[:-1]) * fps
    jerk = (acc[1:] - acc[:-1]) * fps
    expected = float((jerk**2).mean().item())

    assert mean_squared_jerk(joint_pos, fps=fps) == expected


def test_too_short_sequence_returns_zero():
    joint_pos = torch.rand(2, 22, 3)
    assert mean_squared_jerk(joint_pos, fps=20.0) == 0.0
