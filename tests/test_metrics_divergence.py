from __future__ import annotations

import torch

from flowmotion.eval.metrics import rollout_divergence


def test_identical_inputs_give_zero_divergence():
    joint_pos = torch.rand(20, 22, 3)
    divergence = rollout_divergence(joint_pos, joint_pos.clone())
    assert torch.allclose(divergence, torch.zeros_like(divergence), atol=1e-6)


def test_divergence_matches_manual_l2():
    free = torch.rand(15, 22, 3)
    tf = torch.rand(15, 22, 3)
    divergence = rollout_divergence(free, tf)

    expected = torch.linalg.norm((free - tf).reshape(15, -1), dim=-1)
    assert torch.allclose(divergence, expected)


def test_divergence_truncates_to_shorter_input():
    free = torch.rand(20, 22, 3)
    tf = torch.rand(12, 22, 3)
    divergence = rollout_divergence(free, tf)
    assert divergence.shape[0] == 12
