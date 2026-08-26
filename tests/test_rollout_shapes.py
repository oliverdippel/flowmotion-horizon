from __future__ import annotations

import torch

from flowmotion.data.transforms import Normalizer
from flowmotion.model.network import VelocityTransformer
from flowmotion.rollout import free_rollout, teacher_forced_rollout

D, K, H = 8, 3, 3


def _tiny_model():
    return VelocityTransformer(
        D=D,
        d_model=16,
        n_layers=1,
        n_heads=2,
        dim_ff=32,
        K=K,
        H=H,
        num_subjects=2,
        num_actions=2,
        dropout=0.0,
        p_label_dropout=0.0,
    )


def _identity_normalizer():
    return Normalizer(mean=torch.zeros(D), std=torch.ones(D))


def test_free_rollout_shape_matches_requested_length():
    model = _tiny_model().eval()
    normalizer = _identity_normalizer()
    seed_past = torch.randn(K, D)
    out = free_rollout(
        model, normalizer, seed_past, subject_id=0, action_id=0, total_frames=7, H=H, steps=4
    )
    assert out.shape == (7, D)


def test_free_rollout_is_reproducible_given_same_generator_seed():
    model = _tiny_model().eval()
    normalizer = _identity_normalizer()
    seed_past = torch.randn(K, D)
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    out1 = free_rollout(
        model, normalizer, seed_past, 0, 0, total_frames=6, H=H, steps=4, generator=gen1
    )
    out2 = free_rollout(
        model, normalizer, seed_past, 0, 0, total_frames=6, H=H, steps=4, generator=gen2
    )
    assert torch.allclose(out1, out2)


def test_teacher_forced_rollout_full_length_when_enough_real_data():
    model = _tiny_model().eval()
    normalizer = _identity_normalizer()
    real_sequence = torch.randn(20, D)
    out = teacher_forced_rollout(
        model,
        normalizer,
        real_sequence,
        seed_start_idx=0,
        subject_id=0,
        action_id=0,
        total_frames=9,
        K=K,
        H=H,
        steps=4,
    )
    assert out.shape == (9, D)


def test_teacher_forced_rollout_stops_early_when_real_data_runs_out():
    model = _tiny_model().eval()
    normalizer = _identity_normalizer()
    real_sequence = torch.randn(5, D)  # only enough real data for one window (K=3)
    out = teacher_forced_rollout(
        model,
        normalizer,
        real_sequence,
        seed_start_idx=0,
        subject_id=0,
        action_id=0,
        total_frames=9,
        K=K,
        H=H,
        steps=4,
    )
    assert out.shape[0] < 9
    assert out.shape[0] == H  # exactly one window's worth before running out
