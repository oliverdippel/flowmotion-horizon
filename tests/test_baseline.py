from __future__ import annotations

import torch

from flowmotion.baseline import ZeroVelocityRoller

K, H, D = 3, 3, 5


def test_free_repeats_last_seed_frame():
    roller = ZeroVelocityRoller(K=K, H=H)
    seed_past = torch.arange(K * D, dtype=torch.float32).reshape(K, D)
    out = roller.free(seed_past, subject_id=0, action_id=0, total_frames=7, generator=None)
    assert out.shape == (7, D)
    assert torch.allclose(out, seed_past[-1:].expand(7, D))


def test_teacher_forced_shape_matches_requested_length():
    roller = ZeroVelocityRoller(K=K, H=H)
    real_sequence = torch.randn(20, D)
    out = roller.teacher_forced(
        real_sequence, seed_start_idx=0, subject_id=0, action_id=0, total_frames=9, generator=None
    )
    assert out.shape == (9, D)


def test_teacher_forced_stops_early_when_real_data_runs_out():
    roller = ZeroVelocityRoller(K=K, H=H)
    real_sequence = torch.randn(5, D)  # only enough real data for one window (K=3)
    out = roller.teacher_forced(
        real_sequence, seed_start_idx=0, subject_id=0, action_id=0, total_frames=9, generator=None
    )
    assert out.shape[0] < 9
    assert out.shape[0] == H  # exactly one window's worth before running out


def test_teacher_forced_tracks_real_last_frame_each_window():
    """Unlike `free`, teacher-forced re-reads the real window each step, so its output
    changes as the real sequence advances -- this is what makes free-vs-teacher-forced
    divergence nontrivial even for a baseline that produces no motion of its own."""
    roller = ZeroVelocityRoller(K=K, H=H)
    real_sequence = torch.arange(12 * D, dtype=torch.float32).reshape(12, D)
    out = roller.teacher_forced(
        real_sequence, seed_start_idx=0, subject_id=0, action_id=0, total_frames=6, generator=None
    )
    first_window_val = real_sequence[K - 1]
    second_window_val = real_sequence[H + K - 1]
    assert torch.allclose(out[:H], first_window_val.expand(H, D))
    assert torch.allclose(out[H:], second_window_val.expand(H, D))
