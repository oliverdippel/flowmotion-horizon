from __future__ import annotations

import torch

from flowmotion.viz import render_skeleton_comparison_gif


def test_render_skeleton_comparison_gif_writes_a_nonempty_file(tmp_path):
    T = 4
    seq_a = torch.rand(T, 22, 3)
    seq_b = torch.rand(T, 22, 3)
    out_path = tmp_path / "rollout.gif"

    render_skeleton_comparison_gif(seq_a, seq_b, labels=("a", "b"), fps=10.0, out_path=out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_render_skeleton_comparison_gif_truncates_to_shorter_sequence(tmp_path):
    seq_a = torch.rand(6, 22, 3)
    seq_b = torch.rand(3, 22, 3)
    out_path = tmp_path / "rollout.gif"

    render_skeleton_comparison_gif(seq_a, seq_b, labels=("a", "b"), fps=10.0, out_path=out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
