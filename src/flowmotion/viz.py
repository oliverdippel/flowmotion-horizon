"""Renders an animated GIF comparing two skeleton motion sequences side by side -- e.g.
free rollout vs. teacher-forced -- so horizon collapse is something you watch, not just
read off a metric. No real SMPL mesh (see skeleton.py's limitations): this draws the
approximate stick-figure rig as points + bone segments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from flowmotion.data.skeleton import JOINT_PARENTS, NUM_JOINTS


def _axis_limits(*sequences: torch.Tensor, margin: float = 0.2) -> tuple[np.ndarray, float]:
    all_pts = torch.cat([s.detach().reshape(-1, 3) for s in sequences], dim=0)
    mins = all_pts.min(dim=0).values
    maxs = all_pts.max(dim=0).values
    center = ((mins + maxs) / 2).numpy()
    half_range = (maxs - mins).max().item() / 2 + margin
    return center, half_range


def _style_axes(ax, title: str, center: np.ndarray, half_range: float) -> None:
    ax.set_title(title)
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=-60)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def _draw_skeleton(ax, joint_pos: np.ndarray, color: str) -> None:
    ax.scatter(joint_pos[:, 0], joint_pos[:, 1], joint_pos[:, 2], c=color, s=15)
    for j in range(1, NUM_JOINTS):
        p = JOINT_PARENTS[j]
        ax.plot(
            [joint_pos[j, 0], joint_pos[p, 0]],
            [joint_pos[j, 1], joint_pos[p, 1]],
            [joint_pos[j, 2], joint_pos[p, 2]],
            c=color,
            linewidth=2,
        )


def render_skeleton_comparison_gif(
    seq_a: torch.Tensor,
    seq_b: torch.Tensor,
    labels: tuple[str, str],
    fps: float,
    out_path: str | Path,
    color_a: str = "tab:blue",
    color_b: str = "tab:orange",
) -> None:
    """seq_a, seq_b: (T, 22, 3) world joint positions (e.g. from
    `flowmotion.eval.metrics.sequence_features_to_joint_positions`). Renders both as an
    animated GIF, side by side, sharing one time axis and one set of axis limits so the
    two are visually comparable frame by frame."""
    T = min(seq_a.shape[0], seq_b.shape[0])
    a = seq_a[:T].detach().cpu().numpy()
    b = seq_b[:T].detach().cpu().numpy()
    center, half_range = _axis_limits(seq_a[:T], seq_b[:T])

    fig = plt.figure(figsize=(10, 5))
    ax_a = fig.add_subplot(1, 2, 1, projection="3d")
    ax_b = fig.add_subplot(1, 2, 2, projection="3d")

    def update(frame_idx: int):
        ax_a.cla()
        ax_b.cla()
        _style_axes(ax_a, f"{labels[0]} (frame {frame_idx})", center, half_range)
        _style_axes(ax_b, f"{labels[1]} (frame {frame_idx})", center, half_range)
        _draw_skeleton(ax_a, a[frame_idx], color_a)
        _draw_skeleton(ax_b, b[frame_idx], color_b)
        return []

    anim = FuncAnimation(fig, update, frames=T, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=round(fps)))
    plt.close(fig)
