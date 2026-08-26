"""Conditional flow matching / rectified-flow training objective.

x_t = (1-t) x0 + t x1, u_t = x1 - x0, loss = E[|| v_theta(x_t, t, cond) - u_t ||^2].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_training_pair(
    x1: torch.Tensor, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """x1: (B, H, D) target windows -> (x0, t, x_t, u_t)."""
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype, generator=generator)
    t_ = t.view(-1, 1, 1)
    x_t = (1.0 - t_) * x0 + t_ * x1
    u_t = x1 - x0
    return x0, t, x_t, u_t


def flow_matching_loss(
    model,
    past: torch.Tensor,
    target: torch.Tensor,
    subject_id: torch.Tensor,
    action_id: torch.Tensor,
) -> torch.Tensor:
    _, t, x_t, u_t = sample_training_pair(target)
    v_pred = model(x_t, t, past, subject_id, action_id, training=True)
    return F.mse_loss(v_pred, u_t)
