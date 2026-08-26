"""ODE sampling and autoregressive rollout (free and teacher-forced).

All rollout functions operate on absolute (unnormalized, un-recentered) feature-space
frames as their public input/output, and internally re-derive the per-window recentering
+ normalization the model was trained on -- so a caller never has to think about the
recentering convention.

`free_rollout` feeds each predicted window back in as the next past window.
`teacher_forced_rollout` instead re-conditions on real ground-truth frames every step,
while still recording what the model predicted -- used together (matched seed/RNG) by
the eval harness's divergence metric to isolate compounding self-conditioning error.
"""

from __future__ import annotations

import torch

from flowmotion.data.transforms import TRANS_START, Normalizer


def integrate_velocity(
    model,
    past: torch.Tensor,
    subject_id: torch.Tensor,
    action_id: torch.Tensor,
    H: int,
    D: int,
    steps: int = 10,
    x0: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Fixed-step Euler integration of dx/dt = v_theta(x, t, past, labels) from t=0 to t=1."""
    B = past.shape[0]
    device = past.device
    x = (
        x0
        if x0 is not None
        else torch.randn(B, H, D, device=device, dtype=past.dtype, generator=generator)
    )
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i * dt, device=device, dtype=past.dtype)
        v = model(x, t, past, subject_id, action_id, training=False)
        x = x + v * dt
    return x


def _recenter_and_normalize(
    window_abs: torch.Tensor, normalizer: Normalizer
) -> tuple[torch.Tensor, torch.Tensor]:
    """window_abs: (K, D) absolute -> (normalized (1,K,D), reference_xy (2,))."""
    ref_xy = window_abs[0, TRANS_START : TRANS_START + 2].clone()
    recentered = window_abs.clone()
    recentered[:, TRANS_START : TRANS_START + 2] -= ref_xy
    normalized = normalizer.transform(recentered).unsqueeze(0)
    return normalized, ref_xy


def _denormalize_and_uncenter(
    pred_norm: torch.Tensor, ref_xy: torch.Tensor, normalizer: Normalizer
) -> torch.Tensor:
    """pred_norm: (1, H, D) -> absolute (H, D)."""
    pred = normalizer.inverse_transform(pred_norm.squeeze(0))
    pred = pred.clone()
    pred[:, TRANS_START : TRANS_START + 2] += ref_xy
    return pred


def free_rollout(
    model,
    normalizer: Normalizer,
    seed_past_abs: torch.Tensor,
    subject_id: int,
    action_id: int,
    total_frames: int,
    H: int,
    steps: int = 10,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """seed_past_abs: (K, D) absolute seed window -> (total_frames, D) absolute rollout."""
    device = seed_past_abs.device
    D = seed_past_abs.shape[-1]
    subject_t = torch.tensor([subject_id], dtype=torch.long, device=device)
    action_t = torch.tensor([action_id], dtype=torch.long, device=device)

    current_abs = seed_past_abs.clone()
    frames_out: list[torch.Tensor] = []
    while len(frames_out) * H < total_frames:
        past_norm, ref_xy = _recenter_and_normalize(current_abs, normalizer)
        pred_norm = integrate_velocity(
            model, past_norm, subject_t, action_t, H, D, steps=steps, generator=generator
        )
        pred_abs = _denormalize_and_uncenter(pred_norm, ref_xy, normalizer)
        frames_out.append(pred_abs)
        current_abs = pred_abs  # K == H: predicted window becomes the next past window

    return torch.cat(frames_out, dim=0)[:total_frames]


def teacher_forced_rollout(
    model,
    normalizer: Normalizer,
    real_sequence_abs: torch.Tensor,
    seed_start_idx: int,
    subject_id: int,
    action_id: int,
    total_frames: int,
    K: int,
    H: int,
    steps: int = 10,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Re-conditions on real ground-truth windows at every step; records the model's
    predictions (not the real data) so its output is directly comparable to free_rollout.
    Stops early (returning fewer than total_frames rows) if real_sequence_abs runs out."""
    device = real_sequence_abs.device
    D = real_sequence_abs.shape[-1]
    subject_t = torch.tensor([subject_id], dtype=torch.long, device=device)
    action_t = torch.tensor([action_id], dtype=torch.long, device=device)

    frames_out: list[torch.Tensor] = []
    w = 0
    while len(frames_out) * H < total_frames:
        real_start = seed_start_idx + w * H
        if real_start + K > real_sequence_abs.shape[0]:
            break
        past_abs = real_sequence_abs[real_start : real_start + K]
        past_norm, ref_xy = _recenter_and_normalize(past_abs, normalizer)
        pred_norm = integrate_velocity(
            model, past_norm, subject_t, action_t, H, D, steps=steps, generator=generator
        )
        pred_abs = _denormalize_and_uncenter(pred_norm, ref_xy, normalizer)
        frames_out.append(pred_abs)
        w += 1

    if not frames_out:
        return torch.zeros(0, D, device=device)
    return torch.cat(frames_out, dim=0)[:total_frames]
