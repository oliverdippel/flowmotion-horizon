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

from typing import Protocol

import torch

from flowmotion.data.transforms import TRANS_START, Normalizer, yaw_align_window, yaw_unalign_window


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
    """Fixed-step Euler integration of dx/dt = v_theta(x, t, past, labels) from t=0 to t=1.

    Runs under torch.no_grad() -- sampling is pure inference and never needs gradients,
    and eval/rollout call this many times (e.g. thousands of trials in the horizon-eval
    harness), so building an autograd graph here is pure waste."""
    B = past.shape[0]
    device = past.device
    with torch.no_grad():
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
    window_abs: torch.Tensor, normalizer: Normalizer, yaw_align: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """window_abs: (K, D) absolute -> (normalized (1,K,D), reference_xy (2,), yaw).
    `yaw` is 0 (an identity rotation) when `yaw_align` is False, so callers can always
    pass it to `_denormalize_and_uncenter` unconditionally."""
    ref_xy = window_abs[0, TRANS_START : TRANS_START + 2].clone()
    recentered = window_abs.clone()
    recentered[:, TRANS_START : TRANS_START + 2] -= ref_xy
    if yaw_align:
        recentered, yaw = yaw_align_window(recentered)
    else:
        yaw = torch.zeros(())
    normalized = normalizer.transform(recentered).unsqueeze(0)
    return normalized, ref_xy, yaw


def _denormalize_and_uncenter(
    pred_norm: torch.Tensor, ref_xy: torch.Tensor, yaw: torch.Tensor, normalizer: Normalizer
) -> torch.Tensor:
    """pred_norm: (1, H, D) -> absolute (H, D). Un-does yaw alignment before
    un-recentering -- a no-op when `yaw` is 0."""
    pred = normalizer.inverse_transform(pred_norm.squeeze(0))
    pred = yaw_unalign_window(pred, yaw)
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
    yaw_align: bool = False,
) -> torch.Tensor:
    """seed_past_abs: (K, D) absolute seed window -> (total_frames, D) absolute rollout.
    `yaw_align` must match how the model was trained (see TrainConfig.yaw_align) --
    checkpoints record this in their saved cfg."""
    device = seed_past_abs.device
    D = seed_past_abs.shape[-1]
    subject_t = torch.tensor([subject_id], dtype=torch.long, device=device)
    action_t = torch.tensor([action_id], dtype=torch.long, device=device)

    current_abs = seed_past_abs.clone()
    frames_out: list[torch.Tensor] = []
    while len(frames_out) * H < total_frames:
        past_norm, ref_xy, yaw = _recenter_and_normalize(current_abs, normalizer, yaw_align)
        pred_norm = integrate_velocity(
            model, past_norm, subject_t, action_t, H, D, steps=steps, generator=generator
        )
        pred_abs = _denormalize_and_uncenter(pred_norm, ref_xy, yaw, normalizer)
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
    yaw_align: bool = False,
) -> torch.Tensor:
    """Re-conditions on real ground-truth windows at every step; records the model's
    predictions (not the real data) so its output is directly comparable to free_rollout.
    Stops early (returning fewer than total_frames rows) if real_sequence_abs runs out.
    `yaw_align` must match how the model was trained -- see `free_rollout`."""
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
        past_norm, ref_xy, yaw = _recenter_and_normalize(past_abs, normalizer, yaw_align)
        pred_norm = integrate_velocity(
            model, past_norm, subject_t, action_t, H, D, steps=steps, generator=generator
        )
        pred_abs = _denormalize_and_uncenter(pred_norm, ref_xy, yaw, normalizer)
        frames_out.append(pred_abs)
        w += 1

    if not frames_out:
        return torch.zeros(0, D, device=device)
    return torch.cat(frames_out, dim=0)[:total_frames]


class Roller(Protocol):
    """What the eval harness needs to run matched free/teacher-forced rollouts, without
    caring whether the underlying predictor is a trained model or a trivial baseline
    (see `flowmotion.baseline.ZeroVelocityRoller`) -- `eval.harness._run_trials` is
    written against this interface, not against `VelocityTransformer` directly."""

    K: int
    H: int

    def eval(self) -> None: ...

    def free(
        self,
        seed_past_abs: torch.Tensor,
        subject_id: int,
        action_id: int,
        total_frames: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor: ...

    def teacher_forced(
        self,
        real_sequence_abs: torch.Tensor,
        seed_start_idx: int,
        subject_id: int,
        action_id: int,
        total_frames: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor: ...


class ModelRoller:
    """Adapts the trained flow-matching model + normalizer to the `Roller` interface."""

    def __init__(
        self,
        model,
        normalizer: Normalizer,
        steps: int = 10,
        yaw_align: bool = False,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.steps = steps
        self.yaw_align = yaw_align
        self.K = model.K
        self.H = model.H

    def eval(self) -> None:
        self.model.eval()

    def free(
        self,
        seed_past_abs: torch.Tensor,
        subject_id: int,
        action_id: int,
        total_frames: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        return free_rollout(
            self.model,
            self.normalizer,
            seed_past_abs,
            subject_id,
            action_id,
            total_frames=total_frames,
            H=self.H,
            steps=self.steps,
            generator=generator,
            yaw_align=self.yaw_align,
        )

    def teacher_forced(
        self,
        real_sequence_abs: torch.Tensor,
        seed_start_idx: int,
        subject_id: int,
        action_id: int,
        total_frames: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        return teacher_forced_rollout(
            self.model,
            self.normalizer,
            real_sequence_abs,
            seed_start_idx,
            subject_id,
            action_id,
            total_frames=total_frames,
            K=self.K,
            H=self.H,
            steps=self.steps,
            generator=generator,
            yaw_align=self.yaw_align,
        )
