"""Zero-velocity baseline: repeats the last observed frame indefinitely.

This is the standard sanity baseline from human motion prediction (Martinez et al.,
"On Human Motion Prediction Using Recurrent Neural Networks", CVPR 2017), where it
turned out to be surprisingly hard to beat at short horizons. A trained model earning
its keep should separate from it, especially on distributional-drift and divergence,
which is exactly what a model producing no motion at all should fail.

Implements `flowmotion.rollout.Roller` so it plugs into the same eval-harness trial
loop (`eval.harness._run_trials`) as the trained flow-matching model -- same seed
windows, same lengths, same metrics, same matched-RNG protocol. `generator` is
accepted for interface compatibility but unused: the baseline is deterministic.
"""

from __future__ import annotations

import torch


class ZeroVelocityRoller:
    def __init__(self, K: int, H: int) -> None:
        self.K = K
        self.H = H

    def eval(self) -> None:
        pass

    def free(
        self,
        seed_past_abs: torch.Tensor,
        subject_id: int,
        action_id: int,
        total_frames: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        last = seed_past_abs[-1:]
        return last.repeat(total_frames, 1)

    def teacher_forced(
        self,
        real_sequence_abs: torch.Tensor,
        seed_start_idx: int,
        subject_id: int,
        action_id: int,
        total_frames: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Re-reads the real window's last frame every step, mirroring
        `rollout.teacher_forced_rollout`'s early-stop-when-real-data-runs-out behavior
        so the two are directly comparable under the same protocol."""
        D = real_sequence_abs.shape[-1]
        device = real_sequence_abs.device
        frames_out: list[torch.Tensor] = []
        w = 0
        while len(frames_out) * self.H < total_frames:
            real_start = seed_start_idx + w * self.H
            if real_start + self.K > real_sequence_abs.shape[0]:
                break
            last = real_sequence_abs[real_start + self.K - 1 : real_start + self.K]
            frames_out.append(last.repeat(self.H, 1))
            w += 1

        if not frames_out:
            return torch.zeros(0, D, device=device)
        return torch.cat(frames_out, dim=0)[:total_frames]
