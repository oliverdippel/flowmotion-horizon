from __future__ import annotations

import torch

from flowmotion.data.synthetic import generate_synthetic_amass


def make_synthetic_root(tmp_path, **kwargs):
    root = tmp_path / "amass"
    defaults = dict(
        n_datasets=1,
        n_subjects_per_dataset=3,
        n_sequences_per_subject=2,
        frames_range=(120, 160),
        framerate=20.0,
        seed=0,
    )
    defaults.update(kwargs)
    generate_synthetic_amass(root, **defaults)
    return root


def foot_contact_case() -> tuple[torch.Tensor, dict]:
    """Hand-built (T, 22, 3) joint positions with a known, exact foot-contact/skate
    pattern, for asserting foot_skate against exact expected numbers rather than
    "some plausible value".

    Left foot (idx 10): frame0 low+static (excluded, no reference frame) -> frame1 low+
    static (contact, disp=0) -> frame2 low but moved +0.10 in x (contact, disp=0.10) ->
    frame3 lifted high (swing, not contact) -> frame4 lifted+moved (swing, not contact) ->
    frame5 slammed back down (height drops fast -> vertical velocity exceeds threshold,
    not classified as stable contact even though it's now low).

    Right foot (idx 11): static and low for all 6 frames (contact frames 1-5, disp=0 each).

    Expected: left contributes contact_frames=2, total_skate=0.10; right contributes
    contact_frames=5, total_skate=0.0. Combined mean_skate_per_contact_frame = 0.10/7.
    """
    T = 6
    joint_pos = torch.zeros(T, 22, 3)
    for j in range(22):
        joint_pos[:, j, :] = torch.tensor([0.5 + 0.01 * j, 0.5, 1.0])  # static, irrelevant joints

    left = torch.zeros(T, 3)
    left[0] = torch.tensor([0.0, 0.0, 0.02])
    left[1] = torch.tensor([0.0, 0.0, 0.02])
    left[2] = torch.tensor([0.10, 0.0, 0.02])
    left[3] = torch.tensor([0.10, 0.0, 0.30])
    left[4] = torch.tensor([0.20, 0.0, 0.30])
    left[5] = torch.tensor([0.20, 0.0, 0.02])
    joint_pos[:, 10, :] = left

    right = torch.tensor([0.0, 0.0, 0.02]).expand(T, 3).clone()
    joint_pos[:, 11, :] = right

    expected = {
        "mean_skate_per_contact_frame": 0.10 / 7,
        "total_skate": 0.10,
        "contact_frame_count": 7,
        "left_contact_frames": 2,
        "right_contact_frames": 5,
    }
    return joint_pos, expected
