from __future__ import annotations

import pytest
from conftest import foot_contact_case

from flowmotion.eval.metrics import foot_skate


def test_foot_skate_exact_expected_value():
    joint_pos, expected = foot_contact_case()
    result = foot_skate(joint_pos, fps=20.0)

    assert result["contact_frame_count"] == expected["contact_frame_count"]
    assert result["total_skate"] == pytest.approx(expected["total_skate"], abs=1e-6)
    assert result["mean_skate_per_contact_frame"] == pytest.approx(
        expected["mean_skate_per_contact_frame"], abs=1e-6
    )
    assert result["per_foot"]["left"]["contact_frames"] == expected["left_contact_frames"]
    assert result["per_foot"]["right"]["contact_frames"] == expected["right_contact_frames"]
    assert result["per_foot"]["right"]["total_skate"] == pytest.approx(0.0, abs=1e-6)


def test_foot_skate_zero_contact_frames_is_not_reported_as_good():
    # both feet stay above the height threshold the whole time -> zero contact frames.
    import torch

    joint_pos = torch.zeros(10, 22, 3)
    joint_pos[:, :, 2] = 1.0  # every joint well above height_thresh

    result = foot_skate(joint_pos, fps=20.0, height_thresh=0.05)
    assert result["contact_frame_count"] == 0
    assert result["mean_skate_per_contact_frame"] == 0.0
