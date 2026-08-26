from __future__ import annotations

import pytest
from conftest import foot_contact_case

from flowmotion.eval.metrics import estimate_foot_floor, foot_skate


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


def test_foot_skate_uses_calibrated_floor_when_absolute_zero_is_never_reached():
    # a rig where "ground" is actually at world height ~1.0, not 0 -- e.g. this codebase's
    # approximate skeleton on real AMASS data (verified: real ground-truth foot height
    # never drops below ~0.5-0.6 under it). Without a calibrated floor, foot_skate would
    # report zero contact frames despite the foot genuinely planting each cycle.
    import torch

    T = 8
    joint_pos = torch.zeros(T, 22, 3)
    joint_pos[:, :, 2] = 1.2  # every other joint held at 1.2, irrelevant to this test
    # left foot alternates: near-floor (1.0, planted) / lifted (1.3, swing)
    left_heights = torch.tensor([1.0, 1.0, 1.0, 1.3, 1.3, 1.0, 1.0, 1.0])
    joint_pos[:, 10, 2] = left_heights
    joint_pos[:, 11, 2] = 1.0  # right foot always planted at the same floor height

    reference = [joint_pos]  # pretend this trajectory is also the "real reference" data
    floor = estimate_foot_floor(reference, percentile=1.0)
    assert floor["left"] == pytest.approx(1.0, abs=1e-6)
    assert floor["right"] == pytest.approx(1.0, abs=1e-6)

    without_floor = foot_skate(joint_pos, fps=20.0, height_thresh=0.05)
    assert without_floor["contact_frame_count"] == 0  # absolute floor of 0 -> nothing qualifies

    with_floor = foot_skate(joint_pos, fps=20.0, height_thresh=0.05, floor=floor)
    assert with_floor["contact_frame_count"] > 0
