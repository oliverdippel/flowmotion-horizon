from __future__ import annotations

import torch

from flowmotion.data.skeleton import (
    JOINT_PARENTS,
    NUM_JOINTS,
    REST_OFFSETS,
    forward_kinematics,
)


def test_topology_shapes_are_consistent():
    assert len(JOINT_PARENTS) == NUM_JOINTS
    assert len(REST_OFFSETS) == NUM_JOINTS
    assert JOINT_PARENTS[0] == -1  # root has no parent
    for j, p in enumerate(JOINT_PARENTS[1:], start=1):
        assert 0 <= p < j  # every parent index precedes its child (a valid tree order)


def test_joint_parents_match_canonical_smpl_kinematic_tree():
    # The standard published SMPL kinematic tree (root + 21 body joints + 2 hands,
    # e.g. as used in the original SMPL paper and reproduced identically in smplx /
    # human_body_prior reference implementations). We only model the first 22 (hands
    # dropped), so this pins our JOINT_PARENTS against that reference's prefix.
    canonical_24_joint_parents = [
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21,
    ]
    assert JOINT_PARENTS == canonical_24_joint_parents[:NUM_JOINTS]


def test_identity_rotations_reproduce_cumulative_rest_offsets():
    offsets = torch.tensor(REST_OFFSETS)
    identity = torch.eye(3).expand(NUM_JOINTS, 3, 3)
    root_trans = torch.zeros(3)

    joint_pos = forward_kinematics(identity, root_trans)

    expected = torch.zeros(NUM_JOINTS, 3)
    for j in range(1, NUM_JOINTS):
        p = JOINT_PARENTS[j]
        expected[j] = expected[p] + offsets[j]

    assert torch.allclose(joint_pos, expected, atol=1e-6)


def test_forward_kinematics_is_batched():
    identity = torch.eye(3).expand(4, 7, NUM_JOINTS, 3, 3)
    root_trans = torch.rand(4, 7, 3)
    joint_pos = forward_kinematics(identity, root_trans)
    assert joint_pos.shape == (4, 7, NUM_JOINTS, 3)
    assert torch.allclose(joint_pos[..., 0, :], root_trans, atol=1e-6)
