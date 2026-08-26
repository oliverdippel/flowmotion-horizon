"""Reduced 22-joint SMPL body skeleton: kinematic tree, approximate rest pose, forward kinematics.

NOTE (open design call, see README limitations): there is no real SMPL body model here
(betas-conditioned shape is license-gated separately from AMASS mocap data). REST_OFFSETS
below is a hardcoded, approximate adult T-pose bone-offset template, not derived from a real
SMPL shape space -- absolute joint positions are approximate. JOINT_PARENTS matches the
canonical published SMPL kinematic tree (root + 21 body joints, hands dropped) exactly;
`tests/test_skeleton_fk.py` pins it against that reference array as a regression check.

Joint order (index: name, parent):
 0 pelvis (root)       parent -1
 1 left_hip            parent 0
 2 right_hip           parent 0
 3 spine1              parent 0
 4 left_knee           parent 1
 5 right_knee          parent 2
 6 spine2              parent 3
 7 left_ankle          parent 4
 8 right_ankle         parent 5
 9 spine3              parent 6
10 left_foot           parent 7
11 right_foot          parent 8
12 neck                parent 9
13 left_collar         parent 9
14 right_collar        parent 9
15 head                parent 12
16 left_shoulder       parent 13
17 right_shoulder      parent 14
18 left_elbow          parent 16
19 right_elbow         parent 17
20 left_wrist          parent 18
21 right_wrist         parent 19
"""

from __future__ import annotations

import torch

NUM_JOINTS = 22

JOINT_PARENTS: list[int] = [
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
]

# Approximate adult T-pose bone offsets (meters), relative to parent. Axes: x = left(+)/right(-),
# y = front(+)/back(-), z = up(+)/down(-) (up_axis="z", see README limitations).
REST_OFFSETS: list[tuple[float, float, float]] = [
    (0.00, 0.00, 0.00),  # 0 pelvis (root, unused as an offset)
    (0.09, 0.00, -0.07),  # 1 left_hip
    (-0.09, 0.00, -0.07),  # 2 right_hip
    (0.00, 0.00, 0.12),  # 3 spine1
    (0.00, 0.00, -0.40),  # 4 left_knee
    (0.00, 0.00, -0.40),  # 5 right_knee
    (0.00, 0.00, 0.14),  # 6 spine2
    (0.00, 0.00, -0.40),  # 7 left_ankle
    (0.00, 0.00, -0.40),  # 8 right_ankle
    (0.00, 0.00, 0.10),  # 9 spine3
    (0.00, 0.12, -0.05),  # 10 left_foot
    (0.00, 0.12, -0.05),  # 11 right_foot
    (0.00, 0.00, 0.10),  # 12 neck
    (0.08, 0.00, 0.05),  # 13 left_collar
    (-0.08, 0.00, 0.05),  # 14 right_collar
    (0.00, 0.00, 0.12),  # 15 head
    (0.12, 0.00, 0.00),  # 16 left_shoulder
    (-0.12, 0.00, 0.00),  # 17 right_shoulder
    (0.25, 0.00, 0.00),  # 18 left_elbow
    (-0.25, 0.00, 0.00),  # 19 right_elbow
    (0.25, 0.00, 0.00),  # 20 left_wrist
    (-0.25, 0.00, 0.00),  # 21 right_wrist
]

LEFT_FOOT_IDX = 10
RIGHT_FOOT_IDX = 11
UP_AXIS = "z"
UP_AXIS_IDX = {"x": 0, "y": 1, "z": 2}[UP_AXIS]


def _rest_offsets_tensor(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(REST_OFFSETS, dtype=dtype, device=device)  # (22, 3)


def forward_kinematics(joint_rotmats: torch.Tensor, root_trans: torch.Tensor) -> torch.Tensor:
    """joint_rotmats: (..., 22, 3, 3) local rotations. root_trans: (..., 3).

    Returns global joint positions (..., 22, 3), computed by walking the fixed kinematic
    tree in index order (every parent index is smaller than its child's, so a single
    forward pass suffices).
    """
    offsets = _rest_offsets_tensor(joint_rotmats.dtype, joint_rotmats.device)
    lead_shape = joint_rotmats.shape[:-3]

    global_rot: list[torch.Tensor | None] = [None] * NUM_JOINTS
    global_pos: list[torch.Tensor | None] = [None] * NUM_JOINTS
    global_rot[0] = joint_rotmats[..., 0, :, :]
    global_pos[0] = root_trans

    for j in range(1, NUM_JOINTS):
        p = JOINT_PARENTS[j]
        parent_rot, parent_pos = global_rot[p], global_pos[p]
        assert parent_rot is not None and parent_pos is not None  # p < j, already computed
        offset = offsets[j].expand(*lead_shape, 3)
        global_rot[j] = parent_rot @ joint_rotmats[..., j, :, :]
        global_pos[j] = parent_pos + (parent_rot @ offset.unsqueeze(-1)).squeeze(-1)

    return torch.stack([pos for pos in global_pos if pos is not None], dim=-2)
