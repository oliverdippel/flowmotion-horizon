"""Documents the on-disk AMASS format this codebase reads.

AMASS (https://amass.is.tue.mpg.de) distributes each dataset (e.g. "CMU", "KIT",
"BMLmovi", ...) as a directory of per-subject subdirectories, each containing one
.npz file per motion sequence:

    <amass_root>/<dataset_name>/<subject_id>/<sequence_name>.npz

Each .npz contains (SMPL-H body model, "AMASS" split, un-shaped):
    poses:            float32 (T, 156)  -- axis-angle: [root(3), body(63), hands(45+45)]
    trans:             float32 (T, 3)    -- root translation, meters
    betas:             float32 (16,)     -- SMPL-H shape coefficients
    gender:            str               -- "male" / "female" / "neutral"
    mocap_framerate:   float             -- native capture fps (commonly 60, 120, or 150)

This codebase only uses the root + 21 body joints (drops both hands), i.e. the first
`66 = 22 * 3` axis-angle dims of `poses`. `betas`/`gender` are loaded and threaded through
so a real SMPL/SMPL-H forward-kinematics model can be substituted later without touching
the data pipeline (see flowmotion.data.skeleton for the current approximate stand-in).

Nothing in this module downloads or requires AMASS to be present: it exists purely as a
contract that both the real loader (flowmotion.data.loader) and the synthetic fixture
generator (flowmotion.data.synthetic) implement identically, so pointing --data-root at a
real AMASS tree instead of the synthetic fixture requires no code changes.
"""

from __future__ import annotations

NUM_POSE_DIMS = 156
NUM_BODY_JOINTS = 22  # root + 21 body joints, hands dropped
NUM_USED_POSE_DIMS = NUM_BODY_JOINTS * 3  # 66
NUM_BETAS = 16

REQUIRED_KEYS = ("poses", "trans", "betas", "gender", "mocap_framerate")
