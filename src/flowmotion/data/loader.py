"""Discovers and loads AMASS-format sequences (real or synthetic-fixture, same code path).

Directory layout expected: <root>/<dataset_name>/<subject_dir>/<sequence_name>.npz
(see flowmotion.data.amass_format for the .npz key/shape contract).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flowmotion.data.amass_format import NUM_BODY_JOINTS, NUM_USED_POSE_DIMS

AMASS_ROOT_ENV_VAR = "AMASS_ROOT"


def resolve_data_root(explicit: str | Path | None) -> Path:
    """Explicit path wins; otherwise falls back to the AMASS_ROOT env var."""
    if explicit is not None:
        return Path(explicit)
    env_val = os.environ.get(AMASS_ROOT_ENV_VAR)
    if env_val:
        return Path(env_val)
    raise ValueError(
        f"No data root given and {AMASS_ROOT_ENV_VAR} is not set. "
        "Pass --data-root, or export AMASS_ROOT=/path/to/amass."
    )


@dataclass(frozen=True)
class SequenceMeta:
    path: Path
    dataset_name: str
    subject_key: str  # f"{dataset_name}/{subject_dir}" -- unique across datasets
    num_frames: int
    framerate: float


@dataclass
class RawSequence:
    poses: np.ndarray  # (T, 22, 3) float32 axis-angle
    trans: np.ndarray  # (T, 3) float32
    betas: np.ndarray  # (16,) float32
    gender: str
    framerate: float


def discover_sequences(root: str | Path) -> list[SequenceMeta]:
    """Globs `<root>/*/*/*.npz` and reads just enough of each file to build metadata.

    Real AMASS subject directories can contain non-sequence .npz files alongside motion
    sequences -- e.g. a per-subject `shape.npz` holding only `gender`/`betas`, no `poses`.
    Those are skipped (not every .npz under a subject dir is a motion sequence)."""
    root = Path(root)
    metas: list[SequenceMeta] = []
    skipped: list[Path] = []
    for path in sorted(root.glob("*/*/*.npz")):
        dataset_name = path.parent.parent.name
        subject_dir = path.parent.name
        subject_key = f"{dataset_name}/{subject_dir}"
        with np.load(path) as data:
            if "poses" not in data or "mocap_framerate" not in data:
                skipped.append(path)
                continue
            num_frames = int(data["poses"].shape[0])
            framerate = float(data["mocap_framerate"])
        metas.append(
            SequenceMeta(
                path=path,
                dataset_name=dataset_name,
                subject_key=subject_key,
                num_frames=num_frames,
                framerate=framerate,
            )
        )
    if skipped:
        print(f"discover_sequences: skipped {len(skipped)} non-motion .npz file(s) under {root}")
    if not metas:
        raise FileNotFoundError(f"No sequences found under {root} (expected */*/*.npz)")
    return metas


def load_sequence(meta: SequenceMeta) -> RawSequence:
    with np.load(meta.path) as data:
        poses_full = data["poses"]
        trans = data["trans"].astype(np.float32)
        betas = data["betas"].astype(np.float32)
        gender = str(data["gender"])
        framerate = float(data["mocap_framerate"])
    poses = poses_full[:, :NUM_USED_POSE_DIMS].reshape(-1, NUM_BODY_JOINTS, 3).astype(np.float32)
    return RawSequence(poses=poses, trans=trans, betas=betas, gender=gender, framerate=framerate)


def resample_to_fps(seq: RawSequence, target_fps: float = 20.0) -> RawSequence:
    """Nearest-frame striding (no interpolation) down to `target_fps`."""
    if seq.framerate <= 0:
        raise ValueError(f"Invalid framerate {seq.framerate}")
    stride = max(1, round(seq.framerate / target_fps))
    return RawSequence(
        poses=seq.poses[::stride],
        trans=seq.trans[::stride],
        betas=seq.betas,
        gender=seq.gender,
        framerate=seq.framerate / stride,
    )
