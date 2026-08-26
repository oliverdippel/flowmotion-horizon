"""Deterministic synthetic AMASS-shaped fixture generator.

Writes real .npz files matching flowmotion.data.amass_format's contract (same keys,
shapes, and directory layout AMASS itself uses), so the loader is exercised through
the exact same code path as real data -- nothing about the loader is mocked.

Pose content is a smooth, clipped random walk (not i.i.d. noise): i.i.d. per-frame
values would produce anatomically absurd frame-to-frame jumps that make the jerk and
foot-skate metrics meaningless to test against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from flowmotion.data.amass_format import NUM_BETAS, NUM_POSE_DIMS, NUM_USED_POSE_DIMS

_GENDERS = ("male", "female", "neutral")


def _smooth_random_walk(
    rng: np.random.Generator, num_frames: int, num_dims: int, step_std: float, clip: float
) -> np.ndarray:
    deltas = rng.normal(scale=step_std, size=(num_frames, num_dims)).astype(np.float32)
    walk = np.cumsum(deltas, axis=0)
    return np.clip(walk, -clip, clip).astype(np.float32)


def generate_synthetic_amass(
    root: Path,
    n_datasets: int = 2,
    n_subjects_per_dataset: int = 3,
    n_sequences_per_subject: int = 2,
    frames_range: tuple[int, int] = (600, 900),
    framerate: float = 120.0,
    seed: int = 0,
) -> list[Path]:
    """Populate `root` with a synthetic AMASS-shaped dataset tree. Returns written paths."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    written: list[Path] = []

    for d in range(n_datasets):
        dataset_name = f"Dataset{d}"
        for s in range(n_subjects_per_dataset):
            subject_id = f"subject{s:02d}"
            betas = rng.normal(scale=1.0, size=(NUM_BETAS,)).astype(np.float32)
            gender = rng.choice(_GENDERS)
            subj_dir = root / dataset_name / subject_id
            subj_dir.mkdir(parents=True, exist_ok=True)

            for q in range(n_sequences_per_subject):
                seq_name = f"seq{q:02d}"
                num_frames = int(rng.integers(frames_range[0], frames_range[1] + 1))

                poses = np.zeros((num_frames, NUM_POSE_DIMS), dtype=np.float32)
                poses[:, :NUM_USED_POSE_DIMS] = _smooth_random_walk(
                    rng, num_frames, NUM_USED_POSE_DIMS, step_std=0.02, clip=1.0
                )

                trans_xy = _smooth_random_walk(rng, num_frames, 2, step_std=0.01, clip=3.0)
                trans_z = 0.9 + _smooth_random_walk(rng, num_frames, 1, step_std=0.005, clip=0.15)
                trans = np.concatenate([trans_xy, trans_z], axis=1).astype(np.float32)

                out_path = subj_dir / f"{seq_name}.npz"
                np.savez(
                    out_path,
                    poses=poses,
                    trans=trans,
                    betas=betas,
                    gender=str(gender),
                    mocap_framerate=float(framerate),
                )
                written.append(out_path)

    return written
