from __future__ import annotations

import numpy as np

from flowmotion.data.amass_format import NUM_BETAS, NUM_POSE_DIMS, REQUIRED_KEYS
from flowmotion.data.synthetic import generate_synthetic_amass


def test_writes_expected_directory_layout_and_keys(tmp_path):
    root = tmp_path / "amass"
    written = generate_synthetic_amass(
        root,
        n_datasets=2,
        n_subjects_per_dataset=2,
        n_sequences_per_subject=2,
        frames_range=(50, 60),
        seed=0,
    )
    assert len(written) == 2 * 2 * 2
    for path in written:
        assert path.parent.parent.parent == root
        with np.load(path) as data:
            for key in REQUIRED_KEYS:
                assert key in data
            T = data["poses"].shape[0]
            assert data["poses"].shape == (T, NUM_POSE_DIMS)
            assert data["trans"].shape == (T, 3)
            assert data["betas"].shape == (NUM_BETAS,)
            assert 50 <= T <= 60


def test_same_seed_is_deterministic(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    written_a = generate_synthetic_amass(root_a, n_datasets=1, n_subjects_per_dataset=2, seed=7)
    written_b = generate_synthetic_amass(root_b, n_datasets=1, n_subjects_per_dataset=2, seed=7)

    for pa, pb in zip(sorted(written_a), sorted(written_b)):
        with np.load(pa) as da, np.load(pb) as db:
            assert np.array_equal(da["poses"], db["poses"])
            assert np.array_equal(da["trans"], db["trans"])
            assert np.array_equal(da["betas"], db["betas"])


def test_different_seeds_differ(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    written_a = generate_synthetic_amass(root_a, n_datasets=1, n_subjects_per_dataset=1, seed=1)
    written_b = generate_synthetic_amass(root_b, n_datasets=1, n_subjects_per_dataset=1, seed=2)

    with np.load(written_a[0]) as da, np.load(written_b[0]) as db:
        assert not np.array_equal(da["poses"], db["poses"])


def test_pose_content_is_smooth_not_iid_jumps(tmp_path):
    root = tmp_path / "amass"
    written = generate_synthetic_amass(
        root,
        n_datasets=1,
        n_subjects_per_dataset=1,
        n_sequences_per_subject=1,
        frames_range=(200, 200),
        seed=0,
    )
    with np.load(written[0]) as data:
        poses = data["poses"][:, :66]
    deltas = np.abs(np.diff(poses, axis=0))
    # a smooth random walk with step_std=0.02 should very rarely jump more than ~10 sigma
    assert deltas.max() < 0.2
