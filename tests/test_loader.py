from __future__ import annotations

import numpy as np
from conftest import make_synthetic_root

from flowmotion.data.loader import discover_sequences


def test_discover_sequences_skips_non_motion_npz_files(tmp_path):
    # real AMASS subject dirs can contain a per-subject shape.npz (gender + betas only,
    # no poses/mocap_framerate) alongside real motion sequences -- discover_sequences
    # must skip these rather than crash on the missing "poses" key.
    root = make_synthetic_root(
        tmp_path, n_datasets=1, n_subjects_per_dataset=2, n_sequences_per_subject=2
    )
    a_subject_dir = next((root / "Dataset0").iterdir())
    np.savez(a_subject_dir / "shape.npz", gender="male", betas=np.zeros(16, dtype=np.float32))

    sequences = discover_sequences(root)
    assert all(s.path.name != "shape.npz" for s in sequences)
    assert len(sequences) == 2 * 2  # unaffected by the extra non-motion file
