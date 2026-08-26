from __future__ import annotations

from conftest import make_synthetic_root

from flowmotion.train import TrainConfig, train


def _tiny_cfg(data_root, out_dir, **overrides):
    defaults = dict(
        data_root=str(data_root),
        out_dir=str(out_dir),
        K=4,
        H=4,
        stride=4,
        d_model=16,
        n_layers=1,
        n_heads=2,
        dim_ff=32,
        steps=10,
        batch_size=8,
        held_out_frac=0.4,
        seed=0,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_val_loss_disabled_by_default_writes_no_val_log(tmp_path):
    data_root = make_synthetic_root(
        tmp_path,
        n_datasets=1,
        n_subjects_per_dataset=5,
        n_sequences_per_subject=1,
        frames_range=(60, 60),
        framerate=20.0,
        seed=0,
    )
    out_dir = tmp_path / "run"
    train(_tiny_cfg(data_root, out_dir))
    assert not (out_dir / "val_log.csv").exists()


def test_val_loss_tracked_when_enabled(tmp_path):
    data_root = make_synthetic_root(
        tmp_path,
        n_datasets=1,
        n_subjects_per_dataset=5,
        n_sequences_per_subject=1,
        frames_range=(60, 60),
        framerate=20.0,
        seed=0,
    )
    out_dir = tmp_path / "run"
    train(_tiny_cfg(data_root, out_dir, val_every=3, val_batches=2))

    val_log_path = out_dir / "val_log.csv"
    assert val_log_path.exists()
    lines = val_log_path.read_text().strip().splitlines()
    assert lines[0] == "step,val_loss"
    assert len(lines) > 1  # at least one validation point was recorded

    for line in lines[1:]:
        step_str, loss_str = line.split(",")
        assert int(step_str) >= 0
        assert float(loss_str) >= 0.0  # MSE loss, never negative
