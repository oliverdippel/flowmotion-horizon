"""Verifies the parallel eval path (multiple worker processes) produces exactly the
same results as the serial path -- the point of deterministic, identity-based trial
seeding (`_trial_seed`) rather than an iteration-order-dependent counter.
"""

from __future__ import annotations

from conftest import make_synthetic_root

from flowmotion.data.loader import discover_sequences
from flowmotion.eval.harness import EvalConfig, run_horizon_eval
from flowmotion.eval.parallel import run_horizon_eval_parallel
from flowmotion.train import TrainConfig, load_checkpoint, train


def _canonical(df):
    cols = ["subject_id", "seq_id", "seed_idx", "rollout_length"]
    return df.sort_values(cols).reset_index(drop=True)


def test_parallel_eval_matches_serial_eval(tmp_path):
    data_root = make_synthetic_root(
        tmp_path,
        n_datasets=1,
        n_subjects_per_dataset=5,
        n_sequences_per_subject=1,
        frames_range=(60, 60),
        framerate=20.0,
        seed=0,
    )

    train_cfg = TrainConfig(
        data_root=str(data_root),
        out_dir=str(tmp_path / "run"),
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
    checkpoint_path = train(train_cfg)

    ckpt = load_checkpoint(checkpoint_path)
    sequences = discover_sequences(data_root)
    held_out_keys = set(ckpt["held_out_subjects"])
    held_out_sequences = [s for s in sequences if s.subject_key in held_out_keys]
    assert len(held_out_sequences) >= 2  # need at least 2 to make sharding meaningful

    eval_cfg = EvalConfig(rollout_lengths=[8, 16], seeds_per_subject=1, ode_steps=4)

    serial_df = run_horizon_eval(
        ckpt["model"],
        ckpt["normalizer"],
        ckpt["subject_vocab"],
        ckpt["action_vocab"],
        held_out_sequences,
        eval_cfg,
    )
    parallel_df = run_horizon_eval_parallel(checkpoint_path, data_root, eval_cfg, num_workers=2)

    serial_sorted = _canonical(serial_df)
    parallel_sorted = _canonical(parallel_df)

    assert len(serial_sorted) == len(parallel_sorted) > 0
    for col in serial_sorted.columns:
        if serial_sorted[col].dtype.kind in "fc":
            diff = serial_sorted[col].fillna(-999) - parallel_sorted[col].fillna(-999)
            assert diff.abs().max() < 1e-5
        else:
            assert (serial_sorted[col] == parallel_sorted[col]).all()
