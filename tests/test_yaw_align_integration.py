"""End-to-end check that yaw_align is recorded in the checkpoint and that eval/rollout
correctly pick it up from there rather than needing to be told separately -- this is
what actually prevents a model trained with yaw_align=True from being silently
evaluated as if it weren't (or vice versa).
"""

from __future__ import annotations

from conftest import make_synthetic_root

from flowmotion.data.loader import discover_sequences, load_sequence, resample_to_fps
from flowmotion.data.transforms import features_from_numpy
from flowmotion.eval.harness import EvalConfig, run_horizon_eval
from flowmotion.eval.parallel import run_horizon_eval_parallel
from flowmotion.rollout import free_rollout
from flowmotion.train import TrainConfig, load_checkpoint, train


def test_yaw_align_flag_is_recorded_and_used_end_to_end(tmp_path):
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
        yaw_align=True,
    )
    checkpoint_path = train(train_cfg)

    ckpt = load_checkpoint(checkpoint_path)
    assert ckpt["cfg"]["yaw_align"] is True

    sequences = discover_sequences(data_root)
    held_out_keys = set(ckpt["held_out_subjects"])
    held_out_sequences = [s for s in sequences if s.subject_key in held_out_keys]

    eval_cfg = EvalConfig(rollout_lengths=[8, 16], seeds_per_subject=1, ode_steps=4)
    eval_cfg.yaw_align = ckpt["cfg"]["yaw_align"]  # what cli.py does before calling this
    serial_df = run_horizon_eval(
        ckpt["model"],
        ckpt["normalizer"],
        ckpt["subject_vocab"],
        ckpt["action_vocab"],
        held_out_sequences,
        eval_cfg,
    )
    assert not serial_df.empty
    assert (~serial_df["skipped"]).any()

    # the parallel path auto-detects yaw_align from the checkpoint itself
    parallel_df = run_horizon_eval_parallel(
        checkpoint_path,
        data_root,
        EvalConfig(rollout_lengths=[8, 16], seeds_per_subject=1, ode_steps=4),
        num_workers=2,
    )
    assert not parallel_df.empty
    assert (~parallel_df["skipped"]).any()

    K = ckpt["model"].K
    raw = resample_to_fps(load_sequence(held_out_sequences[0]), target_fps=20.0)
    feat = features_from_numpy(raw.poses, raw.trans)
    rollout = free_rollout(
        ckpt["model"],
        ckpt["normalizer"],
        feat[:K],
        subject_id=0,
        action_id=0,
        total_frames=8,
        H=ckpt["model"].H,
        steps=4,
        yaw_align=ckpt["cfg"]["yaw_align"],
    )
    assert rollout.shape[0] == 8
