"""Full pipeline, sized for CI: synthetic fixture -> tiny train -> horizon eval -> report.
Deliberately tiny (K=H=4, d_model=16, 1 layer, ~15 steps) so this runs in well under a
minute on CPU with no real data and no network access.
"""

from __future__ import annotations

from flowmotion.data.loader import discover_sequences
from flowmotion.data.synthetic import generate_synthetic_amass
from flowmotion.eval.harness import EvalConfig, run_horizon_eval
from flowmotion.eval.report import plot_horizon_curves, write_csv, write_json_summary
from flowmotion.train import TrainConfig, load_checkpoint, train


def test_end_to_end_smoke(tmp_path):
    data_root = tmp_path / "amass"
    generate_synthetic_amass(
        data_root,
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
        steps=15,
        batch_size=8,
        held_out_frac=0.2,
        seed=0,
    )
    checkpoint_path = train(train_cfg)
    assert checkpoint_path.exists()

    ckpt = load_checkpoint(checkpoint_path)
    assert len(ckpt["held_out_subjects"]) >= 1

    sequences = discover_sequences(data_root)
    held_out_keys = set(ckpt["held_out_subjects"])
    held_out_sequences = [s for s in sequences if s.subject_key in held_out_keys]
    assert held_out_sequences

    eval_cfg = EvalConfig(rollout_lengths=[8, 16], seeds_per_subject=1, ode_steps=4)
    df = run_horizon_eval(
        ckpt["model"],
        ckpt["normalizer"],
        ckpt["subject_vocab"],
        ckpt["action_vocab"],
        held_out_sequences,
        eval_cfg,
    )
    assert not df.empty
    assert (~df["skipped"]).any(), "every trial was skipped -- fixture too short for eval lengths"

    out_dir = tmp_path / "eval_report"
    out_dir.mkdir()
    write_csv(df, out_dir / "horizon_eval.csv")
    write_json_summary(df, out_dir / "horizon_eval_summary.json")
    plot_horizon_curves(df, out_dir / "horizon_curves.png")

    assert (out_dir / "horizon_eval.csv").stat().st_size > 0
    assert (out_dir / "horizon_eval_summary.json").stat().st_size > 0
    assert (out_dir / "horizon_curves.png").stat().st_size > 0
