"""Command-line entry points: prepare-fixture / train / rollout / eval / demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from flowmotion.config import DEFAULT_ODE_STEPS, TARGET_FPS
from flowmotion.data.dataset import lookup_id
from flowmotion.data.loader import (
    discover_sequences,
    load_sequence,
    resample_to_fps,
    resolve_data_root,
)
from flowmotion.data.synthetic import generate_synthetic_amass
from flowmotion.data.transforms import features_from_numpy
from flowmotion.eval.harness import EvalConfig, run_horizon_eval
from flowmotion.eval.metrics import sequence_features_to_joint_positions
from flowmotion.eval.report import plot_horizon_curves, write_csv, write_json_summary
from flowmotion.rollout import free_rollout, teacher_forced_rollout
from flowmotion.train import TrainConfig, load_checkpoint, train
from flowmotion.viz import render_skeleton_comparison_gif


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flowmotion")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fixture = sub.add_parser("prepare-fixture", help="generate a synthetic AMASS-shaped dataset")
    p_fixture.add_argument("--out", required=True)
    p_fixture.add_argument("--seed", type=int, default=0)
    p_fixture.add_argument("--n-datasets", type=int, default=2)
    p_fixture.add_argument("--n-subjects-per-dataset", type=int, default=3)
    p_fixture.add_argument("--n-sequences-per-subject", type=int, default=2)

    p_train = sub.add_parser("train", help="train the conditional flow-matching model")
    p_train.add_argument("--data-root", default=None)
    p_train.add_argument("--out", dest="out_dir", default="./runs/tiny")
    p_train.add_argument("--steps", type=int, default=500)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--d-model", type=int, default=256)
    p_train.add_argument("--n-layers", type=int, default=4)
    p_train.add_argument("--n-heads", type=int, default=4)
    p_train.add_argument("--cache-size", type=int, default=64)

    p_rollout = sub.add_parser("rollout", help="free-rollout from a trained checkpoint")
    p_rollout.add_argument("--checkpoint", required=True)
    p_rollout.add_argument("--data-root", default=None)
    p_rollout.add_argument("--subject", required=True, help="subject_key, e.g. Dataset0/subject01")
    p_rollout.add_argument("--length", type=int, default=90)
    p_rollout.add_argument("--out", required=True)
    p_rollout.add_argument("--ode-steps", type=int, default=DEFAULT_ODE_STEPS)
    p_rollout.add_argument("--seed", type=int, default=0)

    p_eval = sub.add_parser("eval", help="run the horizon-stability eval harness")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--data-root", default=None)
    p_eval.add_argument("--lengths", default="30,60,90,150,300")
    p_eval.add_argument("--out-dir", required=True)
    p_eval.add_argument("--seeds-per-subject", type=int, default=2)
    p_eval.add_argument("--ode-steps", type=int, default=DEFAULT_ODE_STEPS)

    p_demo = sub.add_parser("demo", help="fixture -> tiny train -> eval, end to end")
    p_demo.add_argument("--out", default="./runs/demo")
    p_demo.add_argument("--seed", type=int, default=0)

    p_viz = sub.add_parser(
        "visualize", help="render a free-vs-teacher-forced rollout comparison as a GIF"
    )
    p_viz.add_argument("--checkpoint", required=True)
    p_viz.add_argument("--data-root", default=None)
    p_viz.add_argument("--subject", required=True, help="subject_key, e.g. Dataset0/subject01")
    p_viz.add_argument("--length", type=int, default=90)
    p_viz.add_argument("--out", required=True)
    p_viz.add_argument("--ode-steps", type=int, default=DEFAULT_ODE_STEPS)
    p_viz.add_argument("--seed", type=int, default=0)

    return parser


def cmd_prepare_fixture(args: argparse.Namespace) -> None:
    out = Path(args.out)
    written = generate_synthetic_amass(
        out,
        n_datasets=args.n_datasets,
        n_subjects_per_dataset=args.n_subjects_per_dataset,
        n_sequences_per_subject=args.n_sequences_per_subject,
        seed=args.seed,
    )
    print(f"wrote {len(written)} synthetic sequences to {out}")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        cache_size=args.cache_size,
    )
    train(cfg)


def cmd_rollout(args: argparse.Namespace) -> None:
    ckpt = load_checkpoint(args.checkpoint)
    model, normalizer = ckpt["model"], ckpt["normalizer"]

    data_root = resolve_data_root(args.data_root)
    sequences = discover_sequences(data_root)
    matches = [s for s in sequences if s.subject_key == args.subject]
    if not matches:
        raise ValueError(f"no sequences found for subject_key={args.subject!r}")
    meta = matches[0]

    raw = resample_to_fps(load_sequence(meta), target_fps=TARGET_FPS)
    feat = features_from_numpy(raw.poses, raw.trans)
    K = model.K
    if feat.shape[0] < K:
        raise ValueError(f"sequence {meta.path} has only {feat.shape[0]} frames, need >= {K}")
    seed_past = feat[:K]

    subject_id = lookup_id(meta.subject_key, ckpt["subject_vocab"])
    action_id = lookup_id(meta.dataset_name, ckpt["action_vocab"])
    generator = torch.Generator().manual_seed(args.seed)

    result = free_rollout(
        model,
        normalizer,
        seed_past,
        subject_id,
        action_id,
        total_frames=args.length,
        H=model.H,
        steps=args.ode_steps,
        generator=generator,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, features=result.detach().numpy())
    print(f"wrote rollout ({result.shape[0]} frames) to {out_path}")


def _run_eval_and_report(
    ckpt: dict, held_out_sequences, eval_cfg: EvalConfig, out_dir: Path
) -> None:
    df = run_horizon_eval(
        ckpt["model"],
        ckpt["normalizer"],
        ckpt["subject_vocab"],
        ckpt["action_vocab"],
        held_out_sequences,
        eval_cfg,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(df, out_dir / "horizon_eval.csv")
    write_json_summary(df, out_dir / "horizon_eval_summary.json")
    plot_horizon_curves(df, out_dir / "horizon_curves.png")
    n_skipped = int(df["skipped"].sum()) if len(df) else 0
    print(f"wrote eval report to {out_dir} ({len(df) - n_skipped} evaluated, {n_skipped} skipped)")


def cmd_eval(args: argparse.Namespace) -> None:
    ckpt = load_checkpoint(args.checkpoint)
    data_root = resolve_data_root(args.data_root)
    sequences = discover_sequences(data_root)
    held_out_keys = set(ckpt["held_out_subjects"])
    held_out_sequences = [s for s in sequences if s.subject_key in held_out_keys]

    lengths = [int(x) for x in args.lengths.split(",")]
    cfg = EvalConfig(
        rollout_lengths=lengths, seeds_per_subject=args.seeds_per_subject, ode_steps=args.ode_steps
    )
    _run_eval_and_report(ckpt, held_out_sequences, cfg, Path(args.out_dir))


def cmd_demo(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    fixture_dir = out_dir / "fixture"
    generate_synthetic_amass(fixture_dir, seed=args.seed)

    train_cfg = TrainConfig(
        data_root=str(fixture_dir),
        out_dir=str(out_dir / "train"),
        steps=200,
        batch_size=16,
        d_model=64,
        n_layers=2,
        n_heads=2,
        seed=args.seed,
    )
    checkpoint_path = train(train_cfg)
    ckpt = load_checkpoint(checkpoint_path)

    sequences = discover_sequences(fixture_dir)
    held_out_keys = set(ckpt["held_out_subjects"])
    held_out_sequences = [s for s in sequences if s.subject_key in held_out_keys]

    eval_cfg = EvalConfig(rollout_lengths=[30, 60, 90], seeds_per_subject=1)
    _run_eval_and_report(ckpt, held_out_sequences, eval_cfg, out_dir / "eval_report")


def cmd_visualize(args: argparse.Namespace) -> None:
    ckpt = load_checkpoint(args.checkpoint)
    model, normalizer = ckpt["model"], ckpt["normalizer"]

    data_root = resolve_data_root(args.data_root)
    sequences = discover_sequences(data_root)
    matches = [s for s in sequences if s.subject_key == args.subject]
    if not matches:
        raise ValueError(f"no sequences found for subject_key={args.subject!r}")
    meta = matches[0]

    raw = resample_to_fps(load_sequence(meta), target_fps=TARGET_FPS)
    feat = features_from_numpy(raw.poses, raw.trans)
    K, H = model.K, model.H
    if feat.shape[0] < K + args.length:
        raise ValueError(
            f"sequence {meta.path} has only {feat.shape[0]} frames, "
            f"need >= {K + args.length} for a length-{args.length} rollout"
        )
    seed_past = feat[:K]

    subject_id = lookup_id(meta.subject_key, ckpt["subject_vocab"])
    action_id = lookup_id(meta.dataset_name, ckpt["action_vocab"])

    free = free_rollout(
        model,
        normalizer,
        seed_past,
        subject_id,
        action_id,
        total_frames=args.length,
        H=H,
        steps=args.ode_steps,
        generator=torch.Generator().manual_seed(args.seed),
    )
    teacher_forced = teacher_forced_rollout(
        model,
        normalizer,
        feat,
        0,
        subject_id,
        action_id,
        total_frames=args.length,
        K=K,
        H=H,
        steps=args.ode_steps,
        generator=torch.Generator().manual_seed(args.seed),
    )

    free_jp = sequence_features_to_joint_positions(free)
    tf_jp = sequence_features_to_joint_positions(teacher_forced)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_skeleton_comparison_gif(
        free_jp, tf_jp, labels=("free rollout", "teacher-forced"), fps=TARGET_FPS, out_path=out_path
    )
    print(f"wrote visualization ({min(free_jp.shape[0], tf_jp.shape[0])} frames) to {out_path}")


_COMMANDS = {
    "prepare-fixture": cmd_prepare_fixture,
    "train": cmd_train,
    "rollout": cmd_rollout,
    "eval": cmd_eval,
    "demo": cmd_demo,
    "visualize": cmd_visualize,
}


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
