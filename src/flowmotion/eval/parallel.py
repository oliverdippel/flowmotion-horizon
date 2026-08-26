"""Parallel horizon-eval: shards held-out sequences across worker processes.

Each held-out (subject, seed, rollout-length) trial is independent -- there's no
state shared between trials beyond the read-only reference statistics and foot-floor
calibration, both computed once up front. This runs the exact same trial logic as
`eval.harness.run_horizon_eval` (`_run_trials`) in each worker, so the parallel and
serial paths can't diverge in behavior; `_trial_seed` being a function of trial
identity rather than iteration order is what makes the result independent of how
work is partitioned.

Workers reload the checkpoint from disk rather than receiving the model object
directly: this avoids relying on any particular process start method (`fork` vs.
`spawn`) to share a loaded `nn.Module` correctly, at the cost of one redundant
(cheap) checkpoint load per worker.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

from flowmotion.data.loader import SequenceMeta, discover_sequences
from flowmotion.eval.harness import (
    EvalConfig,
    LoadedSequence,
    _run_trials,
    compute_reference_artifacts,
    load_sequences,
)


def _partition(items: list, n: int) -> list[list]:
    """Round-robin partition into at most `n` non-empty shards."""
    shards: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        shards[i % n].append(item)
    return [shard for shard in shards if shard]


def _eval_shard(
    checkpoint_path: str,
    shard: list[LoadedSequence],
    subject_vocab: dict[str, int],
    action_vocab: dict[str, int],
    cfg: EvalConfig,
    ref_stats: dict,
    foot_floor: dict,
) -> list[dict]:
    from flowmotion.train import load_checkpoint  # imported here: must be picklable at spawn

    ckpt = load_checkpoint(checkpoint_path)
    return _run_trials(
        ckpt["model"],
        ckpt["normalizer"],
        subject_vocab,
        action_vocab,
        shard,
        cfg,
        ref_stats,
        foot_floor,
    )


def run_horizon_eval_parallel(
    checkpoint_path: str | Path,
    data_root: str | Path,
    cfg: EvalConfig,
    num_workers: int,
) -> pd.DataFrame:
    """Same result as `eval.harness.run_horizon_eval`, computed across `num_workers`
    processes. Held-out sequences (not individual trials) are the unit of sharding, so
    a subject's reference-statistics-relevant data is loaded once regardless of shard."""
    from flowmotion.train import load_checkpoint

    ckpt = load_checkpoint(checkpoint_path)
    cfg = replace(cfg, yaw_align=ckpt["cfg"].get("yaw_align", False))
    sequences: list[SequenceMeta] = discover_sequences(data_root)
    held_out_keys = set(ckpt["held_out_subjects"])
    held_out_sequences = [s for s in sequences if s.subject_key in held_out_keys]

    loaded = load_sequences(held_out_sequences, cfg.fps)
    ref_stats, foot_floor = compute_reference_artifacts(loaded, cfg.fps, cfg.trailing_window)

    shards = _partition(loaded, num_workers)
    checkpoint_str = str(checkpoint_path)
    rows: list[dict] = []

    with ProcessPoolExecutor(max_workers=len(shards)) as pool:
        futures = [
            pool.submit(
                _eval_shard,
                checkpoint_str,
                shard,
                ckpt["subject_vocab"],
                ckpt["action_vocab"],
                cfg,
                ref_stats,
                foot_floor,
            )
            for shard in shards
        ]
        for future in futures:
            rows.extend(future.result())

    return pd.DataFrame(rows)
