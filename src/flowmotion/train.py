"""Training loop for the conditional flow-matching velocity network.

Runs as a single process by default. Launched with `torchrun`, it trains with
`torch.distributed` data parallelism instead -- distributed mode is detected from
the environment variables `torchrun` sets (`WORLD_SIZE`, `RANK`, `LOCAL_RANK`), not
a separate CLI flag, so the same command works either way. Only rank 0 logs and
writes the checkpoint.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from flowmotion.config import DEFAULT_H, DEFAULT_K, TARGET_FPS
from flowmotion.data.dataset import MotionWindowDataset, build_vocab, split_subjects
from flowmotion.data.loader import discover_sequences, resolve_data_root
from flowmotion.data.transforms import FEATURE_DIM, Normalizer
from flowmotion.model.flow_matching import flow_matching_loss
from flowmotion.model.network import VelocityTransformer


@dataclass
class TrainConfig:
    data_root: str | None = None
    out_dir: str = "./runs/tiny"
    K: int = DEFAULT_K
    H: int = DEFAULT_H
    stride: int = 5
    target_fps: float = TARGET_FPS
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dim_ff: int = 512
    dropout: float = 0.1
    p_label_dropout: float = 0.1
    lr: float = 3e-4
    steps: int = 500
    batch_size: int = 32
    held_out_frac: float = 0.15
    seed: int = 0
    log_every: int = field(default=50)
    cache_size: int = 64
    val_every: int = 0  # 0 disables validation-loss tracking entirely
    val_batches: int = 20
    yaw_align: bool = False


def _distributed_info() -> tuple[int, int, int]:
    """(rank, local_rank, world_size), read from the env vars `torchrun` sets. Returns
    (0, 0, 1) -- "not distributed" -- for a plain, non-torchrun process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, local_rank, world_size


def _infinite_batches(loader: DataLoader, sampler: DistributedSampler | None):
    """Yields batches indefinitely, re-iterating the loader (and calling
    `sampler.set_epoch` when distributed) at each epoch boundary, so shuffling
    actually changes across epochs instead of replaying one cached pass forever."""
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def _compute_val_loss(
    model: torch.nn.Module,
    val_dataset: MotionWindowDataset,
    device: torch.device,
    max_batches: int,
    batch_size: int = 32,
) -> float:
    """Mean flow-matching loss over up to `max_batches` batches of held-out-subject
    windows -- the same objective as training, just on data the model never trains on.
    Held-out subjects aren't in the training vocabulary, so this also exercises the
    null-embedding conditioning path (see model/conditioning.py)."""
    loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = flow_matching_loss(
                model, batch["past"], batch["target"], batch["subject_id"], batch["action_id"]
            )
            total += loss.item()
            count += 1
    if was_training:
        model.train()
    return total / max(count, 1)


def train(cfg: TrainConfig) -> Path:
    rank, local_rank, world_size = _distributed_info()
    is_distributed = world_size > 1
    is_main = rank == 0

    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # Same seed on every rank: identical subject split, vocabularies, and (since
    # MotionWindowDataset/Normalizer.fit_streaming have no randomness of their own)
    # identical normalizer stats and model initialization, with no cross-rank
    # broadcast needed for any of it. DDP separately broadcasts model parameters
    # from rank 0 on construction as a second guarantee of consistency.
    torch.manual_seed(cfg.seed)

    data_root = resolve_data_root(cfg.data_root)
    sequences = discover_sequences(data_root)
    subject_keys = [s.subject_key for s in sequences]
    train_subjects, held_out_subjects = split_subjects(
        subject_keys, held_out_frac=cfg.held_out_frac, seed=cfg.seed
    )
    train_set = set(train_subjects)
    train_sequences = [s for s in sequences if s.subject_key in train_set]

    subject_vocab = build_vocab(train_subjects)
    action_vocab = build_vocab([s.dataset_name for s in sequences])

    train_dataset = MotionWindowDataset(
        train_sequences,
        subject_vocab,
        action_vocab,
        cfg.K,
        cfg.H,
        cfg.stride,
        normalizer=None,
        target_fps=cfg.target_fps,
        cache_size=cfg.cache_size,
        yaw_align=cfg.yaw_align,
    )
    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset has zero windows -- check K/H/stride vs sequence lengths."
        )

    # Streaming fit: one pass over every window's features via the same lazy/LRU-cached
    # loader `__getitem__` uses, so fitting a real AMASS-scale corpus never requires
    # holding the whole thing in memory at once.
    normalizer = Normalizer.fit_streaming(train_dataset.iter_recentered_windows())
    train_dataset.normalizer = normalizer

    batch_size = min(cfg.batch_size, len(train_dataset))
    sampler: DistributedSampler | None = None
    if is_distributed:
        sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed
        )
        loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
    else:
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model: torch.nn.Module = VelocityTransformer(
        D=FEATURE_DIM,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dim_ff=cfg.dim_ff,
        K=cfg.K,
        H=cfg.H,
        num_subjects=len(subject_vocab),
        num_actions=len(action_vocab),
        dropout=cfg.dropout,
        p_label_dropout=cfg.p_label_dropout,
    ).to(device)

    if is_distributed:
        ddp_device_ids = [local_rank] if torch.cuda.is_available() else None
        model = DistributedDataParallel(model, device_ids=ddp_device_ids)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    val_dataset = None
    if cfg.val_every > 0:
        held_out_set = set(held_out_subjects)
        held_out_sequences = [s for s in sequences if s.subject_key in held_out_set]
        val_dataset = MotionWindowDataset(
            held_out_sequences,
            subject_vocab,
            action_vocab,
            cfg.K,
            cfg.H,
            cfg.stride,
            normalizer=normalizer,  # train-fit stats, not the held-out set's own
            target_fps=cfg.target_fps,
            cache_size=cfg.cache_size,
            yaw_align=cfg.yaw_align,
        )
        if len(val_dataset) == 0:
            if is_main:
                print("val_every set but the held-out set has zero windows -- skipping validation")
            val_dataset = None

    out_dir = Path(cfg.out_dir)
    writer = None
    log_file = None
    val_writer = None
    val_log_file = None
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(out_dir / "train_log.csv", "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["step", "loss"])
        if val_dataset is not None:
            val_log_file = open(out_dir / "val_log.csv", "w", newline="")
            val_writer = csv.writer(val_log_file)
            val_writer.writerow(["step", "val_loss"])

    data_iter = _infinite_batches(loader, sampler)
    model.train()
    for step in range(cfg.steps):
        batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = flow_matching_loss(
            model, batch["past"], batch["target"], batch["subject_id"], batch["action_id"]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if is_main:
            assert writer is not None
            writer.writerow([step, loss.item()])
            is_last_step = step == cfg.steps - 1
            if step % cfg.log_every == 0 or is_last_step:
                world_note = f" (world_size={world_size})" if is_distributed else ""
                print(f"step {step:5d}  loss {loss.item():.4f}{world_note}")
            if val_dataset is not None and (step % cfg.val_every == 0 or is_last_step):
                val_loss = _compute_val_loss(model, val_dataset, device, cfg.val_batches)
                assert val_writer is not None
                val_writer.writerow([step, val_loss])
                print(f"step {step:5d}  val_loss {val_loss:.4f}")

    if is_main:
        if log_file is not None:
            log_file.close()
        if val_log_file is not None:
            val_log_file.close()

    checkpoint_path = out_dir / "model.pt"
    if is_main:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        torch.save(
            {
                "state_dict": raw_model.state_dict(),
                "model_kwargs": {
                    "D": FEATURE_DIM,
                    "d_model": cfg.d_model,
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "dim_ff": cfg.dim_ff,
                    "K": cfg.K,
                    "H": cfg.H,
                    "num_subjects": len(subject_vocab),
                    "num_actions": len(action_vocab),
                    "dropout": cfg.dropout,
                    "p_label_dropout": cfg.p_label_dropout,
                },
                "normalizer": normalizer.to_state_dict(),
                "subject_vocab": subject_vocab,
                "action_vocab": action_vocab,
                "train_subjects": train_subjects,
                "held_out_subjects": held_out_subjects,
                "cfg": asdict(cfg),
            },
            checkpoint_path,
        )
        print(f"saved checkpoint to {checkpoint_path}")

    if is_distributed:
        dist.barrier()  # hold other ranks until rank 0 finishes writing the checkpoint
        dist.destroy_process_group()

    return checkpoint_path


def load_checkpoint(path: str | Path) -> dict:
    """Always loads to CPU regardless of what device the checkpoint was trained on
    (`map_location="cpu"`), so a checkpoint trained on a CUDA machine can be loaded on
    a CPU-only one -- e.g. train on a free Colab/Kaggle GPU, then run eval/rollout/
    visualize locally, which is the documented workflow for this repository."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = VelocityTransformer(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    normalizer = Normalizer.from_state_dict(ckpt["normalizer"])
    return {
        "model": model,
        "normalizer": normalizer,
        "subject_vocab": ckpt["subject_vocab"],
        "action_vocab": ckpt["action_vocab"],
        "train_subjects": ckpt["train_subjects"],
        "held_out_subjects": ckpt["held_out_subjects"],
        "cfg": ckpt["cfg"],
    }
