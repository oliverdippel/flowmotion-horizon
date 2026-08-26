"""Training loop for the conditional flow-matching velocity network."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from itertools import cycle
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

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


def train(cfg: TrainConfig) -> Path:
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
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = VelocityTransformer(
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
    )
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"

    data_iter = cycle(loader)
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["step", "loss"])
        model.train()
        for step in range(cfg.steps):
            batch = next(data_iter)
            loss = flow_matching_loss(
                model, batch["past"], batch["target"], batch["subject_id"], batch["action_id"]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.writerow([step, loss.item()])
            if step % cfg.log_every == 0 or step == cfg.steps - 1:
                print(f"step {step:5d}  loss {loss.item():.4f}")

    checkpoint_path = out_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
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
    return checkpoint_path


def load_checkpoint(path: str | Path) -> dict:
    ckpt = torch.load(path, weights_only=False)
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
