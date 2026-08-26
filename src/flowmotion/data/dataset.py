"""Subject-level train/held-out split, vocabularies, and windowed motion dataset."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from flowmotion.data.loader import SequenceMeta, load_sequence, resample_to_fps
from flowmotion.data.transforms import FEATURE_DIM, TRANS_START, Normalizer, features_from_numpy


def split_subjects(
    subject_keys: list[str], held_out_frac: float = 0.15, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Deterministic subject-level split. Operates on the SET of unique keys, so it is
    invariant to how many sequences each subject has -- a sequence-level split would leak
    subject identity into "held-out" data, which is exactly the bug this must avoid."""
    unique_keys = sorted(set(subject_keys))
    n = len(unique_keys)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_held = max(1, round(n * held_out_frac)) if n > 1 else 0
    held_positions = set(perm[:n_held].tolist())
    train = [unique_keys[i] for i in range(n) if i not in held_positions]
    held_out = [unique_keys[i] for i in range(n) if i in held_positions]
    return train, held_out


def build_vocab(keys: list[str]) -> dict[str, int]:
    """Sorted-unique -> contiguous int ids. Index `len(vocab)` is reserved as the null id
    for any key not present (e.g. a held-out subject the model was never trained on)."""
    return {k: i for i, k in enumerate(sorted(set(keys)))}


def lookup_id(key: str, vocab: dict[str, int]) -> int:
    return vocab.get(key, len(vocab))


class MotionWindowDataset(Dataset):
    def __init__(
        self,
        sequences: list[SequenceMeta],
        subject_vocab: dict[str, int],
        action_vocab: dict[str, int],
        K: int,
        H: int,
        stride: int,
        normalizer: Normalizer | None,
        target_fps: float = 20.0,
    ):
        self.K = K
        self.H = H
        self.subject_vocab = subject_vocab
        self.action_vocab = action_vocab
        self.normalizer = normalizer

        self.features: list[torch.Tensor] = []
        self.subject_keys: list[str] = []
        self.dataset_names: list[str] = []
        self.windows: list[tuple[int, int]] = []  # (seq_idx, start)

        for seq_idx, meta in enumerate(sequences):
            raw = resample_to_fps(load_sequence(meta), target_fps=target_fps)
            feat = features_from_numpy(raw.poses, raw.trans)
            self.features.append(feat)
            self.subject_keys.append(meta.subject_key)
            self.dataset_names.append(meta.dataset_name)

            n_frames = feat.shape[0]
            last_start = n_frames - K - H
            for start in range(0, last_start + 1, stride):
                self.windows.append((seq_idx, start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        seq_idx, start = self.windows[idx]
        feat = self.features[seq_idx]
        K, H = self.K, self.H

        past = feat[start : start + K].clone()
        target = feat[start + K : start + K + H].clone()

        reference_xy = past[0, TRANS_START : TRANS_START + 2].clone()
        past[:, TRANS_START : TRANS_START + 2] -= reference_xy
        target[:, TRANS_START : TRANS_START + 2] -= reference_xy

        if self.normalizer is not None:
            past = self.normalizer.transform(past)
            target = self.normalizer.transform(target)

        subject_id = lookup_id(self.subject_keys[seq_idx], self.subject_vocab)
        action_id = lookup_id(self.dataset_names[seq_idx], self.action_vocab)

        return {
            "past": past,
            "target": target,
            "subject_id": torch.tensor(subject_id, dtype=torch.long),
            "action_id": torch.tensor(action_id, dtype=torch.long),
        }

    def all_train_features(self) -> torch.Tensor:
        """All (past, target) window features concatenated, for fitting a Normalizer."""
        rows = []
        for seq_idx, start in self.windows:
            feat = self.features[seq_idx][start : start + self.K + self.H].clone()
            ref_xy = feat[0, TRANS_START : TRANS_START + 2].clone()
            feat[:, TRANS_START : TRANS_START + 2] -= ref_xy
            rows.append(feat)
        return torch.cat(rows, dim=0) if rows else torch.zeros(0, FEATURE_DIM)
