"""Subject-level train/held-out split, vocabularies, and a lazily-loaded windowed
motion dataset.

Sequences are NOT loaded/converted upfront: the window index is built purely from
`SequenceMeta` header metadata (num_frames, framerate), and each sequence's feature
tensor is loaded, resampled, and converted on first access, then kept in a small
bounded LRU cache. This keeps peak memory bounded by `cache_size` resident sequences
rather than the whole corpus, which matters once `sequences` is a real AMASS-scale
list (thousands of files) rather than the synthetic fixture.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from flowmotion.data.loader import SequenceMeta, load_sequence, resample_to_fps
from flowmotion.data.transforms import (
    TRANS_START,
    Normalizer,
    features_from_numpy,
    yaw_align_window,
)


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


def _resampled_length(num_frames: int, native_fps: float, target_fps: float) -> int:
    """Frame count after `resample_to_fps`'s nearest-frame striding -- computed from
    header metadata alone, so the window index can be built without loading any sequence."""
    stride = max(1, round(native_fps / target_fps))
    return len(range(0, num_frames, stride))


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
        cache_size: int = 64,
        yaw_align: bool = False,
    ):
        self.sequences = sequences
        self.K = K
        self.H = H
        self.target_fps = target_fps
        self.subject_vocab = subject_vocab
        self.action_vocab = action_vocab
        self.normalizer = normalizer
        self.cache_size = cache_size
        self.yaw_align = yaw_align

        self.subject_keys = [meta.subject_key for meta in sequences]
        self.dataset_names = [meta.dataset_name for meta in sequences]
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()

        self.windows: list[tuple[int, int]] = []  # (seq_idx, start)
        for seq_idx, meta in enumerate(sequences):
            n_frames = _resampled_length(meta.num_frames, meta.framerate, target_fps)
            last_start = n_frames - K - H
            for start in range(0, last_start + 1, stride):
                self.windows.append((seq_idx, start))

    def _get_features(self, seq_idx: int) -> torch.Tensor:
        if seq_idx in self._cache:
            self._cache.move_to_end(seq_idx)
            return self._cache[seq_idx]

        raw = resample_to_fps(load_sequence(self.sequences[seq_idx]), target_fps=self.target_fps)
        feat = features_from_numpy(raw.poses, raw.trans)

        self._cache[seq_idx] = feat
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return feat

    def __len__(self) -> int:
        return len(self.windows)

    def _windowed(self, seq_idx: int, start: int) -> torch.Tensor:
        """(K+H, D) window, recentered (and yaw-aligned if enabled) relative to its own
        first frame -- the shared preprocessing used by both `__getitem__` and
        `iter_recentered_windows`, so the two can't silently disagree."""
        feat = self._get_features(seq_idx)
        window = feat[start : start + self.K + self.H].clone()
        ref_xy = window[0, TRANS_START : TRANS_START + 2].clone()
        window[:, TRANS_START : TRANS_START + 2] -= ref_xy
        if self.yaw_align:
            window, _yaw = yaw_align_window(window)
        return window

    def __getitem__(self, idx: int) -> dict:
        seq_idx, start = self.windows[idx]
        window = self._windowed(seq_idx, start)
        past = window[: self.K]
        target = window[self.K : self.K + self.H]

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

    def iter_recentered_windows(self):
        """Yields each (K+H, D) window, recentered (and yaw-aligned if enabled) but NOT
        normalized, one at a time -- for streaming normalizer fitting
        (`Normalizer.fit_streaming`) without ever materializing the whole corpus as one
        in-memory tensor."""
        for seq_idx, start in self.windows:
            yield self._windowed(seq_idx, start)
