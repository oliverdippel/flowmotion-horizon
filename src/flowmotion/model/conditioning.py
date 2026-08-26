"""Categorical label embeddings with a learned null token and training-time dropout.

The null token is what makes classifier-free-guidance-style conditioning possible later
(CFG sampling itself is out of scope here). It also does double duty for held-out subjects
at eval time: `flowmotion.data.dataset.lookup_id` maps any subject key not seen during
training to `num_labels` (the null index), so a held-out subject is conditioned on the
learned "generic body" embedding rather than crashing on an out-of-vocab id.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LabelEmbedding(nn.Module):
    def __init__(self, num_labels: int, d_model: int):
        super().__init__()
        self.num_labels = num_labels
        self.null_index = num_labels
        self.embed = nn.Embedding(num_labels + 1, d_model)

    def forward(
        self, ids: torch.Tensor, p_dropout: float = 0.0, training: bool = True
    ) -> torch.Tensor:
        ids = ids.clamp(max=self.null_index)
        if training and p_dropout > 0:
            mask = torch.rand(ids.shape, device=ids.device) < p_dropout
            ids = torch.where(mask, torch.full_like(ids, self.null_index), ids)
        return self.embed(ids)
