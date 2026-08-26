"""VelocityTransformer: the flow-matching velocity field v_theta(x_t, t, past, labels).

Conditioning is injected as extra tokens in the transformer sequence (time, subject, action),
not via AdaLN modulation -- this uses stock nn.TransformerEncoder rather than a custom block,
trading some architectural elegance for a much smaller surface area to get wrong in a 2-3 day
build (see plan's open design calls).

Token sequence: [t, subject, action, past_0..K-1, target_0..H-1], length 3+K+H.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from flowmotion.model.conditioning import LabelEmbedding

PAST_TYPE = 0
TARGET_TYPE = 1


def sinusoidal_time_embed(t: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    """t: (B,) in [0, 1] -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t.unsqueeze(-1) * scale * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class VelocityTransformer(nn.Module):
    def __init__(
        self,
        D: int = 135,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dim_ff: int = 512,
        K: int = 10,
        H: int = 10,
        num_subjects: int = 1,
        num_actions: int = 1,
        dropout: float = 0.1,
        p_label_dropout: float = 0.1,
    ):
        super().__init__()
        self.D = D
        self.d_model = d_model
        self.K = K
        self.H = H
        self.p_label_dropout = p_label_dropout

        self.input_proj = nn.Linear(D, d_model)
        self.output_head = nn.Linear(d_model, D)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.subject_embed = LabelEmbedding(num_subjects, d_model)
        self.action_embed = LabelEmbedding(num_actions, d_model)
        self.pos_embed = nn.Embedding(K + H, d_model)
        self.type_embed = nn.Embedding(2, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        past: torch.Tensor,
        subject_id: torch.Tensor,
        action_id: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        device = x_t.device
        p_drop = self.p_label_dropout if training else 0.0

        t_emb = sinusoidal_time_embed(t, self.d_model)
        t_token = self.time_mlp(t_emb).unsqueeze(1)
        subj_token = self.subject_embed(subject_id, p_drop, training).unsqueeze(1)
        action_token = self.action_embed(action_id, p_drop, training).unsqueeze(1)

        past_tokens = self.input_proj(past)
        target_tokens = self.input_proj(x_t)
        body_tokens = torch.cat([past_tokens, target_tokens], dim=1)

        positions = torch.arange(self.K + self.H, device=device)
        pos = self.pos_embed(positions).unsqueeze(0)
        types = torch.cat(
            [
                torch.full((self.K,), PAST_TYPE, dtype=torch.long, device=device),
                torch.full((self.H,), TARGET_TYPE, dtype=torch.long, device=device),
            ]
        )
        type_emb = self.type_embed(types).unsqueeze(0)
        body_tokens = body_tokens + pos + type_emb

        seq = torch.cat([t_token, subj_token, action_token, body_tokens], dim=1)
        h = self.encoder(seq)
        v_pred = self.output_head(h[:, -self.H :, :])
        return v_pred
