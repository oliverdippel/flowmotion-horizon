from __future__ import annotations

import torch

from flowmotion.model.flow_matching import flow_matching_loss, sample_training_pair
from flowmotion.model.network import VelocityTransformer


def test_sample_training_pair_shapes_and_range():
    x1 = torch.randn(5, 4, 8)
    x0, t, x_t, u_t = sample_training_pair(x1)
    assert x0.shape == x1.shape
    assert x_t.shape == x1.shape
    assert u_t.shape == x1.shape
    assert t.shape == (5,)
    assert torch.all(t >= 0) and torch.all(t < 1)
    assert torch.allclose(u_t, x1 - x0)


def test_x_t_interpolates_between_x0_and_x1_at_the_endpoints():
    x1 = torch.randn(3, 4, 8)
    x0, _, _, _ = sample_training_pair(x1)
    t0 = torch.zeros(3)
    t1 = torch.ones(3)
    x_at_0 = (1 - t0.view(-1, 1, 1)) * x0 + t0.view(-1, 1, 1) * x1
    x_at_1 = (1 - t1.view(-1, 1, 1)) * x0 + t1.view(-1, 1, 1) * x1
    assert torch.allclose(x_at_0, x0)
    assert torch.allclose(x_at_1, x1)


def _tiny_model(D=8, K=3, H=3):
    return VelocityTransformer(
        D=D,
        d_model=16,
        n_layers=1,
        n_heads=2,
        dim_ff=32,
        K=K,
        H=H,
        num_subjects=2,
        num_actions=2,
        dropout=0.0,
        p_label_dropout=0.0,
    )


def test_flow_matching_loss_is_scalar_and_differentiable():
    model = _tiny_model()
    past = torch.randn(4, 3, 8)
    target = torch.randn(4, 3, 8)
    subject_id = torch.zeros(4, dtype=torch.long)
    action_id = torch.zeros(4, dtype=torch.long)

    loss = flow_matching_loss(model, past, target, subject_id, action_id)
    assert loss.ndim == 0
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0


def test_loss_decreases_when_overfitting_a_single_batch():
    torch.manual_seed(0)
    model = _tiny_model()
    past = torch.randn(8, 3, 8)
    target = torch.randn(8, 3, 8)
    subject_id = torch.randint(0, 2, (8,))
    action_id = torch.randint(0, 2, (8,))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    first_loss = None
    last_loss = None
    for step in range(50):
        loss = flow_matching_loss(model, past, target, subject_id, action_id)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()

    assert last_loss < first_loss
