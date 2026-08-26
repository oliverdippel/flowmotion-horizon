from __future__ import annotations

import pytest
import torch

from flowmotion.data.transforms import Normalizer


def test_fit_recovers_known_mean_and_std():
    torch.manual_seed(0)
    true_mean = torch.tensor([1.0, -2.0, 0.5])
    true_std = torch.tensor([2.0, 0.5, 1.0])
    features = true_mean + true_std * torch.randn(20000, 3)

    normalizer = Normalizer.fit(features)
    assert torch.allclose(normalizer.mean, true_mean, atol=0.05)
    assert torch.allclose(normalizer.std, true_std, atol=0.05)


def test_fit_streaming_matches_fit_on_the_same_data():
    torch.manual_seed(0)
    features = torch.randn(5000, 4) * torch.tensor([1.0, 3.0, 0.1, 5.0])

    from_fit = Normalizer.fit(features)
    windows = [features[i : i + 50] for i in range(0, features.shape[0], 50)]
    from_streaming = Normalizer.fit_streaming(windows)

    assert torch.allclose(from_fit.mean, from_streaming.mean, atol=1e-3)
    assert torch.allclose(from_fit.std, from_streaming.std, atol=1e-3)


def test_near_frozen_channel_std_is_floored_not_left_near_zero():
    # regression test: a channel that's almost constant across the whole corpus (e.g.
    # AMASS spine joints barely rotate) must NOT get a std near machine epsilon -- that
    # divides tiny floating-point noise up into huge normalized values and destabilizes
    # training (this happened on a real run: loss exploded into the millions).
    torch.manual_seed(0)
    frozen = 0.99 + torch.randn(10000, 1) * 1e-7  # genuine std ~1e-7
    varying = torch.randn(10000, 1) * 0.3
    features = torch.cat([frozen, varying], dim=1)

    normalizer = Normalizer.fit(features)
    assert normalizer.std[0].item() >= 1e-2 - 1e-6  # floored, not ~1e-7 (float32 tolerance)
    assert normalizer.std[1].item() == pytest.approx(0.3, abs=0.02)  # real variation untouched

    # and the practical consequence: normalizing a slightly-off-mean sample doesn't explode
    sample = torch.tensor([[0.9900001, 0.0]])
    normalized = normalizer.transform(sample)
    assert normalized[0, 0].abs().item() < 10.0


def test_transform_inverse_transform_round_trip():
    normalizer = Normalizer(mean=torch.tensor([1.0, 2.0]), std=torch.tensor([0.5, 2.0]))
    x = torch.tensor([[3.0, -1.0], [0.0, 5.0]])
    assert torch.allclose(normalizer.inverse_transform(normalizer.transform(x)), x, atol=1e-6)


def test_fit_streaming_raises_on_empty_input():
    with pytest.raises(ValueError):
        Normalizer.fit_streaming([])
