"""Unit tests for src/sampling.py: distribution shaping, residual distribution and the verification rule.
Run: pytest tests/test_sampling.py -v."""

import math

import pytest
import torch

from src.sampling import (
    acceptance_probability,
    expected_acceptance_rate,
    logits_to_probs,
    residual_distribution,
    speculative_sample_token,
)

PROBS = torch.tensor([0.5, 0.3, 0.15, 0.05])


def test_probs_sum_to_one():
    probs = logits_to_probs(torch.randn(3, 10), temperature=0.8)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(3))


def test_temperature_zero_is_one_hot_argmax():
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 1.0]])
    probs = logits_to_probs(logits, temperature=0.0)
    expected = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    torch.testing.assert_close(probs, expected)


def test_temperature_one_recovers_softmax():
    logits = torch.log(PROBS)
    torch.testing.assert_close(logits_to_probs(logits, temperature=1.0), PROBS)


def test_top_k_keeps_k_tokens():
    probs = logits_to_probs(torch.log(PROBS), top_k=2)
    assert (probs > 0).sum().item() == 2
    torch.testing.assert_close(probs, torch.tensor([0.625, 0.375, 0.0, 0.0]))


def test_top_p_keeps_smallest_nucleus():
    probs = logits_to_probs(torch.log(PROBS), top_p=0.7)
    torch.testing.assert_close(probs, torch.tensor([0.625, 0.375, 0.0, 0.0]))


def test_residual_is_a_distribution_supported_where_p_exceeds_q():
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.2, 0.5, 0.3])
    residual = residual_distribution(p, q)
    torch.testing.assert_close(residual.sum(), torch.tensor(1.0))
    torch.testing.assert_close(residual, torch.tensor([1.0, 0.0, 0.0]))


def test_residual_falls_back_to_p_when_equal():
    torch.testing.assert_close(residual_distribution(PROBS, PROBS), PROBS)


def test_acceptance_probability_cases():
    p = torch.tensor([0.6, 0.4])
    q = torch.tensor([0.3, 0.7])
    assert acceptance_probability(p, q, 0) == 1.0
    assert acceptance_probability(p, q, 1) == pytest.approx(0.4 / 0.7)


def test_expected_acceptance_rate_known_values():
    p = torch.tensor([0.6, 0.4])
    q = torch.tensor([0.3, 0.7])
    assert expected_acceptance_rate(p, q) == pytest.approx(0.7)
    assert expected_acceptance_rate(p, p) == pytest.approx(1.0)
    disjoint_p = torch.tensor([1.0, 0.0])
    disjoint_q = torch.tensor([0.0, 1.0])
    assert expected_acceptance_rate(disjoint_p, disjoint_q) == pytest.approx(0.0)


def test_identical_distributions_always_accept():
    generator = torch.Generator().manual_seed(0)
    for token in range(4):
        out, accepted = speculative_sample_token(PROBS, PROBS, token, generator)
        assert accepted and out == token


def test_disjoint_supports_always_reject_and_follow_p():
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.0, 1.0])
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        out, accepted = speculative_sample_token(p, q, 1, generator)
        assert not accepted and out == 0


def test_acceptance_frequency_matches_beta():
    p = torch.tensor([0.6, 0.4])
    q = torch.tensor([0.3, 0.7])
    generator = torch.Generator().manual_seed(1)
    n = 20000
    draws = torch.multinomial(q, n, replacement=True, generator=generator)
    accepted = sum(speculative_sample_token(p, q, int(x), generator)[1] for x in draws)
    beta = expected_acceptance_rate(p, q)
    std = math.sqrt(beta * (1 - beta) / n)
    assert abs(accepted / n - beta) < 5 * std
