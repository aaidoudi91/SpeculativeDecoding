"""Unit tests for src/theory.py: closed-form predictions checked against Table 1 of Leviathan et al. (2023).
Run: pytest tests/test_theory.py -v."""

import pytest

from src.theory import (
    expected_operations_increase,
    expected_speedup,
    expected_tokens_per_iteration,
    minimum_speedup,
    optimal_gamma,
)

# (alpha, gamma, operations factor, speed factor) from Table 1 with c = c_hat = 0
TABLE_1 = [
    (0.6, 2, 1.53, 1.96),
    (0.7, 3, 1.58, 2.53),
    (0.8, 2, 1.23, 2.44),
    (0.8, 5, 1.63, 3.69),
    (0.9, 2, 1.11, 2.71),
    (0.9, 10, 1.60, 6.86),
]


@pytest.mark.parametrize("alpha,gamma,operations,speed", TABLE_1)
def test_matches_table_1(alpha, gamma, operations, speed):
    assert expected_speedup(alpha, gamma, c=0.0) == pytest.approx(speed, abs=0.01)
    assert expected_operations_increase(alpha, gamma, c_hat=0.0) == pytest.approx(
        operations, abs=0.01
    )


def test_zero_acceptance_gives_one_token_per_call():
    assert expected_tokens_per_iteration(0.0, 5) == pytest.approx(1.0)


def test_full_acceptance_gives_gamma_plus_one_tokens():
    assert expected_tokens_per_iteration(1.0, 5) == pytest.approx(6.0)
    assert expected_tokens_per_iteration(0.999999, 5) == pytest.approx(6.0, abs=1e-3)


def test_speedup_with_zero_acceptance_is_a_slowdown():
    assert expected_speedup(0.0, 4, c=0.1) == pytest.approx(1.0 / 1.4)


def test_minimum_speedup_equals_gamma_one():
    alpha, c = 0.7, 0.05
    assert minimum_speedup(alpha, c) == pytest.approx(expected_speedup(alpha, 1, c))


def test_no_gain_when_alpha_below_cost():
    gamma, speedup = optimal_gamma(alpha=0.5, c=0.6)
    assert gamma == 1 and speedup < 1.0


def test_free_draft_prefers_largest_gamma():
    gamma, _ = optimal_gamma(alpha=0.8, c=0.0, max_gamma=20)
    assert gamma == 20


def test_optimal_gamma_is_interior_for_positive_cost():
    gamma, speedup = optimal_gamma(alpha=0.8, c=0.05)
    assert 1 < gamma < 32
    assert speedup == max(expected_speedup(0.8, g, 0.05) for g in range(1, 33))