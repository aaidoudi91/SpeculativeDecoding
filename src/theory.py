"""Closed-form predictions of Leviathan et al. (2023), Section 3: tokens per call, speedup, cost, optimal gamma."""

from typing import Tuple


def expected_tokens_per_iteration(alpha: float, gamma: int) -> float:
    """Expected tokens produced per target call: (1 - alpha^(gamma+1)) / (1 - alpha), Equation (1)."""
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def expected_speedup(alpha: float, gamma: int, c: float) -> float:
    """Expected wall-time improvement factor, Theorem 3.8, with c the draft/target cost ratio."""
    return expected_tokens_per_iteration(alpha, gamma) / (gamma * c + 1.0)


def expected_operations_increase(alpha: float, gamma: int, c_hat: float) -> float:
    """Expected factor of extra arithmetic operations, Theorem 3.11, with c_hat the draft/target FLOPs ratio."""
    return (gamma * c_hat + gamma + 1.0) / expected_tokens_per_iteration(alpha, gamma)


def minimum_speedup(alpha: float, c: float) -> float:
    """Guaranteed improvement (1 + alpha) / (1 + c) obtained with gamma = 1, valid when alpha > c (Corollary 3.9)."""
    return (1.0 + alpha) / (1.0 + c)


def optimal_gamma(alpha: float, c: float, max_gamma: int = 32) -> Tuple[int, float]:
    """Return the integer gamma in [1, max_gamma] maximizing the expected speedup, and that speedup."""
    best_gamma, best_speedup = 1, expected_speedup(alpha, 1, c)
    for gamma in range(2, max_gamma + 1):
        speedup = expected_speedup(alpha, gamma, c)
        if speedup > best_speedup:
            best_gamma, best_speedup = gamma, speedup
    return best_gamma, best_speedup
