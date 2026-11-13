"""Sampling utilities: logits to distributions and the speculative sampling rule (Leviathan et al., 2023)."""

from typing import Optional, Tuple

import torch


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Convert logits (..., vocab) into an adjusted distribution; temperature 0 gives a one-hot argmax."""
    logits = logits.float()

    if temperature == 0.0:
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return one_hot

    logits = logits / temperature

    if top_k is not None and top_k < logits.size(-1):
        kth_value = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth_value, float("-inf"))

    probs = torch.softmax(logits, dim=-1)

    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Drop a token if the mass strictly before it already reaches top_p
        remove_sorted = (cumulative - sorted_probs) >= top_p
        remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
        probs = probs.masked_fill(remove, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    return probs


def sample_from_probs(
    probs: torch.Tensor, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Draw one token index per distribution along the last dimension."""
    flat = probs.reshape(-1, probs.size(-1))
    tokens = torch.multinomial(flat, num_samples=1, generator=generator)
    return tokens.reshape(probs.shape[:-1])


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Return p' = norm(max(0, p - q)), the correction distribution used after a rejection."""
    residual = torch.clamp(p - q, min=0.0)
    total = residual.sum(dim=-1, keepdim=True)
    # total == 0 only if p == q, where a rejection cannot happen; fall back to p for safety
    safe = total > 0
    return torch.where(safe, residual / total.clamp_min(1e-30), p)


def acceptance_probability(p: torch.Tensor, q: torch.Tensor, token: int) -> float:
    """Probability min(1, p(x) / q(x)) of accepting a token x drafted from q."""
    q_x = q[token].item()
    if q_x <= 0.0:
        return 0.0
    return min(1.0, p[token].item() / q_x)


def speculative_sample_token(
    p: torch.Tensor,
    q: torch.Tensor,
    token: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, bool]:
    """Verify one drafted token: keep it with prob min(1, p/q), else resample from p'. Returns (token, accepted)."""
    r = torch.rand((), generator=generator, device=p.device).item()
    if r < acceptance_probability(p, q, token):
        return token, True
    corrected = sample_from_probs(residual_distribution(p, q), generator)
    return int(corrected.item()), False


def expected_acceptance_rate(p: torch.Tensor, q: torch.Tensor) -> float:
    """Return beta = sum_x min(p(x), q(x)) = 1 - D_LK(p, q) (Theorem 3.5)."""
    return torch.minimum(p, q).sum(dim=-1).item()
