"""Metrics: empirical acceptance rate, tokens per call, call reduction, cost coefficient c, and distribution distances."""

import statistics
from typing import Dict, List, Sequence

import torch

from src.baseline import GenerationOutput
from src.models import LanguageModel
from src.utils import Timer


def acceptance_rate(outputs: Sequence[GenerationOutput]) -> float:
    """Empirical alpha: accepted drafts divided by verified drafts (drafts after a rejection are never verified)."""
    verified = sum(o.verified for o in outputs)
    if verified == 0:
        return 0.0
    return sum(o.accepted for o in outputs) / verified


def mean_tokens_per_iteration(outputs: Sequence[GenerationOutput]) -> float:
    """Average number of tokens produced per target call across runs."""
    counts = [n for o in outputs for n in o.tokens_per_iteration]
    return statistics.fmean(counts) if counts else 0.0


def call_reduction(outputs: Sequence[GenerationOutput]) -> float:
    """Generated tokens divided by target calls: hardware-independent speedup (1.0 for plain decoding)."""
    calls = sum(o.target_calls for o in outputs)
    if calls == 0:
        return 0.0
    return sum(len(o.tokens) for o in outputs) / calls


def summarize(outputs: Sequence[GenerationOutput]) -> Dict[str, float]:
    """Aggregate the main statistics of several generation runs into a flat dictionary."""
    return {
        "alpha": acceptance_rate(outputs),
        "tokens_per_iteration": mean_tokens_per_iteration(outputs),
        "call_reduction": call_reduction(outputs),
        "target_calls": float(sum(o.target_calls for o in outputs)),
        "draft_calls": float(sum(o.draft_calls for o in outputs)),
        "new_tokens": float(sum(len(o.tokens) for o in outputs)),
    }


def _median_forward_time(
    model: LanguageModel, context_len: int, n_runs: int, warmup: int
) -> float:
    """Median wall-clock time of one next-token forward pass on a random context of the given length."""
    ids = torch.randint(0, model.vocab_size, (context_len,))
    for _ in range(warmup):
        model.next_token_logits(ids)
    times: List[float] = []
    for _ in range(n_runs):
        with Timer(model.device) as timer:
            model.next_token_logits(ids)
        times.append(timer.elapsed)
    return statistics.median(times)


def estimate_cost_coefficient(
    target: LanguageModel,
    draft: LanguageModel,
    context_len: int = 64,
    n_runs: int = 30,
    warmup: int = 5,
) -> float:
    """Estimate c, the ratio between one draft forward pass and one target forward pass."""
    draft_time = _median_forward_time(draft, context_len, n_runs, warmup)
    target_time = _median_forward_time(target, context_len, n_runs, warmup)
    return draft_time / target_time


def empirical_distribution(tokens: Sequence[int], vocab_size: int) -> torch.Tensor:
    """Normalized histogram of sampled token ids over the vocabulary."""
    counts = torch.bincount(torch.tensor(list(tokens)), minlength=vocab_size).double()
    return counts / counts.sum()


def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total variation distance 0.5 * sum |p - q| between two distributions."""
    return 0.5 * (p.double() - q.double()).abs().sum().item()
