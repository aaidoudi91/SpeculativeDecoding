"""Losslessness tests: speculative decoding must sample sequences from the target distribution exactly.
Run: pytest tests/test_losslessness.py -v."""

import itertools
from collections import Counter

import pytest
import torch

from src.baseline import autoregressive_generate
from src.sampling import logits_to_probs
from src.speculative import speculative_generate

VOCAB = 4
SEQ_LEN = 3
N_SAMPLES = 5000


class ToyLM:
    """Markov toy language model exposing the same interface as LanguageModel."""

    def __init__(self, table: torch.Tensor) -> None:
        self.table = table
        self.device = torch.device("cpu")
        self.vocab_size = table.size(0)
        self.max_positions = 64

    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.table[input_ids]

    def next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logits(input_ids)[-1]


def make_toy_pair():
    torch.manual_seed(0)
    target_table = torch.randn(VOCAB, VOCAB) * 1.5
    draft_table = target_table + torch.randn(VOCAB, VOCAB) * 1.0
    return ToyLM(target_table), ToyLM(draft_table)


def exact_sequence_probs(model, prompt_token, temperature, top_k):
    """Exact probability of every length-SEQ_LEN continuation under the model."""
    probs = {}
    for seq in itertools.product(range(VOCAB), repeat=SEQ_LEN):
        prob, prev = 1.0, prompt_token
        for token in seq:
            dist = logits_to_probs(model.table[prev], temperature, top_k)
            prob *= dist[token].item()
            prev = token
        probs[seq] = prob
    return probs


def chi_square_statistic(counts, exact, n):
    """Return (statistic, degrees of freedom); cells with zero exact probability must stay empty."""
    stat, cells = 0.0, 0
    for seq, prob in exact.items():
        observed = counts.get(seq, 0)
        if prob == 0.0:
            assert observed == 0
            continue
        expected = n * prob
        stat += (observed - expected) ** 2 / expected
        cells += 1
    return stat, cells - 1


def chi_square_threshold(df):
    """Conservative acceptance threshold: mean + 5 standard deviations of a chi-square variable."""
    return df + 5.0 * (2.0 * df) ** 0.5


def sample_sequences(generate_fn, n, seed):
    generator = torch.Generator().manual_seed(seed)
    prompt = torch.tensor([0])
    counts = Counter()
    for _ in range(n):
        out = generate_fn(prompt, generator)
        counts[tuple(out.tokens)] += 1
    return counts


@pytest.mark.parametrize(
    "gamma,temperature,top_k",
    [(1, 1.0, None), (3, 1.0, None), (4, 0.7, 3)],
)
def test_speculative_matches_target_distribution(gamma, temperature, top_k):
    target, draft = make_toy_pair()
    exact = exact_sequence_probs(target, 0, temperature, top_k)

    def run(prompt, generator):
        return speculative_generate(
            target, draft, prompt, SEQ_LEN, gamma, temperature, top_k, None, generator
        )

    counts = sample_sequences(run, N_SAMPLES, seed=123)
    stat, df = chi_square_statistic(counts, exact, N_SAMPLES)
    assert stat < chi_square_threshold(df), f"chi2={stat:.1f} too large for df={df}"


def test_test_has_power_against_draft_only_sampling():
    target, draft = make_toy_pair()
    exact = exact_sequence_probs(target, 0, 1.0, None)

    def run(prompt, generator):
        return autoregressive_generate(draft, prompt, SEQ_LEN, 1.0, None, None, generator)

    counts = sample_sequences(run, N_SAMPLES, seed=123)
    stat, df = chi_square_statistic(counts, exact, N_SAMPLES)
    assert stat > chi_square_threshold(df)


def test_baseline_matches_target_distribution():
    target, _ = make_toy_pair()
    exact = exact_sequence_probs(target, 0, 1.0, None)

    def run(prompt, generator):
        return autoregressive_generate(target, prompt, SEQ_LEN, 1.0, None, None, generator)

    counts = sample_sequences(run, N_SAMPLES, seed=7)
    stat, df = chi_square_statistic(counts, exact, N_SAMPLES)
    assert stat < chi_square_threshold(df)
