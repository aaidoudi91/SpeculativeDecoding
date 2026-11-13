"""Greedy equivalence and bookkeeping tests: at temperature 0 speculative decoding must equal the target alone.
Run: pytest tests/test_greedy_equivalence.py -v"""

import pytest
import torch

from src.baseline import autoregressive_generate
from src.models import load_pair
from src.speculative import speculative_generate

VOCAB = 5


class ToyLM:
    """Minimal Markov language model exposing the LanguageModel interface."""

    def __init__(self, table: torch.Tensor, max_positions: int = 64) -> None:
        self.table = table
        self.device = torch.device("cpu")
        self.vocab_size = table.size(0)
        self.max_positions = max_positions

    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.table[input_ids]

    def next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logits(input_ids)[-1]


def make_toy_pair(max_positions: int = 64):
    torch.manual_seed(1)
    target_table = torch.randn(VOCAB, VOCAB) * 2.0
    draft_table = target_table + torch.randn(VOCAB, VOCAB) * 1.5
    return ToyLM(target_table, max_positions), ToyLM(draft_table, max_positions)


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
@pytest.mark.parametrize("prompt", [[0], [3, 1, 4], [2, 2]])
def test_toy_greedy_equivalence(gamma, prompt):
    target, draft = make_toy_pair()
    ids = torch.tensor(prompt)
    base = autoregressive_generate(target, ids, 25, temperature=0.0)
    spec = speculative_generate(target, draft, ids, 25, gamma, temperature=0.0)
    assert spec.tokens == base.tokens
    assert spec.target_calls <= base.target_calls


@pytest.mark.parametrize("gamma", [1, 3, 6])
def test_toy_bookkeeping_is_consistent(gamma):
    target, draft = make_toy_pair()
    generator = torch.Generator().manual_seed(0)
    out = speculative_generate(
        target, draft, torch.tensor([0]), 40, gamma, 1.0, None, None, generator
    )
    assert len(out.tokens) == 40
    assert sum(out.tokens_per_iteration) == len(out.tokens)
    assert out.target_calls == len(out.tokens_per_iteration)
    assert all(1 <= n <= gamma + 1 for n in out.tokens_per_iteration)
    assert 0 <= out.accepted <= out.verified <= out.drafted


def test_context_limit_is_respected():
    target, draft = make_toy_pair(max_positions=8)
    prompt = torch.tensor([0, 1, 2, 3, 4])
    spec = speculative_generate(target, draft, prompt, 10, gamma=4, temperature=1.0)
    base = autoregressive_generate(target, prompt, 10, temperature=1.0)
    assert len(spec.tokens) == 3
    assert len(base.tokens) == 3


@pytest.fixture(scope="module")
def gpt2_pair():
    device = torch.device("cpu")
    target, draft, tokenizer = load_pair("gpt2-medium", "gpt2", device)
    return target, draft, tokenizer


@pytest.mark.parametrize("gamma", [1, 4])
@pytest.mark.parametrize("text", ["The capital of France is", "Once upon a time"])
def test_gpt2_greedy_equivalence(gpt2_pair, gamma, text):
    target, draft, tokenizer = gpt2_pair
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    base = autoregressive_generate(target, ids, 20, temperature=0.0)
    spec = speculative_generate(target, draft, ids, 20, gamma, temperature=0.0)
    assert spec.tokens == base.tokens
    assert spec.target_calls <= base.target_calls
    assert sum(spec.tokens_per_iteration) == len(spec.tokens)
