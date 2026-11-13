"""Shared experiment utilities: prompts, model setup, CLI arguments and timed generation."""

import argparse
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from src.baseline import GenerationOutput
from src.models import LanguageModel, load_pair
from src.utils import Timer, get_device, set_seed

PROMPTS: List[str] = [
    "The capital of France is",
    "Once upon a time",
    "In machine learning, the main goal of regularization is",
    "def fibonacci(n):",
    "The theory of relativity states that",
    "Dear Sir or Madam, I am writing to",
    "The best way to learn a new language is",
    "Scientists have recently discovered that",
]


@dataclass
class Setup:
    """Models, tokenizer and device used by an experiment."""

    target: LanguageModel
    draft: LanguageModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the arguments shared by all experiment scripts."""
    parser.add_argument("--target", default="gpt2-medium")
    parser.add_argument("--draft", default="gpt2")
    parser.add_argument("--device", default=None, help="cpu, mps or cuda (default: best available)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    return parser


def setup(args: argparse.Namespace) -> Setup:
    """Seed everything and load the target/draft pair on the requested device."""
    set_seed(args.seed)
    device = get_device(args.device)
    target, draft, tokenizer = load_pair(args.target, args.draft, device)
    return Setup(target, draft, tokenizer, device)


def encode_prompts(tokenizer: PreTrainedTokenizerBase, prompts: List[str]) -> List[torch.Tensor]:
    """Tokenize prompts into 1D tensors of token ids."""
    return [tokenizer(p, return_tensors="pt").input_ids[0] for p in prompts]


def timed_run(
    fn: Callable[[], GenerationOutput], device: torch.device
) -> Tuple[GenerationOutput, float]:
    """Run a generation closure and return its output with the elapsed wall-clock seconds."""
    with Timer(device) as timer:
        output = fn()
    return output, timer.elapsed


def warmup(setup_: Setup, prompt_ids: torch.Tensor, n_tokens: int = 8) -> None:
    """Run both models briefly so that kernel compilation does not pollute timings."""
    for model in (setup_.target, setup_.draft):
        ids = prompt_ids.clone()
        for _ in range(n_tokens):
            model.next_token_logits(ids)
            