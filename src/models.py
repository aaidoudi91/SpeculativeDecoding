"""Model loading: thin wrapper around Hugging Face GPT-2 models exposing per-position logits."""

from typing import Tuple

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, PreTrainedTokenizerBase


class LanguageModel:
    """Wraps a causal LM and returns the logits for every position of a 1D token sequence."""

    def __init__(self, model: GPT2LMHeadModel, device: torch.device, name: str) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.name = name
        self.vocab_size = model.config.vocab_size
        self.max_positions = model.config.n_positions

    @torch.inference_mode()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (seq_len, vocab) for a 1D tensor of token ids."""
        if input_ids.size(0) > self.max_positions:
            raise ValueError(
                f"Sequence length {input_ids.size(0)} exceeds the context size {self.max_positions}"
            )
        out = self.model(input_ids.unsqueeze(0).to(self.device))
        return out.logits[0].float()

    def next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the logits (vocab,) predicting the token that follows the sequence."""
        return self.logits(input_ids)[-1]

    def num_parameters(self) -> int:
        """Number of parameters of the underlying model."""
        return sum(p.numel() for p in self.model.parameters())


def load_model(name: str, device: torch.device) -> LanguageModel:
    """Load a pretrained GPT-2 checkpoint (e.g. 'gpt2', 'gpt2-medium', 'gpt2-large') in float32."""
    model = GPT2LMHeadModel.from_pretrained(name, torch_dtype=torch.float32)
    return LanguageModel(model, device, name)


def load_tokenizer(name: str = "gpt2") -> PreTrainedTokenizerBase:
    """Load the GPT-2 tokenizer, shared by all models of the family."""
    return AutoTokenizer.from_pretrained(name)


def load_pair(
    target_name: str, draft_name: str, device: torch.device
) -> Tuple[LanguageModel, LanguageModel, PreTrainedTokenizerBase]:
    """Load target and draft models plus their shared tokenizer, checking that vocabularies match."""
    target = load_model(target_name, device)
    draft = load_model(draft_name, device)
    if target.vocab_size != draft.vocab_size:
        raise ValueError("Target and draft models must share the same vocabulary")
    return target, draft, load_tokenizer(draft_name)
