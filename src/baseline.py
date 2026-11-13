"""Standard autoregressive decoding, used as the reference for both distribution and speed."""

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from src.models import LanguageModel
from src.sampling import logits_to_probs, sample_from_probs


@dataclass
class GenerationOutput:
    """Result of a generation run: new tokens plus counters used for speed analysis."""

    tokens: List[int]
    target_calls: int = 0
    draft_calls: int = 0
    drafted: int = 0
    accepted: int = 0
    verified: int = 0
    tokens_per_iteration: List[int] = field(default_factory=list)


def autoregressive_generate(
    model: LanguageModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    eos_token_id: Optional[int] = None,
) -> GenerationOutput:
    """Sample max_new_tokens tokens one at a time from the model, one forward pass per token."""
    budget = min(max_new_tokens, model.max_positions - prompt_ids.size(0))
    ids = prompt_ids.to(model.device)
    output = GenerationOutput(tokens=[])

    for _ in range(budget):
        logits = model.next_token_logits(ids)
        probs = logits_to_probs(logits, temperature, top_k, top_p)
        token = int(sample_from_probs(probs, generator).item())
        output.target_calls += 1
        output.tokens.append(token)
        output.tokens_per_iteration.append(1)
        ids = torch.cat([ids, torch.tensor([token], device=model.device)])
        if eos_token_id is not None and token == eos_token_id:
            break

    return output
