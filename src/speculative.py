"""Speculative decoding (Algorithm 1 of Leviathan et al., 2023): draft with a small model, verify with the target."""

from typing import List, Optional

import torch

from src.baseline import GenerationOutput
from src.models import LanguageModel
from src.sampling import logits_to_probs, sample_from_probs, speculative_sample_token


def speculative_generate(
    target: LanguageModel,
    draft: LanguageModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int = 4,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    eos_token_id: Optional[int] = None,
) -> GenerationOutput:
    """Generate max_new_tokens tokens whose distribution matches the target model alone, using gamma drafts per step."""
    device = target.device
    budget = min(max_new_tokens, target.max_positions - prompt_ids.size(0))
    ids = prompt_ids.to(device)
    output = GenerationOutput(tokens=[])

    while len(output.tokens) < budget:
        remaining = budget - len(output.tokens)
        # Keep at least one slot for the correction/bonus token and respect the context size
        gamma_eff = max(0, min(gamma, remaining - 1))
        prefix_len = ids.size(0)

        # Step 1: draft gamma_eff tokens autoregressively with the small model
        draft_tokens: List[int] = []
        q_list: List[torch.Tensor] = []
        cur = ids
        for _ in range(gamma_eff):
            q = logits_to_probs(draft.next_token_logits(cur), temperature, top_k, top_p)
            x = int(sample_from_probs(q, generator).item())
            q_list.append(q)
            draft_tokens.append(x)
            cur = torch.cat([cur, torch.tensor([x], device=device)])
        output.draft_calls += gamma_eff
        output.drafted += gamma_eff

        # Step 2: one parallel target pass gives p_1, ..., p_{gamma+1}
        logits = target.logits(cur)[prefix_len - 1 :]
        p_all = logits_to_probs(logits, temperature, top_k, top_p)
        output.target_calls += 1

        # Step 3: accept drafted tokens until the first rejection
        new_tokens: List[int] = []
        all_accepted = True
        for i in range(gamma_eff):
            token, accepted = speculative_sample_token(
                p_all[i], q_list[i], draft_tokens[i], generator
            )
            output.verified += 1
            new_tokens.append(token)
            if accepted:
                output.accepted += 1
            else:
                all_accepted = False
                break

        # Step 4: if every draft was accepted, add a bonus token from the last target distribution
        if all_accepted:
            bonus = int(sample_from_probs(p_all[gamma_eff], generator).item())
            new_tokens.append(bonus)

        stop = False
        if eos_token_id is not None and eos_token_id in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(eos_token_id) + 1]
            stop = True

        output.tokens.extend(new_tokens)
        output.tokens_per_iteration.append(len(new_tokens))
        ids = torch.cat([ids, torch.tensor(new_tokens, device=device)])
        if stop:
            break

    return output
