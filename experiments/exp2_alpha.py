"""Experiment 2: acceptance rate alpha versus sampling temperature and draft model.
Run: python -m experiments.exp2_alpha --device mps"""

import argparse
import statistics
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.common import PROMPTS, add_common_args, encode_prompts
from src.baseline import autoregressive_generate
from src.metrics import acceptance_rate, mean_tokens_per_iteration
from src.models import LanguageModel, load_model, load_tokenizer
from src.sampling import logits_to_probs
from src.speculative import speculative_generate
from src.theory import expected_tokens_per_iteration
from src.utils import get_device, make_generator, save_json, set_seed


def mean_and_se(values: List[float]) -> Tuple[float, float]:
    """Mean and standard error of a list of per-prompt values."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, statistics.stdev(values) / len(values) ** 0.5


def direct_beta(
    target: LanguageModel,
    draft: LanguageModel,
    prompt: torch.Tensor,
    tokens: List[int],
    temperature: float,
) -> float:
    """Average of sum_x min(p, q) over the positions of a target-generated sequence (Corollary 3.6)."""
    full = torch.cat([prompt.to(target.device), torch.tensor(tokens, device=target.device)])
    start = prompt.size(0) - 1
    p = logits_to_probs(target.logits(full)[start:-1], temperature)
    q = logits_to_probs(draft.logits(full)[start:-1], temperature)
    return torch.minimum(p, q).sum(dim=-1).mean().item()


def plot_alpha(results: list, path: Path) -> None:
    """Alpha versus temperature, one curve per draft model, direct estimate and speculative-run estimate."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, name in enumerate(sorted({r["draft"] for r in results})):
        rows = sorted((r for r in results if r["draft"] == name), key=lambda r: r["temperature"])
        temps = [r["temperature"] for r in rows]
        color = f"C{i}"
        ax.errorbar(
            temps, [r["alpha_direct"] for r in rows], yerr=[r["alpha_direct_se"] for r in rows],
            marker="o", color=color, capsize=3, label=f"{name}: E[min(p,q)]",
        )
        ax.plot(
            temps, [r["alpha_empirical"] for r in rows],
            marker="x", linestyle="--", color=color, label=f"{name}: accepted/verified",
        )
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Acceptance rate alpha")
    ax.set_ylim(0, 1)
    ax.set_title("Acceptance rate vs temperature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--drafts", nargs="+", default=["distilgpt2", "gpt2"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5])
    parser.add_argument("--n-tokens", type=int, default=64)
    parser.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    parser.add_argument("--gamma", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    tokenizer = load_tokenizer("gpt2")
    target = load_model(args.target, device)
    drafts = {name: load_model(name, device) for name in args.drafts}
    prompts = encode_prompts(tokenizer, PROMPTS[: args.n_prompts])

    results = []
    for temperature in args.temperatures:
        gen = make_generator(args.seed, device)
        trajectories = [
            autoregressive_generate(
                target, prompt, args.n_tokens, temperature, generator=gen
            ).tokens
            for prompt in prompts
        ]
        for name, draft in drafts.items():
            betas = [
                direct_beta(target, draft, prompt, tokens, temperature)
                for prompt, tokens in zip(prompts, trajectories)
            ]
            outputs = [
                speculative_generate(
                    target, draft, prompt, args.n_tokens, args.gamma,
                    temperature, generator=gen,
                )
                for prompt in prompts
            ]
            alpha_direct, alpha_direct_se = mean_and_se(betas)
            alpha_emp = acceptance_rate(outputs)
            _, alpha_emp_se = mean_and_se([acceptance_rate([o]) for o in outputs])
            row = {
                "draft": name,
                "temperature": temperature,
                "alpha_direct": alpha_direct,
                "alpha_direct_se": alpha_direct_se,
                "alpha_empirical": alpha_emp,
                "alpha_empirical_se": alpha_emp_se,
                "tokens_per_iteration": mean_tokens_per_iteration(outputs),
                "tokens_per_iteration_predicted": expected_tokens_per_iteration(alpha_emp, args.gamma),
            }
            results.append(row)
            print(
                f"T={temperature:.1f} {name:11s} direct={alpha_direct:.3f}±{alpha_direct_se:.3f} "
                f"empirical={alpha_emp:.3f}±{alpha_emp_se:.3f} "
                f"tok/it={row['tokens_per_iteration']:.2f} (pred {row['tokens_per_iteration_predicted']:.2f})"
            )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json({"args": vars(args), "results": results}, str(out / "exp2_alpha.json"))
    plot_alpha(results, out / "exp2_alpha.png")


if __name__ == "__main__":
    main()
