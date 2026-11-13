"""Experiment 1: check on real GPT-2 that speculative decoding samples sequences from the target distribution.
Run: python -m experiments.exp1_losslessness --device mps --n-samples 1000 --repeats 5."""

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.common import PROMPTS, Setup, add_common_args, encode_prompts, setup
from src.baseline import autoregressive_generate
from src.models import LanguageModel
from src.sampling import logits_to_probs
from src.speculative import speculative_generate
from src.utils import make_generator, save_json

Seq = Tuple[int, ...]


def exact_sequence_probs(
    model: LanguageModel, ids: torch.Tensor, depth: int, temperature: float, top_k: int
) -> Dict[Seq, float]:
    """Exact probability of every continuation of the given depth under the model (small support only)."""
    if depth == 0:
        return {(): 1.0}
    probs = logits_to_probs(model.next_token_logits(ids), temperature, top_k)
    result: Dict[Seq, float] = {}
    for token in probs.nonzero().flatten().tolist():
        extended = torch.cat([ids.to(model.device), torch.tensor([token], device=model.device)])
        for seq, p in exact_sequence_probs(model, extended, depth - 1, temperature, top_k).items():
            result[(token,) + seq] = probs[token].item() * p
    return result


def chi_square_pvalue(stat: float, df: int) -> float:
    """Upper-tail p-value of a chi-square statistic (Wilson-Hilferty approximation)."""
    if df <= 0:
        return float("nan")
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def goodness_of_fit(counts: Counter, exact: Dict[Seq, float], n: int) -> Dict[str, float]:
    """Chi-square test against the exact law (rare cells pooled) plus total variation distance."""
    stat, bins = 0.0, 0
    pooled_expected, pooled_observed = 0.0, 0
    for seq, prob in exact.items():
        expected, observed = n * prob, counts.get(seq, 0)
        if expected >= 5:
            stat += (observed - expected) ** 2 / expected
            bins += 1
        else:
            pooled_expected += expected
            pooled_observed += observed
    if pooled_expected >= 1:
        stat += (pooled_observed - pooled_expected) ** 2 / pooled_expected
        bins += 1
    outside = sum(c for seq, c in counts.items() if seq not in exact)
    keys = set(exact) | set(counts)
    tv = 0.5 * sum(abs(counts.get(k, 0) / n - exact.get(k, 0.0)) for k in keys)
    return {
        "chi2": stat,
        "df": bins - 1,
        "p_value": chi_square_pvalue(stat, bins - 1),
        "tv": tv,
        "outside_support": outside,
    }


def sample_counts(fn, n: int) -> Counter:
    """Run a generation closure n times and count the produced token sequences."""
    counts: Counter = Counter()
    for _ in range(n):
        counts[tuple(fn().tokens)] += 1
    return counts


def plot_comparison(
    s: Setup, exact: Dict[Seq, float], spec: Counter, base: Counter, n: int, path: Path
) -> None:
    """Bar chart of exact, speculative and baseline frequencies for the most likely sequences."""
    top = sorted(exact, key=exact.get, reverse=True)[:12]
    x = range(len(top))
    width = 0.27
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar([i - width for i in x], [exact[t] for t in top], width, label="Exact (target)")
    ax.bar(list(x), [spec.get(t, 0) / n for t in top], width, label="Speculative")
    ax.bar([i + width for i in x], [base.get(t, 0) / n for t in top], width, label="Autoregressive")
    ax.set_xticks(list(x))
    ax.set_xticklabels([repr(s.tokenizer.decode(list(t))) for t in top], rotation=45, ha="right")
    ax.set_ylabel("Sequence probability")
    ax.set_title("Most likely 3-token continuations: exact vs sampled frequencies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    s = setup(args)
    prompt = encode_prompts(s.tokenizer, [PROMPTS[args.prompt_index]])[0]
    n = args.n_samples

    exact = exact_sequence_probs(s.target, prompt, args.seq_len, args.temperature, args.top_k)
    print(f"Exact law computed over {len(exact)} sequences")

    spec_total: Counter = Counter()
    base_total: Counter = Counter()
    runs = []
    for r in range(args.repeats):
        gen = make_generator(args.seed + r, s.device)
        spec = sample_counts(
            lambda: speculative_generate(
                s.target, s.draft, prompt, args.seq_len, args.gamma,
                args.temperature, args.top_k, None, gen,
            ),
            n,
        )
        base = sample_counts(
            lambda: autoregressive_generate(
                s.target, prompt, args.seq_len, args.temperature, args.top_k, None, gen
            ),
            n,
        )
        spec_total.update(spec)
        base_total.update(base)
        run = {
            "seed": args.seed + r,
            "speculative": goodness_of_fit(spec, exact, n),
            "autoregressive": goodness_of_fit(base, exact, n),
        }
        runs.append(run)
        print(
            f"run {r}: spec p={run['speculative']['p_value']:.3f} "
            f"TV={run['speculative']['tv']:.4f} | base p={run['autoregressive']['p_value']:.3f} "
            f"TV={run['autoregressive']['tv']:.4f}"
        )

    total = n * args.repeats
    pooled = {
        "speculative": goodness_of_fit(spec_total, exact, total),
        "autoregressive": goodness_of_fit(base_total, exact, total),
    }
    for name, r in pooled.items():
        print(f"pooled {name:15s} chi2={r['chi2']:.1f} df={r['df']} p={r['p_value']:.3f} TV={r['tv']:.4f}")

    results = {"args": vars(args), "runs": runs, "pooled": pooled}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(results, str(out / "exp1_losslessness.json"))
    plot_comparison(s, exact, spec_total, base_total, total, out / "exp1_losslessness.png")


if __name__ == "__main__":
    main()
