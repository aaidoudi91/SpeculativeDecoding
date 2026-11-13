"""Experiment 3: measured speedup versus gamma, compared with the predictions of Theorem 3.8.
Run: python -m experiments.exp3_speedup_vs_gamma --device cpu --output-dir results/exp3_cpu."""

import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.common import PROMPTS, add_common_args, encode_prompts, setup, timed_run, warmup
from src.baseline import autoregressive_generate
from src.metrics import acceptance_rate, call_reduction, estimate_cost_coefficient
from src.speculative import speculative_generate
from src.theory import expected_speedup, expected_tokens_per_iteration, optimal_gamma
from src.utils import make_generator, save_json


def plot_speedup(rows: List[dict], c: float, path: Path) -> None:
    """Left: wall-time speedup, measured vs theory. Right: target-call reduction, measured vs theory."""
    gammas = [r["gamma"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(gammas, [r["speedup_measured"] for r in rows], "o-", label="Measured")
    ax1.plot(gammas, [r["speedup_predicted"] for r in rows], "s--", label=f"Theory (c={c:.2f})")
    ax1.axhline(1.0, color="gray", linewidth=1)
    ax1.set_xlabel("gamma")
    ax1.set_ylabel("Wall-time speedup")
    ax1.set_title("Wall-time speedup vs gamma")
    ax1.legend()

    ax2.plot(gammas, [r["call_reduction_measured"] for r in rows], "o-", label="Measured")
    ax2.plot(gammas, [r["call_reduction_predicted"] for r in rows], "s--", label="Theory (c=0)")
    ax2.set_xlabel("gamma")
    ax2.set_ylabel("Tokens per target call")
    ax2.set_title("Target-call reduction vs gamma")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--gammas", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-tokens", type=int, default=64)
    parser.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    parser.add_argument("--cost-context", type=int, default=48)
    args = parser.parse_args()

    s = setup(args)
    prompts = encode_prompts(s.tokenizer, PROMPTS[: args.n_prompts])
    warmup(s, prompts[0])
    c = estimate_cost_coefficient(s.target, s.draft, context_len=args.cost_context)
    print(f"Measured cost coefficient c = {c:.3f}")

    gen = make_generator(args.seed, s.device)
    base_time = 0.0
    for prompt in prompts:
        _, elapsed = timed_run(
            lambda p=prompt: autoregressive_generate(
                s.target, p, args.n_tokens, args.temperature, generator=gen
            ),
            s.device,
        )
        base_time += elapsed
    print(f"Baseline total time: {base_time:.2f}s")

    rows = []
    for gamma in args.gammas:
        outputs, spec_time = [], 0.0
        for prompt in prompts:
            out, elapsed = timed_run(
                lambda p=prompt, g=gamma: speculative_generate(
                    s.target, s.draft, p, args.n_tokens, g, args.temperature, generator=gen
                ),
                s.device,
            )
            outputs.append(out)
            spec_time += elapsed
        alpha = acceptance_rate(outputs)
        row = {
            "gamma": gamma,
            "alpha": alpha,
            "speedup_measured": base_time / spec_time,
            "speedup_predicted": expected_speedup(alpha, gamma, c),
            "call_reduction_measured": call_reduction(outputs),
            "call_reduction_predicted": expected_tokens_per_iteration(alpha, gamma),
            "spec_time": spec_time,
        }
        rows.append(row)
        print(
            f"gamma={gamma} alpha={alpha:.3f} speedup={row['speedup_measured']:.2f}x "
            f"(pred {row['speedup_predicted']:.2f}x) "
            f"tok/call={row['call_reduction_measured']:.2f} "
            f"(pred {row['call_reduction_predicted']:.2f})"
        )

    mean_alpha = sum(r["alpha"] for r in rows) / len(rows)
    best_gamma, best_speedup = optimal_gamma(mean_alpha, c)
    print(f"Theoretical optimum for alpha={mean_alpha:.3f}, c={c:.3f}: gamma={best_gamma} ({best_speedup:.2f}x)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {"args": vars(args), "c": c, "baseline_time": base_time, "rows": rows,
         "optimal_gamma": best_gamma, "optimal_speedup": best_speedup},
        str(out_dir / "exp3_speedup.json"),
    )
    plot_speedup(rows, c, out_dir / "exp3_speedup.png")


if __name__ == "__main__":
    main()