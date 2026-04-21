"""Demo script for the Confidence Estimation Pipeline.

Requires OPENAI_API_KEY environment variable to be set.

Usage:
    python demo.py
    python demo.py --method logprob "What is the capital of France?"
    python demo.py --all "What will Bitcoin be worth next year?"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from confidence import ConfidenceEstimator


def print_confidence(label: str, result) -> None:
    status = "UNCERTAIN" if result.score < 0.4 else "CONFIDENT" if result.score > 0.7 else "MODERATE"
    print(f"  [{label}] score={result.score:.4f}  method={result.method_used}  -> {status}")
    if result.raw_logprob is not None:
        print(f"    raw_logprob={result.raw_logprob:.4f}")
    if result.sample_agreement is not None:
        print(f"    sample_agreement={result.sample_agreement:.4f}")


async def run_single(estimator: ConfidenceEstimator, prompt: str, method: str) -> None:
    print(f"\nPrompt: {prompt}")
    print(f"Method: {method}\n")
    result = await estimator.estimate(prompt, method=method)
    print_confidence(method, result)


async def run_all(estimator: ConfidenceEstimator, prompt: str) -> None:
    print(f"\nPrompt: {prompt}")
    print("Running all methods...\n")
    results, draft = await estimator.estimate_all(prompt)
    for name, result in results.items():
        print_confidence(name, result)

    aggregated = estimator.aggregate(results)
    print()
    print_confidence("aggregated", aggregated)
    print(f"\n  Draft response (first 200 chars): {draft[:200]}")
    print(f"  Uncertain: {estimator.is_uncertain(aggregated)}")


async def demo() -> None:
    estimator = ConfidenceEstimator()

    test_prompts = [
        ("What is the capital of France?", "High confidence factual"),
        ("What will the stock market do next Tuesday?", "Low confidence prediction"),
    ]

    for prompt, description in test_prompts:
        print(f"\n{'=' * 70}")
        print(f"  {description}")
        print(f"{'=' * 70}")
        await run_all(estimator, prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence Estimation Pipeline Demo")
    parser.add_argument("prompt", nargs="?", help="Custom prompt to evaluate")
    parser.add_argument("--method", choices=["logprob", "semantic_entropy", "verbalized"], help="Run a single method")
    parser.add_argument("--all", action="store_true", help="Run all methods")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--threshold", type=float, default=0.4, help="Uncertainty threshold")
    parser.add_argument("--aggregation", default="min", choices=["min", "mean", "weighted"])
    args = parser.parse_args()

    estimator = ConfidenceEstimator(
        model=args.model,
        uncertainty_threshold=args.threshold,
        aggregation=args.aggregation,
    )

    if args.prompt:
        if args.method:
            asyncio.run(run_single(estimator, args.prompt, args.method))
        elif args.all:
            asyncio.run(run_all(estimator, args.prompt))
        else:
            asyncio.run(run_all(estimator, args.prompt))
    else:
        asyncio.run(demo())


if __name__ == "__main__":
    main()
