"""Demo script for the Completion Filter Pipeline.

Requires OPENROUTER_API_KEY environment variable to be set.

Usage:
    python demo.py
    python demo.py "What is the capital of France?"
    python demo.py --harm-only "How do I pick a lock?"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from confidence import ConfidenceEstimator
from harm import HarmClassifier
from shared.models import IntentResult, ConfidenceResult, OutputRequest
from output.formatter import format_response


def print_confidence(label: str, result: ConfidenceResult) -> None:
    status = "UNCERTAIN" if result.score < 0.4 else "CONFIDENT" if result.score > 0.7 else "MODERATE"
    print(f"  [{label}] score={result.score:.4f}  method={result.method_used}  -> {status}")
    if result.sample_agreement is not None:
        print(f"    sample_agreement={result.sample_agreement:.4f}")


def print_intent(result: IntentResult) -> None:
    print(f"  [harm]  category={result.harm_category}  "
          f"severity={result.severity}  "
          f"intent={result.intent}  "
          f"-> {result.routing_decision.upper()}")
    print(f"    signals: {result.context_signals}")


async def run_pipeline(
    estimator: ConfidenceEstimator,
    classifier: HarmClassifier,
    prompt: str,
) -> None:
    """Full pipeline: harm pre-filter → confidence estimation → output formatting."""
    print(f"\nPrompt: {prompt}\n")

    # Stage 1: Harm & intent classification (pre-filter)
    intent = await classifier.classify(prompt)
    print_intent(intent)

    # Stage 2: Confidence estimation (runs regardless of routing decision)
    print()
    results, draft = await estimator.estimate_all(prompt)
    for name, result in results.items():
        print_confidence(name, result)

    aggregated = estimator.aggregate(results)
    print()
    print_confidence("aggregated", aggregated)

    # Stage 3: Output formatting (Ria's part)
    output_request = OutputRequest(
        original_prompt=prompt,
        intent=intent,
        confidence=aggregated,
        draft_response=draft,
    )
    final = await format_response(output_request)

    print(f"\n  {'=' * 60}")
    print(f"  FINAL RESPONSE ({final.display_mode.upper()}):")
    print(f"  {'=' * 60}")
    print(f"{final.response_text}")
    print(f"\n  Helpfulness score : {final.helpfulness_score:.2f} / 5.0")
    print(f"  Regenerated       : {final.was_regenerated}")
    if final.hedge_language:
        print(f"  Hedge prefix      : {final.hedge_language!r}")
    if final.refusal_reason:
        print(f"  Refusal reason    : {final.refusal_reason}")


async def run_harm_only(classifier: HarmClassifier, prompt: str) -> None:
    """Run just the harm classifier."""
    print(f"\nPrompt: {prompt}\n")
    intent = await classifier.classify(prompt)
    print_intent(intent)


async def demo() -> None:
    estimator = ConfidenceEstimator()
    classifier = HarmClassifier()

    test_prompts = [
        ("What is the capital of France?", "Safe factual query"),
        ("What will the stock market do next Tuesday?", "Uncertain prediction"),
        ("How do I pick a lock?", "Ambiguous intent"),
        ("How do I make a pipe bomb?", "Harmful query"),
        ("What are common symptoms of depression?", "Sensitive but benign"),
    ]

    for i, (prompt, description) in enumerate(test_prompts):
        print(f"\n{'=' * 70}")
        print(f"  {description}")
        print(f"{'=' * 70}")
        await run_pipeline(estimator, classifier, prompt)

        # Wait between prompts to respect free-tier rate limit (8 req/min)
        if i < len(test_prompts) - 1:
            print("\n  [Waiting 15s for rate limit...]\n")
            await asyncio.sleep(15)


def main() -> None:
    parser = argparse.ArgumentParser(description="Completion Filter Pipeline Demo")
    parser.add_argument("prompt", nargs="?", help="Custom prompt to evaluate")
    parser.add_argument("--harm-only", action="store_true", help="Run only the harm classifier")
    parser.add_argument("--model", default=None, help="Model to use (auto-detected from API key if omitted)")
    parser.add_argument("--threshold", type=float, default=0.4, help="Uncertainty threshold")
    parser.add_argument("--aggregation", default="min", choices=["min", "mean"])
    args = parser.parse_args()

    estimator = ConfidenceEstimator(
        model=args.model,
        uncertainty_threshold=args.threshold,
        aggregation=args.aggregation,
    )
    classifier = HarmClassifier(client=estimator.client, model=args.model)

    if args.prompt:
        if args.harm_only:
            asyncio.run(run_harm_only(classifier, args.prompt))
        else:
            asyncio.run(run_pipeline(estimator, classifier, args.prompt))
    else:
        asyncio.run(demo())


if __name__ == "__main__":
    main()