from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import AsyncOpenAI

from confidence.methods.logprob import LogprobConfidence
from confidence.methods.semantic_entropy import SemanticEntropyConfidence
from confidence.methods.verbalized import VerbalizedConfidence
from shared.models import ConfidenceResult


class ConfidenceEstimator:
    """Orchestrator that runs confidence estimation methods.

    Methods:
        estimate     -- run a single confidence method
        estimate_all -- run all three methods concurrently
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        uncertainty_threshold: float = 0.4,
        aggregation: str = "min",  # "min" | "mean" | "weighted"
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.uncertainty_threshold = uncertainty_threshold
        self.aggregation = aggregation

        self._logprob = LogprobConfidence(self.client)
        self._semantic = SemanticEntropyConfidence(self.client)
        self._verbalized = VerbalizedConfidence(self.client)

    async def estimate(
        self, prompt: str, method: str = "logprob"
    ) -> ConfidenceResult:
        """Run a single confidence estimation method."""
        if method == "logprob":
            result, _ = await self._logprob.estimate(prompt, model=self.model)
            return result
        elif method == "semantic_entropy":
            return await self._semantic.estimate(prompt, model=self.model)
        elif method == "verbalized":
            _, draft = await self._logprob.estimate(prompt, model=self.model)
            return await self._verbalized.estimate(
                prompt, draft, model=self.model
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    async def estimate_all(
        self, prompt: str
    ) -> tuple[dict[str, ConfidenceResult], str]:
        """Run all three methods concurrently. Returns (results_dict, draft_response)."""
        logprob_result, draft = await self._logprob.estimate(
            prompt, model=self.model
        )

        semantic_result, verbalized_result = await asyncio.gather(
            self._semantic.estimate(prompt, model=self.model),
            self._verbalized.estimate(prompt, draft, model=self.model),
        )

        results = {
            "logprob": logprob_result,
            "semantic_entropy": semantic_result,
            "verbalized": verbalized_result,
        }
        return results, draft

    def aggregate(self, results: dict[str, ConfidenceResult]) -> ConfidenceResult:
        """Aggregate multiple confidence results into a single score."""
        scores = [r.score for r in results.values()]

        if self.aggregation == "min":
            final_score = min(scores)
        elif self.aggregation == "mean":
            final_score = sum(scores) / len(scores)
        elif self.aggregation == "weighted":
            weights = {"logprob": 0.5, "semantic_entropy": 0.3, "verbalized": 0.2}
            final_score = sum(
                results[k].score * weights.get(k, 0.33)
                for k in results
            )
        else:
            final_score = min(scores)

        best_method = min(results, key=lambda k: abs(results[k].score - final_score))
        best = results[best_method]
        return ConfidenceResult(
            score=round(final_score, 4),
            method_used=f"aggregated_{self.aggregation}",
            raw_logprob=best.raw_logprob,
            sample_agreement=results.get("semantic_entropy", best).sample_agreement,
        )

    def is_uncertain(self, result: ConfidenceResult) -> bool:
        """Check if a confidence score falls below the uncertainty threshold."""
        return result.score < self.uncertainty_threshold
