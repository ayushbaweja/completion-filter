from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

from confidence.methods.semantic_entropy import SemanticEntropyConfidence
from confidence.methods.verbalized import VerbalizedConfidence
from shared.models import ConfidenceResult

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_DEFAULT_MODEL = "minimax/minimax-m2.5:free"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


def _resolve_client(api_key: str | None = None) -> tuple[str, str, str]:
    """Return (api_key, base_url, default_model) based on which env var is set."""
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if key:
        return key, OPENROUTER_BASE_URL, OPENROUTER_DEFAULT_MODEL
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key, OPENAI_BASE_URL, OPENAI_DEFAULT_MODEL
    raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env")


class ConfidenceEstimator:
    """Orchestrator that runs confidence estimation methods.

    Methods:
        estimate     -- run a single confidence method
        estimate_all -- run both methods concurrently
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        uncertainty_threshold: float = 0.4,
        aggregation: str = "min",  # "min" | "mean"
    ):
        resolved_key, base_url, default_model = _resolve_client(api_key)
        self.client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=base_url,
        )
        self.model = model or default_model
        self.uncertainty_threshold = uncertainty_threshold
        self.aggregation = aggregation

        self._semantic = SemanticEntropyConfidence(self.client)
        self._verbalized = VerbalizedConfidence(self.client)

    async def _generate_draft(self, prompt: str) -> str:
        """Generate a draft response to use for verbalized confidence."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""

    async def estimate(
        self, prompt: str, method: str = "semantic_entropy"
    ) -> ConfidenceResult:
        """Run a single confidence estimation method."""
        if method == "semantic_entropy":
            return await self._semantic.estimate(prompt, model=self.model)
        elif method == "verbalized":
            draft = await self._generate_draft(prompt)
            return await self._verbalized.estimate(
                prompt, draft, model=self.model
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    async def estimate_all(
        self, prompt: str
    ) -> tuple[dict[str, ConfidenceResult], str]:
        """Run both methods concurrently. Returns (results_dict, draft_response)."""
        draft = await self._generate_draft(prompt)

        semantic_result, verbalized_result = await asyncio.gather(
            self._semantic.estimate(prompt, model=self.model),
            self._verbalized.estimate(prompt, draft, model=self.model),
        )

        results = {
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
        else:
            final_score = min(scores)

        best_method = min(results, key=lambda k: abs(results[k].score - final_score))
        best = results[best_method]
        return ConfidenceResult(
            score=round(final_score, 4),
            method_used=f"aggregated_{self.aggregation}",
            raw_logprob=None,
            sample_agreement=results.get("semantic_entropy", best).sample_agreement,
        )

    def is_uncertain(self, result: ConfidenceResult) -> bool:
        """Check if a confidence score falls below the uncertainty threshold."""
        return result.score < self.uncertainty_threshold
