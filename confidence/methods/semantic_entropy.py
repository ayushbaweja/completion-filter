from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from confidence.utils import mean_pairwise_cosine
from shared.models import ConfidenceResult


class SemanticEntropyConfidence:
    """Confidence estimation via multi-sample semantic entropy.

    Generates N responses to the same prompt with temperature > 0,
    embeds them, and measures agreement via mean pairwise cosine similarity.
    High similarity = consistent answers = high confidence.
    """

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def _generate_one(
        self, prompt: str, model: str, temperature: float
    ) -> str:
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def estimate(
        self,
        prompt: str,
        model: str = "gpt-4o",
        n_samples: int = 5,
        temperature: float = 0.8,
        embedding_model: str = "text-embedding-3-small",
    ) -> ConfidenceResult:
        # Generate N responses concurrently
        tasks = [
            self._generate_one(prompt, model, temperature)
            for _ in range(n_samples)
        ]
        responses = await asyncio.gather(*tasks)

        # Embed all responses in a single batch call
        embed_response = await self.client.embeddings.create(
            model=embedding_model,
            input=list(responses),
        )
        embeddings = [item.embedding for item in embed_response.data]

        # Compute mean pairwise cosine similarity
        agreement = mean_pairwise_cosine(embeddings)

        return ConfidenceResult(
            score=round(agreement, 4),
            method_used="semantic_entropy",
            raw_logprob=None,
            sample_agreement=round(agreement, 4),
        )
