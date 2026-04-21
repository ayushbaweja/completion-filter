from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from google import genai
from google.genai import types

from confidence.utils import mean_pairwise_cosine
from shared.models import ConfidenceResult


class SemanticEntropyConfidence:
    """Confidence estimation via multi-sample semantic entropy.

    Generates N responses to the same prompt with temperature > 0,
    embeds them, and measures agreement via mean pairwise cosine similarity.
    High similarity = consistent answers = high confidence.
    """

    def __init__(self, client: genai.Client):
        self.client = client

    def _generate_one(
        self, prompt: str, model: str, temperature: float
    ) -> str:
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return response.text or ""

    async def estimate(
        self,
        prompt: str,
        model: str = "gemma-4-31b-it",
        n_samples: int = 5,
        temperature: float = 0.8,
        embedding_model: str = "text-embedding-004",
    ) -> ConfidenceResult:
        # Generate N responses concurrently
        tasks = [
            asyncio.to_thread(self._generate_one, prompt, model, temperature)
            for _ in range(n_samples)
        ]
        responses = await asyncio.gather(*tasks)

        # Embed all responses
        embed_tasks = [
            asyncio.to_thread(
                self.client.models.embed_content,
                model=embedding_model,
                contents=resp,
            )
            for resp in responses
        ]
        embed_results = await asyncio.gather(*embed_tasks)
        embeddings = [r.embeddings[0].values for r in embed_results]

        # Compute mean pairwise cosine similarity
        agreement = mean_pairwise_cosine(embeddings)

        return ConfidenceResult(
            score=round(agreement, 4),
            method_used="semantic_entropy",
            raw_logprob=None,
            sample_agreement=round(agreement, 4),
        )
