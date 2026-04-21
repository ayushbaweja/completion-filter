from __future__ import annotations

import math

from openai import AsyncOpenAI

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.models import ConfidenceResult


class LogprobConfidence:
    """Confidence estimation via average token log-probabilities.

    Uses OpenAI's logprobs parameter to get per-token log-probabilities,
    then computes exp(mean_logprob) as a confidence score in (0, 1].
    """

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def estimate(
        self,
        prompt: str,
        model: str = "gpt-4o",
    ) -> tuple[ConfidenceResult, str]:
        """Estimate confidence and return (result, generated_text).

        Returns both so the draft response can be reused by other methods.
        """
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            logprobs=True,
            top_logprobs=1,
        )

        choice = response.choices[0]
        text = choice.message.content or ""

        tokens = choice.logprobs.content if choice.logprobs else []
        if not tokens:
            return ConfidenceResult(
                score=0.0, method_used="logprob", raw_logprob=None, sample_agreement=None
            ), text

        avg_logprob = sum(t.logprob for t in tokens) / len(tokens)
        # exp(avg_logprob) = geometric mean of token probabilities
        score = math.exp(avg_logprob)

        return ConfidenceResult(
            score=round(score, 4),
            method_used="logprob",
            raw_logprob=round(avg_logprob, 4),
            sample_agreement=None,
        ), text
