from __future__ import annotations

import asyncio
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from google import genai
from google.genai import types

from shared.models import ConfidenceResult


class LogprobConfidence:
    """Confidence estimation via average token log-probabilities.

    Uses Gemini's response_logprobs to get per-token log-probabilities,
    then computes exp(mean_logprob) as a confidence score in (0, 1].
    """

    def __init__(self, client: genai.Client):
        self.client = client

    async def estimate(
        self,
        prompt: str,
        model: str = "gemma-4-31b-it",
    ) -> tuple[ConfidenceResult, str]:
        """Estimate confidence and return (result, generated_text)."""
        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_logprobs=True,
                logprobs=1,
            ),
        )

        text = response.text or ""

        logprobs_result = response.candidates[0].logprobs_result if response.candidates else None
        if not logprobs_result or not logprobs_result.chosen_candidates:
            return ConfidenceResult(
                score=0.0, method_used="logprob", raw_logprob=None, sample_agreement=None
            ), text

        token_logprobs = [t.log_probability for t in logprobs_result.chosen_candidates]
        if not token_logprobs:
            return ConfidenceResult(
                score=0.0, method_used="logprob", raw_logprob=None, sample_agreement=None
            ), text

        avg_logprob = sum(token_logprobs) / len(token_logprobs)
        score = math.exp(avg_logprob)

        return ConfidenceResult(
            score=round(score, 4),
            method_used="logprob",
            raw_logprob=round(avg_logprob, 4),
            sample_agreement=None,
        ), text
