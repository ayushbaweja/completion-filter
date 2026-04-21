from __future__ import annotations

import math
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openai import AsyncOpenAI

from shared.models import ConfidenceResult

_FALLBACK_PROMPT = (
    "Rate your confidence in the following answer on a scale of 0.0 to 1.0 "
    "based on how certain each part of the answer is. "
    "Reply with ONLY a decimal number. Nothing else.\n\n"
    "Question: {prompt}\nAnswer: {response}"
)


class LogprobConfidence:
    """Confidence estimation via average token log-probabilities.

    Requests logprobs from the API. If the model doesn't support them,
    falls back to a numeric self-assessment prompt.
    """

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def estimate(
        self,
        prompt: str,
        model: str = "minimax/minimax-m2.5:free",
    ) -> tuple[ConfidenceResult, str]:
        """Estimate confidence and return (result, generated_text)."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                logprobs=True,
                top_logprobs=1,
            )
        except Exception:
            # Model doesn't accept logprobs param at all
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

        choice = response.choices[0]
        text = choice.message.content or ""

        # Try to extract logprobs
        tokens = choice.logprobs.content if choice.logprobs else []
        if tokens:
            avg_logprob = sum(t.logprob for t in tokens) / len(tokens)
            score = math.exp(avg_logprob)
            return ConfidenceResult(
                score=round(score, 4),
                method_used="logprob",
                raw_logprob=round(avg_logprob, 4),
                sample_agreement=None,
            ), text

        # Fallback: ask model to self-rate numerically
        fallback = await self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": _FALLBACK_PROMPT.format(prompt=prompt, response=text),
                }
            ],
            temperature=0,
            max_tokens=5,
        )
        raw = (fallback.choices[0].message.content or "").strip()
        match = re.search(r"(0?\.\d+|1\.0|[01])", raw)
        if match:
            score = min(1.0, max(0.0, float(match.group(1))))
        else:
            score = 0.5

        return ConfidenceResult(
            score=round(score, 4),
            method_used="logprob_fallback",
            raw_logprob=None,
            sample_agreement=None,
        ), text
