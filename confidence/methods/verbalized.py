from __future__ import annotations

import re

from openai import AsyncOpenAI

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.models import ConfidenceResult

_SYSTEM_PROMPT = (
    "You are a calibration assistant. Given a question and a draft answer, "
    "estimate how confident you are that the answer is correct. "
    "Reply with ONLY a number between 0 and 100 representing your confidence "
    "percentage. Nothing else."
)


class VerbalizedConfidence:
    """Confidence estimation by asking the model to self-assess."""

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def estimate(
        self,
        prompt: str,
        draft_response: str,
        model: str = "gpt-4o",
    ) -> ConfidenceResult:
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {prompt}\n\n"
                        f"Draft Answer: {draft_response}\n\n"
                        "Confidence (0-100):"
                    ),
                },
            ],
            temperature=0,
            max_tokens=5,
        )

        raw = (response.choices[0].message.content or "").strip()
        match = re.search(r"(\d+)", raw)
        if match:
            pct = min(100, max(0, int(match.group(1))))
            score = pct / 100.0
        else:
            score = 0.5  # fallback if parsing fails

        return ConfidenceResult(
            score=round(score, 4),
            method_used="verbalized",
            raw_logprob=None,
            sample_agreement=None,
        )
