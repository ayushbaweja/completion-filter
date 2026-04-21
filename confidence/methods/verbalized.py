from __future__ import annotations

import asyncio
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from google import genai
from google.genai import types

from shared.models import ConfidenceResult

_SYSTEM_PROMPT = (
    "You are a calibration assistant. Given a question and a draft answer, "
    "estimate how confident you are that the answer is correct. "
    "Reply with ONLY a number between 0 and 100 representing your confidence "
    "percentage. Nothing else."
)


class VerbalizedConfidence:
    """Confidence estimation by asking the model to self-assess."""

    def __init__(self, client: genai.Client):
        self.client = client

    async def estimate(
        self,
        prompt: str,
        draft_response: str,
        model: str = "gemma-4-31b-it",
    ) -> ConfidenceResult:
        user_msg = (
            f"Question: {prompt}\n\n"
            f"Draft Answer: {draft_response}\n\n"
            "Confidence (0-100):"
        )

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=model,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0,
                max_output_tokens=5,
            ),
        )

        raw = (response.text or "").strip()
        match = re.search(r"(\d+)", raw)
        if match:
            pct = min(100, max(0, int(match.group(1))))
            score = pct / 100.0
        else:
            score = 0.5

        return ConfidenceResult(
            score=round(score, 4),
            method_used="verbalized",
            raw_logprob=None,
            sample_agreement=None,
        )
