from __future__ import annotations

import json

from openai import AsyncOpenAI

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.models import IntentResult

_SYSTEM_PROMPT = """\
You are a content safety classifier. Analyze the user's query and output JSON with these fields:
- "population_benign_pct": 0-100, what percentage of people asking this would have benign intent
- "intent_category": one of "informational", "harmful", or "ambiguous"
- "harm_score": float 0-1 (0 = completely safe, 1 = clearly harmful)
- "routing_decision": one of "confident" (safe to answer normally), "hedge" (answer with caveats), "clarify" (ask follow-up), "refuse" (decline)
- "context_signals": list of short strings explaining your reasoning

Be calibrated: most questions are benign. Only flag truly harmful requests.\
"""


class HarmFilter:
    """Prompt-based harm classifier that routes between ethical refusal
    and epistemic uncertainty."""

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def classify(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
    ) -> IntentResult:
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}

        return IntentResult(
            population_benign_pct=float(data.get("population_benign_pct", 50.0)),
            intent_category=data.get("intent_category", "ambiguous"),
            routing_decision=data.get("routing_decision", "clarify"),
            context_signals=data.get("context_signals", []),
            harm_score=float(data.get("harm_score", 0.5)),
        )
