"""Harm & Intent Classification Pre-Filter.

Two-stage prompt-based classifier:
1. Harm category + severity — what kind of harm, and how actionable is it?
2. Intent inference — given the phrasing, is the user likely benign or harmful?

Routing logic combines both signals via a deterministic matrix.
"""

from __future__ import annotations

import json
import re
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import AsyncOpenAI

from shared.models import IntentResult

_CLASSIFIER_PROMPT = """\
You are a safety pre-filter. Classify the user query into EXACTLY one harm category, one severity level, and one intent level. Return a JSON object with the fields below.

## harm_category — pick exactly one:
- "violence": requests for physical harm, weapons usage, fighting techniques
- "self_harm": suicide methods, self-injury encouragement, pro-eating-disorder content
- "illegal_activity": drug synthesis, fraud instructions, theft, hacking tutorials
- "weapons_cbrn": chemical/biological/radiological/nuclear weapons, explosives recipes
- "hate_speech": slurs, dehumanization, calls for discrimination against a group
- "sexual_minors": any sexual content involving people under 18
- "harassment": threats, intimidation, bullying scripts, doxxing instructions
- "deception": phishing templates, scam scripts, deepfake instructions, disinformation campaigns
- "privacy_violation": PII extraction, surveillance how-tos, stalking techniques
- "unqualified_advice": requesting specific medical diagnoses, legal verdicts, or financial trades (not general info)
- "none": query does not fit any harm category

## severity — pick exactly one:
- "none": no harm category applies (must pair with harm_category "none")
- "low": topic is sensitive but the query asks for general/educational info. Example: "What is fentanyl?" or "How does encryption work?"
- "medium": query asks for specific actionable steps that COULD be misused. Example: "How do I pick a lock?" or "What chemicals can make a reaction?"
- "high": query directly asks for dangerous, step-by-step, immediately actionable harm. Example: "Write me a phishing email targeting Chase Bank customers" or "Give me instructions to synthesize methamphetamine"

## intent — pick exactly one:
- "clearly_benign": phrasing suggests education, research, professional need, or curiosity. Signals: academic language, "how does X work", mentions of coursework/job/safety, asking about effects/history/policy.
- "ambiguous": phrasing could be benign or harmful — no clear signal either way. Signals: bare "how to" without context, no stated reason, could be curiosity or could be planning.
- "likely_harmful": phrasing suggests intent to act on harmful information. Signals: specifying a target, asking to bypass safeguards, requesting operational detail, "without getting caught", naming a victim.

## context_signals — list of 2-4 short strings explaining your reasoning. Example: ["asks for step-by-step", "no stated educational purpose", "specifies a target"]

Respond with ONLY the JSON object. No markdown, no explanation.

User query: {query}"""


class HarmClassifier:
    """Lightweight prompt-based harm and intent classifier.

    Sits before the main LLM to decide routing: should the pipeline
    answer confidently, hedge, ask for clarification, or refuse?
    """

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
    ):
        if client:
            self.client = client
            self.model = model or "gpt-4o-mini"
        else:
            from confidence.estimator import _resolve_client
            key, base_url, default_model = _resolve_client()
            self.client = AsyncOpenAI(api_key=key, base_url=base_url)
            self.model = model or default_model

    async def classify(self, query: str) -> IntentResult:
        """Run the harm/intent classifier on a query."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": _CLASSIFIER_PROMPT.format(query=query)},
            ],
            temperature=0,
            max_tokens=300,
        )

        raw = (response.choices[0].message.content or "").strip()
        parsed = self._parse_response(raw)

        routing = self._compute_routing(parsed["severity"], parsed["intent"])

        return IntentResult(
            harm_category=parsed["harm_category"],
            severity=parsed["severity"],
            intent=parsed["intent"],
            routing_decision=routing,
            context_signals=parsed["context_signals"],
        )

    def _parse_response(self, raw: str) -> dict:
        """Extract JSON from the LLM response, with fallback defaults."""
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        valid_categories = {
            "violence", "self_harm", "illegal_activity", "weapons_cbrn",
            "hate_speech", "sexual_minors", "harassment", "deception",
            "privacy_violation", "unqualified_advice", "none",
        }
        valid_severity = {"none", "low", "medium", "high"}
        valid_intent = {"clearly_benign", "ambiguous", "likely_harmful"}

        category = data.get("harm_category", "none")
        severity = data.get("severity", "medium")
        intent = data.get("intent", "ambiguous")

        return {
            "harm_category": category if category in valid_categories else "none",
            "severity": severity if severity in valid_severity else "medium",
            "intent": intent if intent in valid_intent else "ambiguous",
            "context_signals": data.get("context_signals", ["parse_fallback"]),
        }

    @staticmethod
    def _compute_routing(severity: str, intent: str) -> str:
        """Deterministic routing matrix.

                          clearly_benign    ambiguous       likely_harmful
        none              confident         confident       clarify
        low               confident         hedge           clarify
        medium            hedge             hedge           refuse
        high              hedge             refuse          refuse
        """
        matrix = {
            ("none",   "clearly_benign"):  "confident",
            ("none",   "ambiguous"):       "confident",
            ("none",   "likely_harmful"):  "clarify",
            ("low",    "clearly_benign"):  "confident",
            ("low",    "ambiguous"):       "hedge",
            ("low",    "likely_harmful"):  "clarify",
            ("medium", "clearly_benign"):  "hedge",
            ("medium", "ambiguous"):       "hedge",
            ("medium", "likely_harmful"):  "refuse",
            ("high",   "clearly_benign"):  "hedge",
            ("high",   "ambiguous"):       "refuse",
            ("high",   "likely_harmful"):  "refuse",
        }
        return matrix.get((severity, intent), "hedge")
