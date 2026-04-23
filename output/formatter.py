import json
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

from shared.models import OutputRequest, FinalOutput

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

HEDGE_TEMPLATES = {
    "high_uncertainty": "I'm not very confident in this — you should verify it: ",
    "medium_uncertainty": "I'm fairly confident, though there may be nuances I'm missing: ",
    "domain": {
        "medical": "I'm not a medical professional — please consult a doctor. That said: ",
        "legal": "This isn't legal advice, but generally speaking: ",
        "financial": "I'm not a financial advisor, but: ",
    }
}

MEDICAL_KEYWORDS = [
    "medication", "diagnosis", "symptoms", "doctor", "headache",
    "treatment", "pain", "health", "disease", "illness", "injury",
    "medicine", "drug", "dose", "prescription", "hospital", "nurse",
]

LEGAL_KEYWORDS = [
    "legal", "lawsuit", "contract", "lawyer", "attorney", "court",
    "sue", "liability", "rights", "law", "illegal", "crime",
]


def _get_client() -> tuple[AsyncOpenAI, str]:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env")
    return AsyncOpenAI(api_key=key, base_url=OPENROUTER_BASE_URL), OPENROUTER_DEFAULT_MODEL


async def _call_with_retry(
    client: AsyncOpenAI,
    model: str,
    messages: list,
    temperature: float,
    max_retries: int = 3,
):
    """Call the API with automatic retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 20 * (attempt + 1)
                print(f"  [Rate limited, waiting {wait}s...]")
                await asyncio.sleep(wait)
            else:
                raise


async def format_response(request: OutputRequest) -> FinalOutput:
    """Main entry point. Takes pipeline state, returns final formatted response."""
    client, model = _get_client()
    mode = _determine_display_mode(request)

    if mode == "refusal":
        return await _handle_refusal(request, client, model)

    response = _apply_hedging_to_text(request, mode, request.draft_response)
    helpfulness = await _evaluate_helpfulness(request, response, client, model)

    if helpfulness < 3.5:
        raw_regen = await _regenerate_response(request, mode, client, model)
        response = _apply_hedging_to_text(request, mode, raw_regen)
        helpfulness = await _evaluate_helpfulness(request, response, client, model)
        was_regenerated = True
    else:
        was_regenerated = False

    return FinalOutput(
        response_text=response,
        display_mode=mode,
        helpfulness_score=helpfulness,
        hedge_language=_extract_hedge_language(response) if mode == "hedged" else None,
        refusal_reason=None,
        was_regenerated=was_regenerated,
    )


def _determine_display_mode(request: OutputRequest) -> str:
    """
    Intent routing_decision takes priority.
    Confidence score can escalate hedge → clarify but cannot override a refusal.
    """
    routing = request.intent.routing_decision
    confidence = request.confidence.score

    if routing == "refuse":
        return "refusal"

    if routing == "clarify":
        return "clarify"

    # routing is "confident" or "hedge" — let confidence score fine-tune
    if confidence >= 0.75:
        return "confident"
    elif confidence >= 0.4:
        return "hedged"
    else:
        return "clarify"


def _apply_hedging_to_text(request: OutputRequest, mode: str, text: str) -> str:
    """Apply the appropriate hedge prefix to any text string."""
    if mode == "confident":
        return text

    confidence = request.confidence.score
    prompt_lower = request.original_prompt.lower()
    harm_category = request.intent.harm_category

    # Domain-specific hedges based on harm_category Hugh already classified
    if harm_category == "unqualified_advice":
        if any(w in prompt_lower for w in MEDICAL_KEYWORDS):
            hedge = HEDGE_TEMPLATES["domain"]["medical"]
        elif any(w in prompt_lower for w in LEGAL_KEYWORDS):
            hedge = HEDGE_TEMPLATES["domain"]["legal"]
        else:
            hedge = HEDGE_TEMPLATES["domain"]["financial"]
    elif confidence < 0.4:
        hedge = HEDGE_TEMPLATES["high_uncertainty"]
    else:
        hedge = HEDGE_TEMPLATES["medium_uncertainty"]

    return hedge + text


def _extract_hedge_language(response: str) -> str:
    """Return the hedge prefix if present at the start of the response."""
    candidates = [
        HEDGE_TEMPLATES["high_uncertainty"],
        HEDGE_TEMPLATES["medium_uncertainty"],
        HEDGE_TEMPLATES["domain"]["medical"],
        HEDGE_TEMPLATES["domain"]["legal"],
        HEDGE_TEMPLATES["domain"]["financial"],
    ]
    for hedge in candidates:
        if response.startswith(hedge):
            return hedge
    return ""


async def _handle_refusal(
    request: OutputRequest, client: AsyncOpenAI, model: str
) -> FinalOutput:
    """Generate a non-judgmental refusal with a safe redirect."""
    refusal_prompt = f"""The user asked: "{request.original_prompt}"
This was flagged as: {request.intent.harm_category} (severity={request.intent.severity}, intent={request.intent.intent})

Write a short refusal (2-3 sentences) that:
1. Declines clearly without being preachy or judgmental
2. Briefly explains why (reference the category if helpful)
3. Offers a safe alternative if one exists

Reply with only the refusal text, no preamble."""

    response = await _call_with_retry(
        client, model,
        [{"role": "user", "content": refusal_prompt}],
        temperature=0.3,
    )
    refusal_text = response.choices[0].message.content or "I'm not able to help with that."

    return FinalOutput(
        response_text=refusal_text,
        display_mode="refusal",
        helpfulness_score=0.0,
        hedge_language=None,
        refusal_reason=f"{request.intent.harm_category} / {request.intent.severity}",
        was_regenerated=False,
    )


async def _regenerate_response(
    request: OutputRequest, mode: str, client: AsyncOpenAI, model: str
) -> str:
    """Ask the model to produce a better response (without hedge prefix — added separately)."""
    regen_prompt = f"""The user asked: "{request.original_prompt}"
A previous response was not helpful enough.
Write a clearer, more useful response.
{"Express appropriate uncertainty since confidence is low." if mode == "hedged" else ""}
Reply with only the response text, no preamble."""

    response = await _call_with_retry(
        client, model,
        [{"role": "user", "content": regen_prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content or request.draft_response


async def _evaluate_helpfulness(
    request: OutputRequest, response: str, client: AsyncOpenAI, model: str
) -> float:
    """Score the response on four dimensions and return a weighted average (1-5)."""
    eval_prompt = f"""Rate this AI response. Reply ONLY with valid JSON, nothing else.

User message: "{request.original_prompt}"
Harm category: "{request.intent.harm_category}"
Severity: "{request.intent.severity}"
AI response: "{response}"

{{
  "task_completion": 0,
  "appropriate_detail": 0,
  "hedge_calibration": 0,
  "actionability": 0,
  "reasoning": ""
}}

Replace each 0 with a score from 1 to 5. Fill in reasoning with one sentence."""

    api_response = await _call_with_retry(
        client, model,
        [{"role": "user", "content": eval_prompt}],
        temperature=0,
    )
    raw = (api_response.choices[0].message.content or "").strip()

    # Strip markdown code fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        scores = json.loads(raw)
        return (
            scores["task_completion"] * 0.35 +
            scores["appropriate_detail"] * 0.25 +
            scores["hedge_calibration"] * 0.25 +
            scores["actionability"] * 0.15
        )
    except (json.JSONDecodeError, KeyError):
        return 3.0