import asyncio
from shared.models import OutputRequest, IntentResult, ConfidenceResult
from output.formatter import format_response

mock_request = OutputRequest(
    original_prompt="What's the best treatment for a headache?",
    intent=IntentResult(
        harm_category="unqualified_advice",
        severity="low",
        intent="clearly_benign",
        routing_decision="hedge",
        context_signals=["medical question", "asking for treatment"]
    ),
    confidence=ConfidenceResult(
        score=0.45,
        method_used="semantic_entropy",
        raw_logprob=None,
        sample_agreement=0.45
    ),
    draft_response="You could try ibuprofen or acetaminophen for a headache."
)

result = asyncio.run(format_response(mock_request))
print("Response:", result.response_text)
print("Mode:", result.display_mode)
print("Helpfulness:", result.helpfulness_score)
print("Hedge language:", result.hedge_language)
print("Regenerated:", result.was_regenerated)