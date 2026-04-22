from dataclasses import dataclass, field
from typing import Optional

# Harm categories drawn from OpenAI Moderation + HarmBench taxonomies
HARM_CATEGORIES = [
    "violence",             # physical harm, weapons, fighting techniques
    "self_harm",            # suicide, self-injury, eating disorders
    "illegal_activity",     # drugs, fraud, theft, hacking instructions
    "weapons_cbrn",         # chemical, biological, radiological, nuclear, explosives
    "hate_speech",          # slurs, dehumanization, targeted group attacks
    "sexual_minors",        # any sexual content involving minors
    "harassment",           # threats, intimidation, bullying, doxxing
    "deception",            # scams, phishing, impersonation, misinformation
    "privacy_violation",    # PII extraction, surveillance, stalking
    "unqualified_advice",   # medical/legal/financial advice without credentials
    "none",                 # no harm category applies
]

@dataclass
class IntentResult:
    """Output from Hugh's classifier"""
    harm_category: str                # one of HARM_CATEGORIES
    severity: str                     # "none" | "low" | "medium" | "high"
    intent: str                       # "clearly_benign" | "ambiguous" | "likely_harmful"
    routing_decision: str             # "confident" | "hedge" | "clarify" | "refuse"
    context_signals: list[str] = field(default_factory=list)

@dataclass  
class ConfidenceResult:
    """Output from Ayush's estimator"""
    score: float                      # 0-1
    method_used: str                  # "logprob" | "semantic_entropy" | "verbalized"
    raw_logprob: Optional[float]
    sample_agreement: Optional[float]

@dataclass
class OutputRequest:
    """What Ria's part receives"""
    original_prompt: str
    intent: IntentResult
    confidence: ConfidenceResult
    draft_response: str               # Raw LLM response before formatting

@dataclass
class FinalOutput:
    """What Ria's part produces"""
    response_text: str
    display_mode: str                 # "confident" | "hedged" | "refusal"
    helpfulness_score: float          # 0-5
    hedge_language: Optional[str]
    refusal_reason: Optional[str]
    was_regenerated: bool