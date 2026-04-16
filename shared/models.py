from dataclasses import dataclass
from typing import Optional

@dataclass
class IntentResult:
    """Output from Hugh's classifier"""
    population_benign_pct: float      # 0-100
    intent_category: str              # "informational" | "harmful" | "ambiguous"
    routing_decision: str             # "confident" | "hedge" | "clarify" | "refuse"
    context_signals: list[str]
    harm_score: float                 # 0-1

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