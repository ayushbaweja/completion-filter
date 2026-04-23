from __future__ import annotations

import asyncio
import math
import re
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from openai import AsyncOpenAI

from confidence.utils import mean_pairwise_cosine
from shared.models import ConfidenceResult


def _tfidf_embed(texts: list[str]) -> list[list[float]]:
    """Compute simple TF-IDF embeddings locally (no API needed)."""
    tokenized = [re.findall(r"\w+", t.lower()) for t in texts]

    # Build vocabulary
    vocab: dict[str, int] = {}
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        unique = set(tokens)
        for tok in unique:
            doc_freq[tok] += 1
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    n_docs = len(texts)
    embeddings: list[list[float]] = []
    for tokens in tokenized:
        tf = Counter(tokens)
        vec = [0.0] * len(vocab)
        for tok, count in tf.items():
            idf = math.log((n_docs + 1) / (doc_freq[tok] + 1)) + 1
            vec[vocab[tok]] = count * idf
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        embeddings.append(vec)

    return embeddings


class SemanticEntropyConfidence:
    """Confidence estimation via multi-sample semantic entropy.

    Generates N responses to the same prompt with temperature > 0,
    embeds them locally via TF-IDF, and measures agreement via
    mean pairwise cosine similarity.
    High similarity = consistent answers = high confidence.
    """

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def _generate_one(
        self, prompt: str, model: str, temperature: float
    ) -> str:
        for attempt in range(4):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1024,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    wait = 20 * (attempt + 1)
                    print(f"  [Semantic entropy rate limited, waiting {wait}s...]")
                    await asyncio.sleep(wait)
                else:
                    raise

    async def estimate(
        self,
        prompt: str,
        model: str = "minimax/minimax-m2.5:free",
        n_samples: int = 5,
        temperature: float = 0.8,
    ) -> ConfidenceResult:
        # Generate N responses concurrently
        tasks = [
            self._generate_one(prompt, model, temperature)
            for _ in range(n_samples)
        ]
        responses = await asyncio.gather(*tasks)

        # Embed locally via TF-IDF
        embeddings = _tfidf_embed(list(responses))

        # Compute mean pairwise cosine similarity
        agreement = mean_pairwise_cosine(embeddings)

        return ConfidenceResult(
            score=round(agreement, 4),
            method_used="semantic_entropy",
            raw_logprob=None,
            sample_agreement=round(agreement, 4),
        )
