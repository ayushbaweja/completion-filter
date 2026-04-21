from __future__ import annotations

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def mean_pairwise_cosine(embeddings: list[list[float]]) -> float:
    """Compute mean pairwise cosine similarity across all (N choose 2) pairs."""
    n = len(embeddings)
    if n < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += cosine_similarity(embeddings[i], embeddings[j])
            count += 1
    return total / count


def normalize_score(raw: float, low: float, high: float) -> float:
    """Clamp and linearly map a raw value into [0, 1]."""
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (raw - low) / (high - low)))
