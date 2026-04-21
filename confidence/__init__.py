"""Confidence Estimation Pipeline.

Usage:
    from confidence import ConfidenceEstimator
"""


def __getattr__(name: str):
    if name == "ConfidenceEstimator":
        from .estimator import ConfidenceEstimator
        return ConfidenceEstimator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ConfidenceEstimator"]
