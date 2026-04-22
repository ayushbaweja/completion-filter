"""Harm & Intent Classification Pre-Filter."""


def __getattr__(name: str):
    if name == "HarmClassifier":
        from .classifier import HarmClassifier
        return HarmClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["HarmClassifier"]
