"""Evaluation utilities.

Pure-Python text metrics live in :mod:`brainpatch.evaluation.metrics` and can be
computed anywhere. Model-dependent measurements (log-probabilities, capability
probes) require torch and live in :mod:`brainpatch.research.ml.evaluation`.
"""

from brainpatch.evaluation.metrics import (
    GenerationMetrics,
    compare_generations,
    distinct_n,
    jaccard_similarity,
    longest_repeated_ngram,
    most_common_ngram_fraction,
    repetition_rate,
    score_generation,
)

__all__ = [
    "GenerationMetrics",
    "compare_generations",
    "distinct_n",
    "jaccard_similarity",
    "longest_repeated_ngram",
    "most_common_ngram_fraction",
    "repetition_rate",
    "score_generation",
]
