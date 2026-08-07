"""Model-free text metrics for detecting intervention side effects.

Steering a residual stream can wreck a model's fluency long before it changes
the behaviour you were aiming at. These metrics are the cheap tripwires: they
catch degeneration (loops, single-token spam, truncation) without needing a
judge model or a paid API.

They are *not* measures of quality. A high `distinct_2` does not mean the answer
is good; a low one strongly suggests the answer is broken. Use them to reject
interventions, not to award them.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

_WORD_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")


def tokenize_words(text: str) -> list[str]:
    """Cheap whitespace/punctuation tokenizer, lowercased.

    Deliberately model-independent so a metric never depends on which tokenizer
    happened to be loaded.

    >>> tokenize_words("Hello, world! Hello.")
    ['hello', ',', 'world', '!', 'hello', '.']
    """
    return _WORD_RE.findall(text.lower())


def distinct_n(tokens: Sequence[str], n: int = 2) -> float:
    """Ratio of unique n-grams to total n-grams; 1.0 means no repetition.

    Returns 0.0 when the text is too short to contain an n-gram.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_rate(tokens: Sequence[str], n: int = 3) -> float:
    """Fraction of n-grams that occur more than once.

    The complement of a "novel n-gram" rate. High values indicate looping.
    """
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts: dict[tuple[str, ...], int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(grams)


def longest_repeated_ngram(tokens: Sequence[str], max_n: int = 20) -> int:
    """Length of the longest n-gram that appears at least twice.

    A blunt but reliable degeneration detector: healthy prose rarely repeats a
    10-gram, a looping model repeats one constantly.
    """
    best = 0
    upper = min(max_n, len(tokens) // 2)
    for n in range(1, upper + 1):
        seen: set[tuple[str, ...]] = set()
        found = False
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            if gram in seen:
                found = True
                break
            seen.add(gram)
        if found:
            best = n
        else:
            # No repeat of length n means no repeat of any longer length.
            break
    return best


def most_common_ngram_fraction(tokens: Sequence[str], n: int = 2) -> float:
    """Share of all n-grams taken by the single most frequent one.

    Added after an observed miss: at high steering strength the model produced
    ``"as they encounter each other, as they interact with each other, as they
    collide, as they merge, ..."`` -- obviously degenerate, yet it passed the
    distinct-n and longest-repeat checks because each clause ends differently.
    A single bigram occupying ~18% of all bigrams catches that pattern, where
    set-based diversity measures do not.
    """
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts: dict[tuple[str, ...], int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values()) / len(grams)


def type_token_ratio(tokens: Sequence[str]) -> float:
    """Unique tokens / total tokens. 0.0 for empty input."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def shannon_entropy(tokens: Sequence[str]) -> float:
    """Unigram entropy in bits. Collapsed output has near-zero entropy."""
    if not tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


@dataclass
class GenerationMetrics:
    """Model-free summary of one generated string."""

    num_chars: int
    num_words: int
    distinct_1: float
    distinct_2: float
    distinct_3: float
    repetition_3: float
    longest_repeat: int
    type_token_ratio: float
    entropy: float
    top_bigram_fraction: float
    is_empty: bool

    @property
    def degeneration_flag(self) -> bool:
        """Heuristic tripwire for obviously broken output.

        Thresholds are conservative: they fire on text that is plainly looping
        or empty, not on text that is merely repetitive. A True here means
        "inspect this generation", not "this intervention failed".

        These are heuristics, and they have been observed to miss things -- the
        ``top_bigram_fraction`` clause exists because the earlier rules scored a
        clearly-looping generation as clean. Treat a False as "no obvious
        breakage detected", not as "output is fine".
        """
        if self.is_empty:
            return True
        if self.num_words >= 30 and self.distinct_2 < 0.35:
            return True
        if self.longest_repeat >= 10:
            return True
        if self.num_words >= 20 and self.entropy < 2.0:
            return True
        if self.num_words >= 30 and self.top_bigram_fraction > 0.10:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["degeneration_flag"] = self.degeneration_flag
        return data


def score_generation(text: str) -> GenerationMetrics:
    """Compute all model-free metrics for one generated string."""
    tokens = tokenize_words(text)
    return GenerationMetrics(
        num_chars=len(text),
        num_words=len(tokens),
        distinct_1=distinct_n(tokens, 1),
        distinct_2=distinct_n(tokens, 2),
        distinct_3=distinct_n(tokens, 3),
        repetition_3=repetition_rate(tokens, 3),
        longest_repeat=longest_repeated_ngram(tokens),
        type_token_ratio=type_token_ratio(tokens),
        entropy=shannon_entropy(tokens),
        top_bigram_fraction=most_common_ngram_fraction(tokens, 2),
        is_empty=not text.strip(),
    )


def jaccard_similarity(a: str, b: str, n: int = 3) -> float:
    """n-gram Jaccard overlap between two strings, in ``[0, 1]``.

    Used to quantify *semantic drift*: how far an intervened generation moved
    from its baseline. 1.0 means identical n-gram sets.
    """
    ta, tb = tokenize_words(a), tokenize_words(b)
    if len(ta) < n or len(tb) < n:
        return 1.0 if a.strip() == b.strip() else 0.0
    ga = {tuple(ta[i : i + n]) for i in range(len(ta) - n + 1)}
    gb = {tuple(tb[i : i + n]) for i in range(len(tb) - n + 1)}
    union = ga | gb
    if not union:
        return 1.0
    return len(ga & gb) / len(union)


def compare_generations(baseline: str, intervened: str) -> dict[str, Any]:
    """Side-by-side comparison of a baseline and an intervened generation."""
    base_metrics = score_generation(baseline)
    int_metrics = score_generation(intervened)
    return {
        "baseline": base_metrics.to_dict(),
        "intervened": int_metrics.to_dict(),
        "identical": baseline == intervened,
        "jaccard_3": jaccard_similarity(baseline, intervened, n=3),
        "length_ratio": (
            int_metrics.num_words / base_metrics.num_words if base_metrics.num_words else None
        ),
        "distinct_2_delta": int_metrics.distinct_2 - base_metrics.distinct_2,
        "entropy_delta": int_metrics.entropy - base_metrics.entropy,
        "degeneration_introduced": int_metrics.degeneration_flag and not base_metrics.degeneration_flag,
    }
