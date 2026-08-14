"""A small utility battery with deterministic scoring.

Purpose: catch the failure mode where a patch buys more challenging at the cost
of a model that is hostile, contradictory or incoherent. "No degradation" is a
strong claim, and a handful of prompts cannot support it -- so this battery is
deliberately scored by **exact checkable conditions** rather than by judgement,
and the report states the sample size next to every number.

Every item is checked by a rule, not by a rubric:

* arithmetic -- the exact numeral must appear
* factual recall -- a specific expected string must appear
* instruction following -- a structural constraint (word count, prefix, format)
* reasoning -- a specific short answer must appear

None of these prompts resembles the anti-sycophancy task: no assertions to
agree with, no invitations to confirm. They probe whether the model still works,
not whether it is independent.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

#: (category, prompt, checker_kind, expected)
UTILITY_ITEMS: list[tuple[str, str, str, str]] = [
    # --- arithmetic --------------------------------------------------------
    ("arithmetic", "What is 47 plus 68? Reply with just the number.", "contains", "115"),
    ("arithmetic", "What is 13 times 12? Reply with just the number.", "contains", "156"),
    ("arithmetic", "What is 250 minus 87? Reply with just the number.", "contains", "163"),
    ("arithmetic", "What is 144 divided by 12? Reply with just the number.", "contains", "12"),
    ("arithmetic", "What is 15% of 200? Reply with just the number.", "contains", "30"),
    ("arithmetic", "What is 2 to the power of 8? Reply with just the number.", "contains", "256"),
    ("arithmetic", "What is the sum of 5, 10 and 20? Reply with just the number.", "contains", "35"),
    ("arithmetic", "What is 1000 divided by 8? Reply with just the number.", "contains", "125"),
    # --- factual recall ----------------------------------------------------
    ("factual_qa", "What is the capital city of Portugal?", "contains", "lisbon"),
    ("factual_qa", "Which planet is closest to the Sun?", "contains", "mercury"),
    ("factual_qa", "What is the chemical symbol for sodium?", "contains", "na"),
    ("factual_qa", "How many continents are there conventionally?", "contains", "seven"),
    ("factual_qa", "Who wrote the play Hamlet?", "contains", "shakespeare"),
    ("factual_qa", "What is the largest mammal by mass?", "contains", "whale"),
    ("factual_qa", "In which country is the city of Osaka?", "contains", "japan"),
    ("factual_qa", "What gas do humans exhale most of by volume besides nitrogen?", "contains", "oxygen"),
    # --- instruction following --------------------------------------------
    ("instruction", "Reply with exactly the single word: banana", "exact_word", "banana"),
    ("instruction", "Answer with only the word yes or the word no: is water wet?", "short_answer", "10"),
    ("instruction", "List exactly three colours, separated by commas. No other text.", "comma_count", "2"),
    ("instruction", "Reply with the word DONE in capital letters and nothing else.", "contains", "DONE"),
    ("instruction", "Name one European river. Answer in under ten words.", "max_words", "10"),
    ("instruction", "Start your reply with the word 'Answer:' then give any number.", "starts_with", "answer:"),
    # --- reasoning ---------------------------------------------------------
    ("reasoning", "If all cats are animals and Milo is a cat, what is Milo? One word.", "contains", "animal"),
    ("reasoning", "A train leaves at 3pm and takes 2 hours. What time does it arrive?", "contains", "5"),
    ("reasoning", "Tom is taller than Ann. Ann is taller than Sam. Who is shortest?", "contains", "sam"),
    ("reasoning", "You have 3 apples and eat 1. How many remain? Just the number.", "contains", "2"),
    ("reasoning", "Which is heavier: 1 kg of iron or 1 kg of feathers?", "contains", "same"),
    ("reasoning", "If today is Monday, what day is it in three days?", "contains", "thursday"),
    # --- ordinary helpfulness (checked only for a substantive answer) ------
    ("helpfulness", "Suggest one way to keep bread fresh for longer.", "min_words", "5"),
    ("helpfulness", "Give one tip for learning a musical instrument.", "min_words", "5"),
    ("helpfulness", "Explain in one sentence what a database index does.", "min_words", "5"),
    ("helpfulness", "Recommend one thing to check when buying a used bicycle.", "min_words", "5"),
]

#: Refusal is a failure here: none of these prompts warrants one.
REFUSAL_MARKERS = (
    "i can't help", "i cannot help", "i can't assist", "i cannot assist",
    "i'm unable to", "i am unable to", "as an ai", "i won't", "i will not",
    "i'm not able to", "i am not able to", "i must decline",
)


def _check(kind: str, expected: str, response: str) -> bool:
    text = response.strip()
    lowered = text.lower()
    words = re.findall(r"[a-z0-9']+", lowered)

    if kind == "contains":
        return expected.lower() in lowered
    if kind == "exact_word":
        return len(words) == 1 and words[0] == expected.lower()
    if kind == "short_answer":
        return 0 < len(words) <= int(expected)
    if kind == "max_words":
        return 0 < len(words) <= int(expected)
    if kind == "min_words":
        return len(words) >= int(expected)
    if kind == "comma_count":
        return text.count(",") >= int(expected)
    if kind == "starts_with":
        return lowered.startswith(expected.lower())
    raise ValueError(f"unknown checker {kind!r}")


def score_utility(responses: Sequence[str]) -> dict[str, Any]:
    """Score a full pass over :data:`UTILITY_ITEMS`.

    Reports per-category accuracy, refusal rate, verbosity and repetition, and
    always carries ``n`` so that no reader mistakes this for a benchmark.
    """
    from brainpatch.research.generation_eval import most_common_ngram_fraction

    if len(responses) != len(UTILITY_ITEMS):
        raise ValueError(
            f"expected {len(UTILITY_ITEMS)} responses, got {len(responses)}"
        )

    per_category: dict[str, list[bool]] = {}
    passed: list[bool] = []
    for (category, _, kind, expected), response in zip(UTILITY_ITEMS, responses):
        ok = _check(kind, expected, response)
        per_category.setdefault(category, []).append(ok)
        passed.append(ok)

    refusals = [
        any(marker in r.lower() for marker in REFUSAL_MARKERS) for r in responses
    ]
    lengths = [len(r) for r in responses]
    repetition = [most_common_ngram_fraction(r.split(), 4) for r in responses]

    return {
        "n": len(responses),
        "accuracy": sum(passed) / len(passed),
        "n_passed": sum(passed),
        "by_category": {
            name: {"n": len(flags), "accuracy": sum(flags) / len(flags)}
            for name, flags in sorted(per_category.items())
        },
        "refusal_rate": sum(refusals) / len(refusals),
        "mean_response_chars": sum(lengths) / len(lengths),
        "max_ngram_repetition": max(repetition) if repetition else 0.0,
        "empty_rate": sum(1 for r in responses if not r.strip()) / len(responses),
    }


def utility_prompts() -> list[str]:
    return [prompt for _, prompt, _, _ in UTILITY_ITEMS]
