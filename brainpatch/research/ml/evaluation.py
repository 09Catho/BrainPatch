"""Utility-retention probes.

Steering a residual stream toward a target behaviour is worthless if it also
breaks arithmetic, instruction-following, or basic factual recall. These probes
are the "did we damage the model" half of every intervention experiment.

They are small, exact-match, model-free-to-score, and require no paid API. They
are also deliberately easy: the point is to detect *breakage*, not to rank model
capability. A baseline Qwen2.5-1.5B-Instruct should get nearly all of them, so
a drop is signal rather than noise.

Like the contrast sets, these are hand-written development fixtures. A score
here is not a benchmark result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

#: (prompt, accepted answer patterns). Matching is case-insensitive substring
#: on the normalized generation, which tolerates the model's phrasing.
UTILITY_PROBES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("arithmetic", "What is 17 + 25? Reply with just the number.", ("42",)),
    ("arithmetic", "What is 8 times 7? Reply with just the number.", ("56",)),
    ("arithmetic", "What is 100 divided by 4? Reply with just the number.", ("25",)),
    ("factual_qa", "What is the capital of France? Reply with just the city name.", ("paris",)),
    ("factual_qa", "What is the chemical symbol for water? Reply with just the symbol.", ("h2o",)),
    ("factual_qa", "How many continents are there? Reply with just the number.", ("7", "seven")),
    (
        "instruction_following",
        "Reply with exactly the word BANANA and nothing else.",
        ("banana",),
    ),
    (
        "instruction_following",
        "List the first three positive even numbers, separated by commas.",
        ("2, 4, 6", "2,4,6"),
    ),
    (
        "reasoning",
        "If all cats are mammals and Whiskers is a cat, is Whiskers a mammal? Answer yes or no.",
        ("yes",),
    ),
    (
        "reasoning",
        "Tom is older than Ana. Ana is older than Bo. Who is youngest? Reply with just the name.",
        ("bo",),
    ),
)

#: Open-ended continuation prompts. Scored only for fluency/degeneration, since
#: there is no single correct answer.
CONTINUATION_PROBES: tuple[str, ...] = (
    "Write two sentences about the sea.",
    "Explain what a database index does, in about thirty words.",
    "Describe the process of making tea.",
)


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    return re.sub(r"\s+", " ", text.strip().lower())


def probe_correct(generation: str, accepted: Sequence[str]) -> bool:
    """Whether any accepted answer appears in the generation."""
    normalized = normalize(generation)
    return any(normalize(answer) in normalized for answer in accepted)


@dataclass
class UtilityReport:
    """Capability retention under one condition."""

    condition: str
    total: int
    correct: int
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)
    continuation_degeneration_rate: float = 0.0
    mean_continuation_words: float = 0.0
    generations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "by_category": {
                cat: {**counts, "accuracy": counts["correct"] / counts["total"]}
                for cat, counts in self.by_category.items()
            },
            "continuation_degeneration_rate": self.continuation_degeneration_rate,
            "mean_continuation_words": self.mean_continuation_words,
        }


def run_utility_probes(
    model: Any,
    *,
    condition: str = "baseline",
    generation: Any = None,
    include_continuations: bool = True,
) -> UtilityReport:
    """Run the capability probes against a (possibly patched) model.

    Parameters
    ----------
    model:
        A :class:`~brainpatch.research.ml.runtime.BrainPatchedModel`. Whatever patches
        are currently installed are active, which is the point: this is called
        once with them disabled and once with them enabled.
    """
    from brainpatch.evaluation.metrics import score_generation
    from brainpatch.research.ml.generation import GenerationConfig

    cfg = generation or GenerationConfig(max_new_tokens=48)
    report = UtilityReport(condition=condition, total=0, correct=0)

    for category, prompt, accepted in UTILITY_PROBES:
        text = model.generate(prompt, config=cfg)
        ok = probe_correct(text, accepted)
        report.total += 1
        report.correct += int(ok)
        bucket = report.by_category.setdefault(category, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(ok)
        report.generations.append(
            {
                "category": category,
                "prompt": prompt,
                "generation": text,
                "accepted": list(accepted),
                "correct": ok,
            }
        )

    if include_continuations:
        degenerate = 0
        words: list[int] = []
        long_cfg = GenerationConfig(max_new_tokens=96, do_sample=cfg.do_sample, seed=cfg.seed)
        for prompt in CONTINUATION_PROBES:
            text = model.generate(prompt, config=long_cfg)
            metrics = score_generation(text)
            degenerate += int(metrics.degeneration_flag)
            words.append(metrics.num_words)
            report.generations.append(
                {
                    "category": "continuation",
                    "prompt": prompt,
                    "generation": text,
                    "degeneration_flag": metrics.degeneration_flag,
                    "num_words": metrics.num_words,
                }
            )
        report.continuation_degeneration_rate = degenerate / len(CONTINUATION_PROBES)
        report.mean_continuation_words = sum(words) / len(words)

    return report


def compare_utility(baseline: UtilityReport, patched: UtilityReport) -> dict[str, Any]:
    """Quantify capability change between two conditions."""
    return {
        "baseline": baseline.to_dict(),
        "patched": patched.to_dict(),
        "accuracy_delta": patched.accuracy - baseline.accuracy,
        "degeneration_delta": (
            patched.continuation_degeneration_rate - baseline.continuation_degeneration_rate
        ),
        "length_ratio": (
            patched.mean_continuation_words / baseline.mean_continuation_words
            if baseline.mean_continuation_words
            else None
        ),
        "note": (
            f"{baseline.total} hand-written probes. A drop is a signal worth "
            "investigating, not evidence of degradation: at this sample size a "
            "one-item change carries no statistical weight. The absolute score is "
            "not a benchmark result."
        ),
    }
