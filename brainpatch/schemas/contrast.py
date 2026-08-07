"""Behavioural contrast datasets.

A contrast example pairs one prompt with a *positive* response (the behaviour we
want more of) and a *negative* response (the behaviour we want less of). The
difference in internal activations between the two is the starting point for
candidate-feature search.

These are **development fixtures**, not benchmarks. The sets shipped in
``examples/contrast/`` are small, hand-written, synthetic, and were never
validated against human judgement or an external standard. They exist to
exercise the pipeline; any number computed on them is a smoke-test number.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator


@dataclass
class ContrastExample:
    """One prompt with a contrasting pair of responses."""

    prompt: str
    positive_response: str
    negative_response: str
    category: str = "uncategorized"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("contrast example has an empty prompt")
        if not self.positive_response.strip():
            raise ValueError(f"empty positive_response for prompt {self.prompt[:60]!r}")
        if not self.negative_response.strip():
            raise ValueError(f"empty negative_response for prompt {self.prompt[:60]!r}")
        if self.positive_response.strip() == self.negative_response.strip():
            raise ValueError(
                f"positive and negative responses are identical for prompt {self.prompt[:60]!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContrastExample":
        missing = {"prompt", "positive_response", "negative_response"} - set(data)
        if missing:
            raise ValueError(f"contrast example is missing keys: {sorted(missing)}")
        return cls(
            prompt=str(data["prompt"]),
            positive_response=str(data["positive_response"]),
            negative_response=str(data["negative_response"]),
            category=str(data.get("category", "uncategorized")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ContrastSet:
    """A named collection of contrast examples.

    Attributes
    ----------
    synthetic:
        True for hand-written development fixtures. Always keep this honest --
        it is what stops a fixture from being reported as a benchmark.
    """

    name: str
    description: str
    examples: list[ContrastExample] = field(default_factory=list)
    synthetic: bool = True
    version: str = "0.1"

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[ContrastExample]:
        return iter(self.examples)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("contrast set must have a name")
        if not self.examples:
            raise ValueError(f"contrast set {self.name!r} is empty")
        for example in self.examples:
            example.validate()

    def categories(self) -> list[str]:
        """Distinct categories present, in sorted order."""
        return sorted({e.category for e in self.examples})

    def filter(self, category: str) -> "ContrastSet":
        """A new set containing only examples from ``category``."""
        return ContrastSet(
            name=f"{self.name}:{category}",
            description=self.description,
            examples=[e for e in self.examples if e.category == category],
            synthetic=self.synthetic,
            version=self.version,
        )

    def split(self, holdout_fraction: float, seed: int = 0) -> tuple["ContrastSet", "ContrastSet"]:
        """Deterministically split into (train, holdout).

        Uses a seeded :class:`random.Random` so that patch search and held-out
        evaluation never accidentally see the same examples across runs.
        """
        import random

        if not 0.0 < holdout_fraction < 1.0:
            raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
        indices = list(range(len(self.examples)))
        random.Random(seed).shuffle(indices)
        n_holdout = max(1, int(round(len(indices) * holdout_fraction)))
        holdout_idx = set(indices[:n_holdout])

        train = [e for i, e in enumerate(self.examples) if i not in holdout_idx]
        holdout = [e for i, e in enumerate(self.examples) if i in holdout_idx]
        return (
            ContrastSet(f"{self.name}:train", self.description, train, self.synthetic, self.version),
            ContrastSet(
                f"{self.name}:holdout", self.description, holdout, self.synthetic, self.version
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "synthetic": self.synthetic,
            "version": self.version,
            "examples": [e.to_dict() for e in self.examples],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContrastSet":
        if "name" not in data:
            raise ValueError("contrast set is missing 'name'")
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            examples=[ContrastExample.from_dict(e) for e in data.get("examples", [])],
            synthetic=bool(data.get("synthetic", True)),
            version=str(data.get("version", "0.1")),
        )

    @classmethod
    def from_json(cls, text: str) -> "ContrastSet":
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_examples(
        cls, name: str, description: str, examples: Iterable[ContrastExample]
    ) -> "ContrastSet":
        return cls(name=name, description=description, examples=list(examples))
