"""Feature-database records.

A :class:`FeatureRecord` describes one SAE feature: how often it fires, how
strongly, and which token contexts drive it hardest.

Scientific note
---------------
``FeatureRecord.hypothesis`` is exactly that -- a *hypothesis*. Top activating
examples are correlational evidence and nothing more. The field
:attr:`FeatureRecord.evidence_level` records how much support a semantic
description actually has, and it never advances past ``"correlational"``
automatically. Only an intervention experiment with controls can move a feature
to ``"causal"``, and that transition is performed by the validation pipeline
writing a result, never by a labelling heuristic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

#: Ordered strength of evidence behind a semantic claim about a feature.
EvidenceLevel = Literal[
    "none",  # no description offered
    "correlational",  # top-activating contexts look suggestive; nothing more
    "predictive",  # feature activation predicts a behaviour on held-out data
    "interventional",  # steering it changes behaviour, controls not yet complete
    "causal",  # steering changes behaviour and scale-matched controls do not
]

EVIDENCE_ORDER: tuple[str, ...] = (
    "none",
    "correlational",
    "predictive",
    "interventional",
    "causal",
)


@dataclass
class FeatureContext:
    """One high-activation occurrence of a feature, with surrounding text."""

    example_index: int
    token_position: int
    token_id: int
    token_text: str
    activation: float
    #: Decoded text a few tokens either side, for human inspection.
    context_before: str = ""
    context_after: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureContext":
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        return cls(**{k: v for k, v in data.items() if k in known})  # type: ignore[arg-type]


@dataclass
class FeatureStats:
    """Activation statistics for one feature over an activation corpus."""

    fire_count: int = 0
    total_tokens: int = 0
    mean_activation: float = 0.0
    """Mean over *firing* tokens only (zeros excluded)."""
    max_activation: float = 0.0
    std_activation: float = 0.0
    decoder_norm: float = 0.0

    @property
    def firing_rate(self) -> float:
        """Fraction of tokens on which this feature is active."""
        if self.total_tokens == 0:
            return 0.0
        return self.fire_count / self.total_tokens

    @property
    def is_dead(self) -> bool:
        """A feature that never fired over the analysed corpus."""
        return self.fire_count == 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["firing_rate"] = self.firing_rate
        data["is_dead"] = self.is_dead
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureStats":
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        return cls(**{k: v for k, v in data.items() if k in known})  # type: ignore[arg-type]


@dataclass
class FeatureRecord:
    """A single entry in the feature database."""

    feature_id: int
    stats: FeatureStats = field(default_factory=FeatureStats)
    top_contexts: list[FeatureContext] = field(default_factory=list)

    #: Tentative, human- or machine-suggested description. NOT a validated label.
    hypothesis: str | None = None
    evidence_level: EvidenceLevel = "none"
    #: Free-form pointers to experiments that produced the evidence.
    evidence_refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.evidence_level not in EVIDENCE_ORDER:
            raise ValueError(
                f"unknown evidence_level {self.evidence_level!r}; "
                f"expected one of {EVIDENCE_ORDER}"
            )
        if self.hypothesis is not None and self.evidence_level == "none":
            # A description always carries at least correlational weight; being
            # explicit here prevents an unlabelled-looking record from silently
            # shipping a semantic claim.
            self.evidence_level = "correlational"

    @property
    def is_validated(self) -> bool:
        """True only when controlled intervention evidence exists."""
        return self.evidence_level == "causal"

    def label_for_display(self) -> str:
        """Human-facing label that never overstates the evidence."""
        if self.hypothesis is None:
            return f"feature {self.feature_id} (no description)"
        if self.evidence_level == "causal":
            return f"feature {self.feature_id}: {self.hypothesis}"
        return f"feature {self.feature_id}: {self.hypothesis} [{self.evidence_level}, unvalidated]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "stats": self.stats.to_dict(),
            "top_contexts": [c.to_dict() for c in self.top_contexts],
            "hypothesis": self.hypothesis,
            "evidence_level": self.evidence_level,
            "evidence_refs": list(self.evidence_refs),
        }

    def to_json(self) -> str:
        """Single-line JSON, suitable for a ``features.jsonl`` row."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureRecord":
        return cls(
            feature_id=int(data["feature_id"]),
            stats=FeatureStats.from_dict(data.get("stats", {})),
            top_contexts=[FeatureContext.from_dict(c) for c in data.get("top_contexts", [])],
            hypothesis=data.get("hypothesis"),
            evidence_level=data.get("evidence_level", "none"),
            evidence_refs=list(data.get("evidence_refs", [])),
        )
