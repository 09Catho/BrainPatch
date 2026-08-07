"""Token-indexed strength schedules for dynamic steering.

A schedule maps *generated token index* to a strength multiplier, using
step-hold semantics: the value at index ``n`` is the value of the largest
keyframe ``<= n``. This is what makes "turn the patch on 20 tokens into the
answer" expressible::

    StrengthSchedule({0: 0.0, 20: 1.0, 40: 2.0})

    index:     0 ... 19   20 ... 39   40 ...
    strength:  0.0        1.0         2.0

Optional linear interpolation smooths the transitions, which matters because an
abrupt large change in the residual stream mid-generation can itself derail the
model -- a confound worth being able to rule out.

Everything here is pure arithmetic so it can be tested without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class StrengthSchedule:
    """Piecewise strength multiplier over generated-token index.

    Parameters
    ----------
    keyframes:
        Mapping of token index -> strength multiplier. Indices count *generated*
        tokens, starting at 0 for the first token the model produces; prompt
        tokens are not counted.
    interpolate:
        When True, values between keyframes are linearly interpolated. When
        False (default) the schedule is a step function.
    default:
        Strength used before the first keyframe when index 0 is not specified.
    """

    keyframes: Mapping[int, float]
    interpolate: bool = False
    default: float = 1.0

    def __post_init__(self) -> None:
        if not self.keyframes:
            raise ValueError("a schedule needs at least one keyframe")
        normalized: dict[int, float] = {}
        for key, value in self.keyframes.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"schedule keys must be integers, got {key!r}") from exc
            if index < 0:
                raise ValueError(f"schedule token index must be >= 0, got {index}")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"schedule value at {index} must be numeric, got {value!r}")
            normalized[index] = float(value)
        # frozen dataclass: bypass __setattr__ to install the cleaned mapping
        object.__setattr__(self, "keyframes", dict(sorted(normalized.items())))

    @property
    def steps(self) -> list[int]:
        """Sorted keyframe indices."""
        return list(self.keyframes)

    def strength_at(self, token_index: int) -> float:
        """Strength multiplier for the given generated-token index.

        >>> s = StrengthSchedule({0: 0.0, 20: 1.0, 40: 2.0})
        >>> s.strength_at(0), s.strength_at(19), s.strength_at(20), s.strength_at(100)
        (0.0, 0.0, 1.0, 2.0)
        """
        if token_index < 0:
            raise ValueError(f"token_index must be >= 0, got {token_index}")

        steps = self.steps
        # Before the first keyframe: fall back to `default`.
        if token_index < steps[0]:
            return self.default

        # Find the last keyframe at or before token_index.
        lo = 0
        for i, step in enumerate(steps):
            if step <= token_index:
                lo = i
            else:
                break

        left = steps[lo]
        left_value = self.keyframes[left]
        if not self.interpolate or lo + 1 >= len(steps):
            return left_value

        right = steps[lo + 1]
        right_value = self.keyframes[right]
        span = right - left
        if span <= 0:  # defensive; sorted unique keys make this unreachable
            return left_value
        alpha = (token_index - left) / span
        return left_value + alpha * (right_value - left_value)

    def is_constant(self) -> bool:
        """True if the schedule never changes strength."""
        values = set(self.keyframes.values())
        if len(values) > 1:
            return False
        return self.steps[0] == 0 or values == {self.default}

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyframes": {str(k): v for k, v in self.keyframes.items()},
            "interpolate": self.interpolate,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StrengthSchedule":
        """Build from a serialized schedule.

        Accepts both the full form ``{"keyframes": {...}, ...}`` and the bare
        ``{"0": 0.0, "20": 1.0}`` shorthand used inside patch files.
        """
        if "keyframes" in data:
            raw = data["keyframes"]
            return cls(
                keyframes={int(k): float(v) for k, v in raw.items()},
                interpolate=bool(data.get("interpolate", False)),
                default=float(data.get("default", 1.0)),
            )
        return cls(keyframes={int(k): float(v) for k, v in data.items()})

    @classmethod
    def constant(cls, strength: float = 1.0) -> "StrengthSchedule":
        """A schedule that holds one strength for the whole generation."""
        return cls(keyframes={0: float(strength)}, default=float(strength))
