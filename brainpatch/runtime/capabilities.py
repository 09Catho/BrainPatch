"""Backend capability reporting.

Backends genuinely differ in what they can do, and the honest thing is to say so
rather than emulate a feature badly. llama.cpp applies a control vector for a
whole run; vLLM batches concurrent requests with shared model state; MLX has no
CI hardware here. Each of those is a real constraint, and a user deserves to
know before they design around a capability that is not there.

:class:`Capabilities` is what ``brainpatch backends`` and ``brainpatch doctor``
print, and what the runtime consults before accepting an operation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Every capability flag, in display order.
CAPABILITY_FLAGS: tuple[str, ...] = (
    "static_intervention",
    "dynamic_schedule",
    "multiple_patches",
    "streaming",
    "cpu",
    "cuda",
    "mps",
    "apple_silicon",
    "server",
    "concurrent_requests",
    "quantization",
    "per_request_strength",
)


@dataclass
class Capabilities:
    """What a backend can actually do.

    Defaults are all False on purpose: a backend must opt in to each claim, so a
    forgotten flag understates rather than overstates support.
    """

    name: str

    static_intervention: bool = False
    """Apply a fixed-strength patch for a whole generation."""

    dynamic_schedule: bool = False
    """Change strength between generated tokens."""

    multiple_patches: bool = False
    streaming: bool = False

    cpu: bool = False
    cuda: bool = False
    mps: bool = False
    apple_silicon: bool = False

    server: bool = False
    concurrent_requests: bool = False
    per_request_strength: bool = False

    quantization: tuple[str, ...] = ()
    """Quantization formats actually exercised, e.g. ``("Q4_K_M",)``."""

    notes: dict[str, str] = field(default_factory=dict)
    """Per-capability explanation, especially for the False ones."""

    def supports(self, flag: str) -> bool:
        if flag not in CAPABILITY_FLAGS:
            raise ValueError(f"unknown capability {flag!r}; expected one of {CAPABILITY_FLAGS}")
        value = getattr(self, flag)
        return bool(value)

    def require(self, flag: str) -> None:
        """Raise a useful error if the backend cannot do ``flag``."""
        if not self.supports(flag):
            note = self.notes.get(flag)
            detail = f" {note}" if note else ""
            raise NotImplementedError(
                f"the {self.name!r} backend does not support {flag!r}.{detail}"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["quantization"] = list(self.quantization)
        return data

    def matrix_row(self) -> dict[str, Any]:
        """Compact row for the capability matrix table."""
        row: dict[str, Any] = {"backend": self.name}
        for flag in CAPABILITY_FLAGS:
            value = getattr(self, flag)
            row[flag] = bool(value) if not isinstance(value, tuple) else bool(value)
        row["quantization"] = ", ".join(self.quantization) or "-"
        return row
