"""Patch-to-model compatibility checking.

A patch vector is a direction in one model's residual basis at one layer. The
same architecture with different weights has a different basis, so "it is also a
Qwen2ForCausalLM" is not evidence that a direction transfers. Applying a patch
to the wrong model does not degrade gracefully -- it adds an arbitrary vector and
produces confident nonsense, which is worse than an error.

Hence three explicit modes, defaulting to the strictest:

``strict`` (default)
    Model id must match. Revision must match when both are known. Hidden size
    and layer count must match.

``architecture``
    Model id may differ but the architecture string, hidden size and layer count
    must match. For fine-tunes and merges of the same base. Warns.

``unsafe``
    Only the geometry is checked -- hidden size must match and the layer must
    exist, because anything else cannot be executed at all. Warns loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from brainpatch.patch.format import Manifest

CompatibilityMode = Literal["strict", "architecture", "unsafe"]
COMPATIBILITY_MODES: tuple[str, ...] = ("strict", "architecture", "unsafe")


class PatchCompatibilityError(ValueError):
    """The patch does not match the model it is being applied to."""


@dataclass
class ModelDescriptor:
    """What the backend knows about the loaded model."""

    model_id: str
    hidden_size: int
    num_layers: int
    architecture: str = ""
    revision: str | None = None


@dataclass
class CompatibilityReport:
    """Outcome of a compatibility check."""

    ok: bool
    mode: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise PatchCompatibilityError(
                "patch is not compatible with this model:\n  - "
                + "\n  - ".join(self.errors)
                + f"\n\nChecked in '{self.mode}' mode. If you understand the risk, "
                "re-run with compatibility_mode='architecture' or 'unsafe'."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def check_compatibility(
    manifest: Manifest,
    model: ModelDescriptor,
    *,
    mode: CompatibilityMode = "strict",
) -> CompatibilityReport:
    """Check a patch against a loaded model. Never raises; inspect ``.ok``."""
    if mode not in COMPATIBILITY_MODES:
        raise ValueError(f"unknown compatibility mode {mode!r}; expected {COMPATIBILITY_MODES}")

    errors: list[str] = []
    warnings: list[str] = []
    spec = manifest.base_model

    # -- geometry: required in every mode, because it is what makes the
    #    arithmetic executable at all.
    if spec.hidden_size != model.hidden_size:
        errors.append(
            f"hidden size mismatch: patch was built for {spec.hidden_size}, "
            f"model has {model.hidden_size}"
        )
    for layer in manifest.layers:
        if layer >= model.num_layers:
            errors.append(
                f"patch targets layer {layer} but the model has {model.num_layers} layers"
            )

    if mode == "unsafe":
        if spec.model_id != model.model_id:
            warnings.append(
                f"UNSAFE MODE: applying a patch built for {spec.model_id!r} to "
                f"{model.model_id!r}. Feature directions are not transferable between "
                "models; output is not meaningful."
            )
        return CompatibilityReport(ok=not errors, mode=mode, errors=errors, warnings=warnings)

    # -- architecture-level checks
    if spec.num_layers and spec.num_layers != model.num_layers:
        errors.append(
            f"layer count mismatch: patch declares {spec.num_layers}, "
            f"model has {model.num_layers}"
        )
    if spec.architecture and model.architecture and spec.architecture != model.architecture:
        errors.append(
            f"architecture mismatch: patch is for {spec.architecture!r}, "
            f"model is {model.architecture!r}"
        )

    if mode == "architecture":
        if spec.model_id != model.model_id:
            warnings.append(
                f"model id differs (patch: {spec.model_id!r}, loaded: {model.model_id!r}). "
                "Allowed in 'architecture' mode, but the direction was fitted on "
                "different weights and may not transfer."
            )
        return CompatibilityReport(ok=not errors, mode=mode, errors=errors, warnings=warnings)

    # -- strict
    if spec.model_id != model.model_id:
        errors.append(
            f"model mismatch: patch targets {spec.model_id!r} but {model.model_id!r} "
            "is loaded"
        )
    if spec.revision and model.revision and spec.revision != model.revision:
        errors.append(
            f"revision mismatch: patch was derived from {spec.revision[:12]}… but "
            f"{model.revision[:12]}… is loaded"
        )
    elif spec.revision and not model.revision:
        warnings.append(
            f"patch pins revision {spec.revision[:12]}… but the loaded model's "
            "revision is unknown, so it could not be checked"
        )

    return CompatibilityReport(ok=not errors, mode=mode, errors=errors, warnings=warnings)


def validate_strength(manifest: Manifest, strength: float) -> float:
    """Clamp a live strength to the patch's declared envelope, warning if clipped."""
    clamped = manifest.clamp_strength(strength)
    return clamped
