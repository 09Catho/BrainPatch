"""The backend contract every inference engine adapter implements.

Design rule: this module imports nothing heavy. It defines the interface and the
bookkeeping that is identical across engines -- which patches are installed, what
their live strength is, how a schedule resolves at token *n* -- so each backend
only has to implement the part that is genuinely engine-specific: loading a
model, injecting a vector, and generating.

That split is what keeps the tricky logic (strength resolution, the guarantee
that strength 0 is exactly baseline) in one tested place rather than
reimplemented four times with four different bugs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

from brainpatch.patch.format import Manifest
from brainpatch.patch.loader import LoadedPatch
from brainpatch.patch.validation import (
    CompatibilityMode,
    CompatibilityReport,
    ModelDescriptor,
    check_compatibility,
)
from brainpatch.runtime.capabilities import Capabilities
from brainpatch.runtime.scheduling import StrengthSchedule

#: Coefficients below this are treated as exactly zero, so a zeroed patch is
#: bit-identical to baseline rather than merely close to it.
STRENGTH_EPSILON = 1e-12


@dataclass
class GenerationConfig:
    """Sampling settings, shared verbatim across compared conditions."""

    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    seed: int = 0
    stop: list[str] = field(default_factory=list)

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "stop": list(self.stop),
        }


@dataclass
class ActivePatch:
    """An installed patch plus its live, user-controllable state."""

    patch: LoadedPatch
    strength: float = 1.0
    enabled: bool = True
    schedule: StrengthSchedule | None = None

    def __post_init__(self) -> None:
        if self.schedule is None and self.patch.manifest.schedule is not None:
            self.schedule = StrengthSchedule.from_dict(self.patch.manifest.schedule)

    @property
    def name(self) -> str:
        return self.patch.manifest.name

    @property
    def manifest(self) -> Manifest:
        return self.patch.manifest

    def multiplier_at(self, token_index: int) -> float:
        """Live multiplier at a generated-token index, clamped to the envelope."""
        if not self.enabled:
            return 0.0
        value = self.manifest.clamp_strength(self.strength)
        if self.schedule is not None:
            value *= self.schedule.strength_at(token_index)
        return value


@dataclass
class ResolvedEdit:
    """One vector to add, with its final coefficient, at one layer."""

    layer: int
    hook: str
    vector_key: str
    coefficient: float
    patch_name: str


class BrainPatchBackend(ABC):
    """Common interface across Transformers, llama.cpp, vLLM and MLX."""

    #: Short identifier used by ``--backend`` and in capability tables.
    name: str = "abstract"

    def __init__(self) -> None:
        self._patches: dict[str, ActivePatch] = {}
        self._compatibility_mode: CompatibilityMode = "strict"

    # -- engine-specific -------------------------------------------------------

    @classmethod
    @abstractmethod
    def is_available(cls) -> tuple[bool, str]:
        """``(available, reason)``. Must not raise, even with nothing installed."""

    @classmethod
    @abstractmethod
    def capabilities(cls) -> Capabilities:
        """What this backend can do. Callable without the engine installed."""

    @abstractmethod
    def load_model(self, model: str, **kwargs: Any) -> None:
        """Load the frozen base model. Never modifies weights on disk."""

    @abstractmethod
    def describe_model(self) -> ModelDescriptor:
        """Architecture facts discovered from the loaded model."""

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any) -> str:
        """Generate a completion with all enabled patches active."""

    def stream(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any
    ) -> Iterator[str]:
        """Yield incremental text. Backends without streaming yield once."""
        self.capabilities().require("streaming")
        yield self.generate(prompt, config, **kwargs)

    # -- shared patch bookkeeping ---------------------------------------------

    @property
    def patches(self) -> dict[str, ActivePatch]:
        return self._patches

    def validate_patch(
        self, patch: LoadedPatch, *, mode: CompatibilityMode | None = None
    ) -> CompatibilityReport:
        return check_compatibility(
            patch.manifest,
            self.describe_model(),
            mode=mode or self._compatibility_mode,
        )

    def install_patch(
        self,
        patch: LoadedPatch,
        *,
        strength: float | None = None,
        mode: CompatibilityMode | None = None,
    ) -> ActivePatch:
        """Validate then register a patch. Nothing changes if validation fails."""
        report = self.validate_patch(patch, mode=mode)
        report.raise_if_failed()
        for warning in report.warnings:
            self._warn(warning)

        backend_status = patch.manifest.backend_status(self.name)
        if backend_status in {"unsupported", "implemented"}:
            self._warn(
                f"patch {patch.manifest.name!r} declares backend '{self.name}' as "
                f"'{backend_status}' -- it has not been verified on this engine."
            )

        active = ActivePatch(
            patch=patch,
            strength=(
                patch.manifest.default_strength if strength is None else float(strength)
            ),
        )
        if len(self._patches) >= 1 and not self.capabilities().multiple_patches:
            existing = next(iter(self._patches))
            if existing != active.name:
                raise NotImplementedError(
                    f"the {self.name!r} backend supports one patch at a time; "
                    f"{existing!r} is already installed"
                )
        self._patches[active.name] = active
        self._on_patches_changed()
        return active

    def remove_patch(self, name: str) -> None:
        if name not in self._patches:
            raise KeyError(f"no patch named {name!r} is installed on this backend")
        del self._patches[name]
        self._on_patches_changed()

    def set_strength(self, name: str, strength: float) -> float:
        """Set live strength; returns the value after clamping."""
        active = self._require(name)
        clamped = active.manifest.clamp_strength(strength)
        if clamped != float(strength):
            self._warn(
                f"strength {strength} clamped to {clamped} by patch "
                f"{name!r} (max_abs_strength={active.manifest.max_abs_strength})"
            )
        active.strength = clamped
        self._on_patches_changed()
        return clamped

    def set_enabled(self, name: str, enabled: bool) -> None:
        self._require(name).enabled = bool(enabled)
        self._on_patches_changed()

    def set_schedule(self, name: str, schedule: StrengthSchedule | dict[int, float] | None) -> None:
        self.capabilities().require("dynamic_schedule")
        active = self._require(name)
        if isinstance(schedule, dict):
            schedule = StrengthSchedule(schedule)
        active.schedule = schedule
        self._on_patches_changed()

    def list_patches(self) -> list[str]:
        return list(self._patches)

    def _require(self, name: str) -> ActivePatch:
        if name not in self._patches:
            installed = ", ".join(self._patches) or "none"
            raise KeyError(f"no patch named {name!r} is installed (installed: {installed})")
        return self._patches[name]

    # -- edit resolution -------------------------------------------------------

    def resolve_edits(
        self,
        token_index: int = 0,
        layer: int | None = None,
        *,
        is_prompt_pass: bool | None = None,
    ) -> list[ResolvedEdit]:
        """Vectors to add at a generated-token index.

        An empty list is the signal to leave activations completely untouched.
        That is what makes strength 0 identical to baseline rather than a no-op
        addition that still round-trips through floating point.

        ``is_prompt_pass`` lets an intervention restrict itself to the prompt or
        to generated tokens. Backends that cannot distinguish the two pass
        ``None``, in which case every intervention applies and the caller is
        responsible for knowing that a site-restricted patch is being applied
        more broadly than it was measured.
        """
        edits: list[ResolvedEdit] = []
        for active in self._patches.values():
            multiplier = active.multiplier_at(token_index)
            if abs(multiplier) < STRENGTH_EPSILON:
                continue
            for intervention in active.manifest.interventions:
                if layer is not None and intervention.layer != layer:
                    continue
                site = getattr(intervention, "site", "all")
                if is_prompt_pass is not None and site != "all":
                    if site == "prompt" and not is_prompt_pass:
                        continue
                    if site == "continuation" and is_prompt_pass:
                        continue
                coefficient = multiplier * intervention.coefficient
                if abs(coefficient) < STRENGTH_EPSILON:
                    continue
                edits.append(
                    ResolvedEdit(
                        layer=intervention.layer,
                        hook=intervention.hook,
                        vector_key=intervention.vector,
                        coefficient=coefficient,
                        patch_name=active.name,
                    )
                )
        return edits

    def active_layers(self) -> list[int]:
        return sorted({e.layer for e in self.resolve_edits(0)})

    def vector_values(self, patch_name: str, key: str) -> list[float]:
        return list(self._require(patch_name).patch.vector_for(key).data)

    # -- hooks for subclasses --------------------------------------------------

    def _on_patches_changed(self) -> None:
        """Called after any patch-state mutation. Override to rebuild caches."""

    def _warn(self, message: str) -> None:
        import warnings

        warnings.warn(f"[brainpatch:{self.name}] {message}", stacklevel=3)

    def unload(self) -> None:
        """Release engine resources. Safe to call more than once."""

    def __enter__(self) -> "BrainPatchBackend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.unload()
