"""``BrainPatchedModel`` -- the user-facing Python API.

::

    from brainpatch import BrainPatchedModel

    model = BrainPatchedModel.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        backend="transformers",
        device="auto",
    )

    patch = model.install("09Catho/example-patch")
    patch.strength = 0.8

    print(model.generate("Evaluate my idea."))

This is a thin facade over a :class:`~brainpatch.runtime.base.BrainPatchBackend`.
It owns no intervention logic of its own -- that lives in the backend contract,
so every engine behaves identically where it can and reports honestly where it
cannot.

Nothing here knows or cares where a patch was trained.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

from brainpatch.patch.loader import LoadedPatch, load_patch
from brainpatch.patch.registry import PatchRegistry, default_registry
from brainpatch.patch.validation import CompatibilityMode
from brainpatch.runtime.auto import select_backend
from brainpatch.runtime.base import ActivePatch, BrainPatchBackend, GenerationConfig
from brainpatch.runtime.capabilities import Capabilities
from brainpatch.runtime.scheduling import StrengthSchedule


class PatchHandle:
    """Live control surface for one installed patch.

    ``patch.strength = 0.8`` and ``patch.schedule = {...}`` are the ergonomic
    forms; both route to the backend so clamping and capability checks still
    apply.
    """

    def __init__(self, model: "BrainPatchedModel", name: str) -> None:
        self._model = model
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def _active(self) -> ActivePatch:
        return self._model.backend.patches[self._name]

    @property
    def strength(self) -> float:
        return self._active.strength

    @strength.setter
    def strength(self, value: float) -> None:
        self._model.backend.set_strength(self._name, value)

    @property
    def enabled(self) -> bool:
        return self._active.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._model.backend.set_enabled(self._name, bool(value))

    @property
    def schedule(self) -> StrengthSchedule | None:
        return self._active.schedule

    @schedule.setter
    def schedule(self, value: StrengthSchedule | dict[int, float] | None) -> None:
        self._model.backend.set_schedule(self._name, value)

    @property
    def manifest(self) -> Any:
        return self._active.manifest

    @property
    def evidence_level(self) -> str:
        return str(self._active.manifest.evidence_level)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "strength": self.strength,
            "enabled": self.enabled,
            "evidence_level": self.evidence_level,
            "layers": self._active.manifest.layers,
            "scheduled": self.schedule is not None,
        }

    def __repr__(self) -> str:
        state = "on" if self.enabled else "off"
        return f"<PatchHandle {self._name!r} strength={self.strength:+.3f} [{state}]>"


class BrainPatchedModel:
    """A frozen language model with installable activation patches."""

    def __init__(self, backend: BrainPatchBackend, registry: PatchRegistry | None = None) -> None:
        self.backend = backend
        self.registry = registry or default_registry()
        self.compatibility_mode: CompatibilityMode = "strict"

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        backend: str = "auto",
        device: str = "auto",
        dtype: str = "auto",
        revision: str | None = None,
        registry: PatchRegistry | None = None,
        compatibility_mode: CompatibilityMode = "strict",
        **kwargs: Any,
    ) -> "BrainPatchedModel":
        """Load a frozen base model on the chosen (or best available) backend.

        ``backend="auto"`` picks the first available engine. Naming a backend
        that is unavailable raises rather than substituting a different one.
        """
        backend_cls = select_backend(backend)
        instance = backend_cls()
        instance.load_model(model, revision=revision, device=device, dtype=dtype, **kwargs)
        patched = cls(instance, registry=registry)
        patched.compatibility_mode = compatibility_mode
        return patched

    # -- patch management ------------------------------------------------------

    def install(
        self,
        ref: str | os.PathLike[str] | LoadedPatch,
        *,
        strength: float | None = None,
        compatibility_mode: CompatibilityMode | None = None,
    ) -> PatchHandle:
        """Install a patch by installed name, file path, HF reference, or object.

        Resolution order: an already-installed registry name, then a filesystem
        path, then a Hugging Face ``owner/repo`` reference (which downloads only
        the patch artifact, never the base model).
        """
        if isinstance(ref, LoadedPatch):
            loaded = ref
        else:
            text = str(ref)
            try:
                loaded = load_patch(self.registry.resolve(text))
            except Exception:
                # Not installed and not a local file: try the Hub, then install.
                installed = self.registry.install(text)
                loaded = installed.load()

        active = self.backend.install_patch(
            loaded,
            strength=strength,
            mode=compatibility_mode or self.compatibility_mode,
        )
        return PatchHandle(self, active.name)

    def remove_patch(self, name: str) -> None:
        self.backend.remove_patch(name)

    def enable_patch(self, name: str) -> None:
        self.backend.set_enabled(name, True)

    def disable_patch(self, name: str) -> None:
        self.backend.set_enabled(name, False)

    def set_patch_strength(self, name: str, strength: float) -> float:
        return self.backend.set_strength(name, strength)

    def set_patch_schedule(
        self, name: str, schedule: StrengthSchedule | dict[int, float] | None
    ) -> None:
        self.backend.set_schedule(name, schedule)

    def list_patches(self) -> list[str]:
        return self.backend.list_patches()

    def patch(self, name: str) -> PatchHandle:
        if name not in self.backend.patches:
            raise KeyError(f"no patch named {name!r} is installed")
        return PatchHandle(self, name)

    # -- generation ------------------------------------------------------------

    def generate(self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any) -> str:
        return self.backend.generate(prompt, config, **kwargs)

    def stream(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any
    ) -> Iterator[str]:
        return self.backend.stream(prompt, config, **kwargs)

    def compare(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any
    ) -> dict[str, str]:
        """Generate the same prompt with patches off and on.

        Disables patches rather than uninstalling them, so strengths and
        schedules survive the comparison.
        """
        states = {name: p.enabled for name, p in self.backend.patches.items()}
        try:
            for name in states:
                self.backend.set_enabled(name, False)
            baseline = self.generate(prompt, config, **kwargs)
            for name in states:
                self.backend.set_enabled(name, True)
            patched = self.generate(prompt, config, **kwargs)
        finally:
            for name, was_enabled in states.items():
                self.backend.set_enabled(name, was_enabled)
        return {"baseline": baseline, "patched": patched}

    # -- introspection ---------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return type(self.backend).capabilities()

    def describe(self) -> dict[str, Any]:
        descriptor = self.backend.describe_model()
        return {
            "backend": self.backend.name,
            "model": descriptor.model_id,
            "architecture": descriptor.architecture,
            "hidden_size": descriptor.hidden_size,
            "num_layers": descriptor.num_layers,
            "revision": descriptor.revision,
            "compatibility_mode": self.compatibility_mode,
            "patches": [PatchHandle(self, n).describe() for n in self.backend.patches],
        }

    def unload(self) -> None:
        self.backend.unload()

    def __enter__(self) -> "BrainPatchedModel":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.unload()

    def __repr__(self) -> str:
        patches = ", ".join(self.backend.patches) or "none"
        return f"<BrainPatchedModel backend={self.backend.name!r} patches=[{patches}]>"
