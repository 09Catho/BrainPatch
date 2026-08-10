"""BrainPatch: tiny, installable activation patches for frozen language models.

    from brainpatch import BrainPatchedModel

    model = BrainPatchedModel.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model.install("09Catho/example-patch")
    print(model.generate("Evaluate my idea."))

Import policy
-------------
``import brainpatch`` works on a bare Python 3.10+ with no ML stack: the patch
format, registry, validation and CLI are pure standard library plus a couple of
small pure-Python dependencies.

Anything that needs torch, transformers, vLLM or llama.cpp is imported lazily
when a backend is actually instantiated. :class:`BrainPatchedModel` is exposed
through ``__getattr__`` for the same reason -- naming it here must not drag the
runtime's dependencies into a process that only wants to inspect a patch file.

Research tooling lives under :mod:`brainpatch.research` and is never imported by
the runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "1.1.0"

from brainpatch.patch.format import (
    FORMAT_VERSION,
    SUFFIX,
    BaseModelSpec,
    Intervention,
    Manifest,
    PatchFormatError,
)
from brainpatch.patch.loader import LoadedPatch, load_patch, save_patch
from brainpatch.patch.registry import PatchRegistry, default_registry
from brainpatch.patch.validation import PatchCompatibilityError
from brainpatch.steering.schedule import StrengthSchedule

if TYPE_CHECKING:  # pragma: no cover - typing only
    from brainpatch.runtime.base import GenerationConfig
    from brainpatch.runtime.model import BrainPatchedModel, PatchHandle

#: Symbols resolved lazily so the top-level import stays light.
_LAZY: dict[str, tuple[str, str]] = {
    "BrainPatchedModel": ("brainpatch.runtime.model", "BrainPatchedModel"),
    "PatchHandle": ("brainpatch.runtime.model", "PatchHandle"),
    "GenerationConfig": ("brainpatch.runtime.base", "GenerationConfig"),
    "Capabilities": ("brainpatch.runtime.capabilities", "Capabilities"),
    "available_backends": ("brainpatch.runtime.auto", "available_backends"),
    "environment_report": ("brainpatch.runtime.auto", "environment_report"),
}

__all__ = [
    "BaseModelSpec",
    "BrainPatchedModel",
    "Capabilities",
    "FORMAT_VERSION",
    "GenerationConfig",
    "Intervention",
    "LoadedPatch",
    "Manifest",
    "PatchCompatibilityError",
    "PatchFormatError",
    "PatchHandle",
    "PatchRegistry",
    "SUFFIX",
    "StrengthSchedule",
    "__version__",
    "available_backends",
    "default_registry",
    "environment_report",
    "load_patch",
    "save_patch",
]


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module, attribute = target
    return getattr(importlib.import_module(module), attribute)


def __dir__() -> list[str]:
    return sorted(__all__)
