"""The BrainPatch runtime: backends, capabilities, scheduling, and the model API.

This subpackage is **infrastructure-independent**. It knows how to apply a
vector to a layer of a frozen model; it knows nothing about SAEs, activation
corpora, Modal, or how the patch it is applying came to exist.

That separation is the product: research happens wherever the author has GPUs,
and the resulting artifact runs on the user's own machine with whatever
inference engine they already use.

``brainpatch.runtime.model`` and ``brainpatch.runtime.base`` are importable with
no ML stack present; the engine-specific code under
:mod:`brainpatch.backends` imports torch/vllm/llama.cpp only when instantiated.
"""

from brainpatch.runtime.auto import (
    BackendNotAvailable,
    BackendStatus,
    available_backends,
    backend_class,
    environment_report,
    select_backend,
)
from brainpatch.runtime.base import (
    ActivePatch,
    BrainPatchBackend,
    GenerationConfig,
    ResolvedEdit,
)
from brainpatch.runtime.capabilities import CAPABILITY_FLAGS, Capabilities
from brainpatch.runtime.model import BrainPatchedModel, PatchHandle
from brainpatch.runtime.scheduling import StrengthSchedule

__all__ = [
    "ActivePatch",
    "BackendNotAvailable",
    "BackendStatus",
    "BrainPatchBackend",
    "BrainPatchedModel",
    "CAPABILITY_FLAGS",
    "Capabilities",
    "GenerationConfig",
    "PatchHandle",
    "ResolvedEdit",
    "StrengthSchedule",
    "available_backends",
    "backend_class",
    "environment_report",
    "select_backend",
]
