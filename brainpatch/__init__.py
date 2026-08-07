"""BrainPatch: experimental activation-space behavioural interventions.

Import policy
-------------
This top-level package is **pure Python**. It must import successfully on a
machine with no ``torch``, ``transformers`` or ``datasets`` installed, because
the local development machine is a source-editing and Modal control-plane
machine only.

Everything that needs the ML stack lives under :mod:`brainpatch.ml` and is
imported lazily -- typically from inside a Modal Function body. Do not add a
top-level ``import torch`` anywhere reachable from here.

The one convenience exception is :class:`BrainPatchedModel`, exposed here via
``__getattr__`` so that ``from brainpatch import BrainPatchedModel`` works in a
GPU environment without making the import mandatory everywhere else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

from brainpatch.paths import VolumePaths
from brainpatch.schemas.contrast import ContrastExample, ContrastSet
from brainpatch.schemas.feature import FeatureRecord, FeatureStats
from brainpatch.schemas.manifest import ActivationManifest, ShardRecord
from brainpatch.schemas.patch import (
    BrainPatchSpec,
    FeatureEdit,
    PatchCompatibilityError,
    PatchValidationError,
    SAEReference,
)
from brainpatch.schemas.sae import SAEConfig
from brainpatch.steering.schedule import StrengthSchedule

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    from brainpatch.ml.runtime import BrainPatchedModel

__all__ = [
    "__version__",
    "ActivationManifest",
    "BrainPatchSpec",
    "BrainPatchedModel",
    "ContrastExample",
    "ContrastSet",
    "FeatureEdit",
    "FeatureRecord",
    "FeatureStats",
    "PatchCompatibilityError",
    "PatchValidationError",
    "SAEConfig",
    "SAEReference",
    "ShardRecord",
    "StrengthSchedule",
    "VolumePaths",
]


def __getattr__(name: str) -> Any:
    """Lazily expose ML-only symbols without importing torch at package import."""
    if name == "BrainPatchedModel":
        from brainpatch.ml.runtime import BrainPatchedModel as _BrainPatchedModel

        return _BrainPatchedModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
