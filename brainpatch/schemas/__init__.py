"""Serializable schemas shared between the local control plane and Modal.

Every schema here is a plain :mod:`dataclasses` structure with explicit
``to_dict`` / ``from_dict`` methods. No pydantic, no torch: these types travel
between a laptop with no ML stack and a GPU container, so they must be
constructible from nothing but the standard library.
"""

from brainpatch.schemas.contrast import ContrastExample, ContrastSet
from brainpatch.schemas.feature import FeatureContext, FeatureRecord, FeatureStats
from brainpatch.schemas.manifest import ActivationManifest, ShardRecord
from brainpatch.schemas.patch import (
    BrainPatchSpec,
    FeatureEdit,
    PatchCompatibilityError,
    PatchValidationError,
    SAEReference,
)
from brainpatch.schemas.sae import SAEConfig

__all__ = [
    "ActivationManifest",
    "BrainPatchSpec",
    "ContrastExample",
    "ContrastSet",
    "FeatureContext",
    "FeatureEdit",
    "FeatureRecord",
    "FeatureStats",
    "PatchCompatibilityError",
    "PatchValidationError",
    "SAEConfig",
    "SAEReference",
    "ShardRecord",
]
