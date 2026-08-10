"""The portable BrainPatch artifact: format, loading, validation, registry.

Everything here is importable with **no ML stack installed**. Parsing a patch,
checking its compatibility metadata, installing it, and reporting its size all
work on a bare Python 3.10+.

The one exception is :mod:`brainpatch.patch.compiler`, which reads SAE
checkpoints and therefore needs torch. It is not imported here; import it
explicitly when you mean to compile.
"""

from brainpatch.patch.format import (
    ABSOLUTE_MAX_STRENGTH,
    FORMAT_VERSION,
    SUFFIX,
    BaseModelSpec,
    Intervention,
    Manifest,
    PatchFormatError,
)
from brainpatch.patch.loader import (
    LoadedPatch,
    PatchLoadError,
    load_patch,
    patch_size_report,
    save_patch,
)
from brainpatch.patch.registry import (
    InstalledPatch,
    PatchRegistry,
    RegistryError,
    default_registry,
    registry_home,
)
from brainpatch.patch.validation import (
    CompatibilityReport,
    ModelDescriptor,
    PatchCompatibilityError,
    check_compatibility,
)

__all__ = [
    "ABSOLUTE_MAX_STRENGTH",
    "BaseModelSpec",
    "CompatibilityReport",
    "FORMAT_VERSION",
    "InstalledPatch",
    "Intervention",
    "LoadedPatch",
    "Manifest",
    "ModelDescriptor",
    "PatchCompatibilityError",
    "PatchFormatError",
    "PatchLoadError",
    "PatchRegistry",
    "RegistryError",
    "SUFFIX",
    "check_compatibility",
    "default_registry",
    "load_patch",
    "patch_size_report",
    "registry_home",
    "save_patch",
]
