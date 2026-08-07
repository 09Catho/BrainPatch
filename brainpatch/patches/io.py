"""Filesystem I/O for BrainPatch files.

Patches are small JSON documents. They are read and written identically on a
laptop and inside a Modal container, so this module stays on the standard
library.
"""

from __future__ import annotations

import os
from pathlib import Path

from brainpatch.schemas.patch import BrainPatchSpec, PatchValidationError

PATCH_SUFFIX = ".json"


def load_patch(path: str | os.PathLike[str]) -> BrainPatchSpec:
    """Load and validate a single patch file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    PatchValidationError
        If the file is not a well-formed patch.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"patch file not found: {p}")
    try:
        return BrainPatchSpec.from_json(p.read_text(encoding="utf-8"))
    except PatchValidationError as exc:
        raise PatchValidationError(f"{p}: {exc}") from exc


def dump_patch(spec: BrainPatchSpec) -> str:
    """Serialize a patch to its canonical JSON representation."""
    spec.validate()
    return spec.to_json() + "\n"


def save_patch(spec: BrainPatchSpec, path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
    """Write a patch to disk, refusing to clobber unless ``overwrite``."""
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} already exists; pass overwrite=True to replace it")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dump_patch(spec), encoding="utf-8")
    return p


def discover_patches(directory: str | os.PathLike[str]) -> list[Path]:
    """List patch files in ``directory``, sorted by name. Missing dir -> []."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob(f"*{PATCH_SUFFIX}") if p.is_file())


def load_patch_dir(
    directory: str | os.PathLike[str], *, strict: bool = True
) -> tuple[list[BrainPatchSpec], list[tuple[Path, str]]]:
    """Load every patch in a directory.

    Parameters
    ----------
    strict:
        When True (default) the first malformed patch raises. When False,
        failures are collected and returned so a CLI listing can show the
        healthy patches alongside the broken ones.

    Returns
    -------
    (specs, failures)
        ``failures`` is a list of ``(path, message)`` pairs, always empty when
        ``strict`` is True.
    """
    specs: list[BrainPatchSpec] = []
    failures: list[tuple[Path, str]] = []
    for path in discover_patches(directory):
        try:
            specs.append(load_patch(path))
        except (PatchValidationError, OSError) as exc:
            if strict:
                raise
            failures.append((path, str(exc)))
    return specs, failures
