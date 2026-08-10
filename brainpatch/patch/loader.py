"""Reading and writing ``.brainpatch`` archives.

Security posture
----------------
A ``.brainpatch`` file is **untrusted input**. It typically arrives from the
internet, and the whole reason the format is ZIP-of-JSON-and-safetensors rather
than a pickle is that applying one must never be able to run code.

So this loader:

* reads members **by exact name**, never by iterating and trusting what it finds
* rejects absolute paths, ``..`` traversal, symlinks and directory entries
* enforces per-member and total size ceilings before decompressing (zip bombs)
* verifies sha256 for every member against ``checksums.json``
* parses only JSON and safetensors, both of which are inert

Nothing here evaluates, imports, or executes any part of an artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brainpatch.patch import tensors as ts
from brainpatch.patch.format import (
    CHECKSUMS_NAME,
    MANIFEST_NAME,
    README_NAME,
    SUFFIX,
    VECTORS_NAME,
    Manifest,
    PatchFormatError,
)

#: Members the loader will read. Anything else in the archive is ignored, and
#: an *unexpected* member is reported rather than silently accepted.
KNOWN_MEMBERS = (MANIFEST_NAME, VECTORS_NAME, CHECKSUMS_NAME, README_NAME)

#: A patch is meant to be tiny. These ceilings are generous by two orders of
#: magnitude and exist purely to bound a hostile archive.
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200


class PatchLoadError(PatchFormatError):
    """The archive could not be read safely."""


@dataclass
class LoadedPatch:
    """A parsed, checksum-verified patch artifact."""

    manifest: Manifest
    vectors: dict[str, ts.Tensor]
    readme: str | None = None
    source: str | None = None
    #: Size of the archive on disk, for honest size reporting.
    archive_bytes: int = 0

    @property
    def name(self) -> str:
        return self.manifest.name

    def vector_for(self, key: str) -> ts.Tensor:
        if key not in self.vectors:
            raise PatchLoadError(
                f"patch {self.name!r} references vector {key!r}, which is not in "
                f"{VECTORS_NAME} (present: {sorted(self.vectors)})"
            )
        return self.vectors[key]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.manifest.name,
            "format_version": self.manifest.format_version,
            "description": self.manifest.description,
            "base_model": self.manifest.base_model.to_dict(),
            "num_interventions": len(self.manifest.interventions),
            "layers": self.manifest.layers,
            "num_vectors": len(self.vectors),
            "vector_dtype": next(iter(self.vectors.values())).dtype if self.vectors else None,
            "evidence_level": self.manifest.evidence_level,
            "compatibility": self.manifest.compatibility,
            "archive_bytes": self.archive_bytes,
            "source": self.source,
        }


def _safe_member_names(zf: zipfile.ZipFile) -> list[str]:
    """Validate every entry's name and size before reading anything."""
    total = 0
    names: list[str] = []
    for info in zf.infolist():
        name = info.filename

        if info.is_dir():
            continue

        # Path safety first: these are the checks that actually matter, and
        # they give the clearest diagnostics.
        if name.startswith("/") or name.startswith("\\"):
            raise PatchLoadError(f"archive member {name!r} uses an absolute path")
        if ".." in Path(name.replace("\\", "/")).parts:
            raise PatchLoadError(f"archive member {name!r} escapes the archive root")
        if os.path.isabs(name) or (len(name) > 1 and name[1] == ":"):
            raise PatchLoadError(f"archive member {name!r} uses an absolute path")

        # Unix mode lives in the high 16 bits of external_attr, but the type
        # bits are frequently absent -- CPython's own ``writestr`` stores plain
        # permissions (0o600) with no S_IFREG. So reject only what is positively
        # identified as a symlink or other special file; treat "no type bits" as
        # an ordinary member rather than as suspicious.
        mode = info.external_attr >> 16
        if mode and stat.S_IFMT(mode):
            if stat.S_ISLNK(mode):
                raise PatchLoadError(f"archive member {name!r} is a symlink")
            if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
                raise PatchLoadError(f"archive member {name!r} is not a regular file")

        if info.file_size > MAX_MEMBER_BYTES:
            raise PatchLoadError(
                f"archive member {name!r} is {info.file_size} bytes, over the "
                f"{MAX_MEMBER_BYTES} limit"
            )
        if info.compress_size > 0:
            ratio = info.file_size / info.compress_size
            if ratio > MAX_COMPRESSION_RATIO:
                raise PatchLoadError(
                    f"archive member {name!r} has a {ratio:.0f}x compression ratio, "
                    "which looks like a zip bomb"
                )
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise PatchLoadError(f"archive expands to over {MAX_TOTAL_BYTES} bytes")

        names.append(name)
    return names


def load_patch(path: str | os.PathLike[str], *, verify_checksums: bool = True) -> LoadedPatch:
    """Load and verify a ``.brainpatch`` archive.

    Parameters
    ----------
    verify_checksums:
        Leave this True. It is a parameter only so the writer can round-trip a
        freshly-built archive before its checksums exist.

    Raises
    ------
    PatchLoadError
        On a malformed, unsafe, or checksum-mismatched archive.
    """
    p = Path(path)
    if not p.is_file():
        raise PatchLoadError(f"patch file not found: {p}")

    archive_bytes = p.stat().st_size
    try:
        with zipfile.ZipFile(p) as zf:
            present = set(_safe_member_names(zf))

            unexpected = present - set(KNOWN_MEMBERS)
            if unexpected:
                raise PatchLoadError(
                    f"archive contains unexpected members: {sorted(unexpected)}. "
                    f"A BrainPatch may only contain {list(KNOWN_MEMBERS)}."
                )
            for required in (MANIFEST_NAME, VECTORS_NAME):
                if required not in present:
                    raise PatchLoadError(f"archive is missing required member {required!r}")

            raw = {name: zf.read(name) for name in present}
    except zipfile.BadZipFile as exc:
        raise PatchLoadError(f"{p} is not a valid archive: {exc}") from exc

    if verify_checksums:
        if CHECKSUMS_NAME not in raw:
            raise PatchLoadError(f"archive is missing {CHECKSUMS_NAME!r}")
        _verify_checksums(raw)

    manifest = Manifest.from_json(raw[MANIFEST_NAME].decode("utf-8"))

    try:
        vectors, _ = ts.load(raw[VECTORS_NAME])
    except ts.SafetensorsError as exc:
        raise PatchLoadError(f"{VECTORS_NAME} is malformed: {exc}") from exc

    # Every referenced vector must exist and be a 1-D tensor of hidden_size.
    hidden = manifest.base_model.hidden_size
    for key in manifest.vector_keys:
        if key not in vectors:
            raise PatchLoadError(
                f"manifest references vector {key!r} which is absent from {VECTORS_NAME}"
            )
        tensor = vectors[key]
        if len(tensor.shape) != 1:
            raise PatchLoadError(
                f"vector {key!r} must be 1-D, got shape {tensor.shape}"
            )
        if tensor.shape[0] != hidden:
            raise PatchLoadError(
                f"vector {key!r} has length {tensor.shape[0]} but the patch declares "
                f"hidden_size {hidden}"
            )

    readme = raw.get(README_NAME)
    return LoadedPatch(
        manifest=manifest,
        vectors=vectors,
        readme=readme.decode("utf-8") if readme else None,
        source=str(p),
        archive_bytes=archive_bytes,
    )


def _verify_checksums(raw: dict[str, bytes]) -> None:
    try:
        recorded = json.loads(raw[CHECKSUMS_NAME].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PatchLoadError(f"{CHECKSUMS_NAME} is not valid JSON: {exc}") from exc
    if not isinstance(recorded, dict):
        raise PatchLoadError(f"{CHECKSUMS_NAME} must be a JSON object")

    for name, blob in raw.items():
        if name == CHECKSUMS_NAME:
            continue
        expected = recorded.get(name)
        if expected is None:
            raise PatchLoadError(f"{CHECKSUMS_NAME} has no entry for member {name!r}")
        actual = hashlib.sha256(blob).hexdigest()
        if actual != expected:
            raise PatchLoadError(
                f"checksum mismatch for {name!r}: manifest records {expected}, "
                f"archive contains {actual}. The patch is corrupt or was modified."
            )


def save_patch(
    manifest: Manifest,
    vectors: dict[str, ts.Tensor],
    path: str | os.PathLike[str],
    *,
    readme: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a ``.brainpatch`` archive, with checksums, deterministically.

    ZIP entries are written with a fixed timestamp and in fixed order, so two
    builds from identical inputs produce byte-identical archives -- which is
    what makes a published patch hash verifiable.
    """
    manifest.validate()

    p = Path(path)
    if p.suffix != SUFFIX:
        p = p.with_suffix(SUFFIX)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} already exists; pass overwrite=True to replace it")

    missing = [k for k in manifest.vector_keys if k not in vectors]
    if missing:
        raise PatchFormatError(f"manifest references vectors not provided: {missing}")

    hidden = manifest.base_model.hidden_size
    for key, tensor in vectors.items():
        if len(tensor.shape) != 1 or tensor.shape[0] != hidden:
            raise PatchFormatError(
                f"vector {key!r} must be 1-D of length {hidden}, got shape {tensor.shape}"
            )

    members: dict[str, bytes] = {
        MANIFEST_NAME: (manifest.to_json() + "\n").encode("utf-8"),
        VECTORS_NAME: ts.dump(vectors, metadata={"format": "brainpatch-vectors-v1"}),
    }
    if readme:
        members[README_NAME] = readme.encode("utf-8")

    checksums = {name: hashlib.sha256(blob).hexdigest() for name, blob in sorted(members.items())}
    members[CHECKSUMS_NAME] = (
        json.dumps(checksums, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16  # regular file, rw-r--r--
            zf.writestr(info, members[name])
    return p


def patch_size_report(loaded: LoadedPatch) -> dict[str, Any]:
    """Size breakdown, for honest reporting rather than marketing claims."""
    vector_bytes = sum(t.nbytes for t in loaded.vectors.values())
    return {
        "archive_bytes": loaded.archive_bytes,
        "archive_kb": round(loaded.archive_bytes / 1024, 2),
        "num_vectors": len(loaded.vectors),
        "vector_payload_bytes": vector_bytes,
        "hidden_size": loaded.manifest.base_model.hidden_size,
        "dtype": next(iter(loaded.vectors.values())).dtype if loaded.vectors else None,
    }
