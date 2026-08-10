"""The local patch registry: ``~/.brainpatch``.

Installing a patch means copying a small verified archive into a local store and
recording where it came from. It does **not** mean downloading a base model --
patches are tens of KB and models are gigabytes, and conflating the two is how a
"quick install" becomes a 3 GB surprise.

Layout::

    ~/.brainpatch/
        config.json          registry-level settings
        patches/
            <name>.brainpatch
            <name>.source.json    provenance of the install
        cache/                    downloads, safe to delete

Everything here is plain files. Uninstalling is deleting them; there is no
database to corrupt and nothing to migrate.

Network access happens only in :func:`install_from_hub`, only via
``huggingface_hub``, and only for the patch artifact itself.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brainpatch.patch.format import SUFFIX
from brainpatch.patch.loader import LoadedPatch, load_patch

#: Overridable for tests and for users with an unusual home directory.
ENV_HOME = "BRAINPATCH_HOME"

#: ``owner/repo`` or ``owner/repo:file.brainpatch``
_HUB_REF_RE = re.compile(
    r"^(?P<repo>[A-Za-z0-9][\w.-]*/[\w.-]+)(?::(?P<file>[\w./-]+))?$"
)


class RegistryError(RuntimeError):
    """The registry could not satisfy the request."""


def registry_home() -> Path:
    """Root of the local registry, honouring ``BRAINPATCH_HOME``."""
    override = os.environ.get(ENV_HOME)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".brainpatch"


@dataclass
class InstalledPatch:
    """A patch present in the local registry."""

    name: str
    path: Path
    source: dict[str, Any]

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def load(self) -> LoadedPatch:
        return load_patch(self.path)


class PatchRegistry:
    """File-backed store of installed patches."""

    def __init__(self, home: str | os.PathLike[str] | None = None) -> None:
        self.home = Path(home).expanduser() if home is not None else registry_home()

    # -- paths -----------------------------------------------------------------

    @property
    def patches_dir(self) -> Path:
        return self.home / "patches"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def config_path(self) -> Path:
        return self.home / "config.json"

    def ensure_dirs(self) -> None:
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        return self.patches_dir / f"{name}{SUFFIX}"

    def _source_path(self, name: str) -> Path:
        return self.patches_dir / f"{name}.source.json"

    # -- queries ---------------------------------------------------------------

    def list_patches(self) -> list[InstalledPatch]:
        """Every installed patch, sorted by name. Missing dir -> empty list."""
        if not self.patches_dir.is_dir():
            return []
        out: list[InstalledPatch] = []
        for path in sorted(self.patches_dir.glob(f"*{SUFFIX}")):
            name = path.name[: -len(SUFFIX)]
            out.append(InstalledPatch(name=name, path=path, source=self._read_source(name)))
        return out

    def is_installed(self, name: str) -> bool:
        return self.path_for(name).is_file()

    def get(self, name: str) -> InstalledPatch:
        path = self.path_for(name)
        if not path.is_file():
            available = [p.name for p in self.list_patches()]
            raise RegistryError(
                f"patch {name!r} is not installed. "
                + (f"Installed: {', '.join(available)}" if available else "No patches installed.")
            )
        return InstalledPatch(name=name, path=path, source=self._read_source(name))

    def resolve(self, ref: str) -> Path:
        """Resolve a name, a path, or an installed patch to a file on disk.

        Accepts an installed name first, then a filesystem path -- so a bare
        name never accidentally reads a same-named file from the CWD.
        """
        if self.is_installed(ref):
            return self.path_for(ref)
        candidate = Path(ref).expanduser()
        if candidate.is_file():
            return candidate
        raise RegistryError(
            f"{ref!r} is neither an installed patch nor an existing file. "
            "Use `brainpatch list` to see what is installed."
        )

    def _read_source(self, name: str) -> dict[str, Any]:
        path = self._source_path(name)
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    # -- mutation --------------------------------------------------------------

    def install_file(
        self,
        path: str | os.PathLike[str],
        *,
        name: str | None = None,
        overwrite: bool = False,
        source: dict[str, Any] | None = None,
    ) -> InstalledPatch:
        """Verify then install a local ``.brainpatch`` file.

        The archive is fully loaded and checksum-verified *before* anything is
        written, so a corrupt download cannot land in the registry.
        """
        src = Path(path).expanduser()
        loaded = load_patch(src)  # raises on anything malformed or unsafe

        patch_name = name or loaded.manifest.name
        target = self.path_for(patch_name)
        if target.exists() and not overwrite:
            raise RegistryError(
                f"patch {patch_name!r} is already installed. "
                "Pass --force to replace it."
            )

        self.ensure_dirs()
        shutil.copyfile(src, target)

        record = {
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "origin": str(src),
            "kind": "file",
            "format_version": loaded.manifest.format_version,
            "base_model": loaded.manifest.base_model.model_id,
            "evidence_level": loaded.manifest.evidence_level,
            **(source or {}),
        }
        self._source_path(patch_name).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return InstalledPatch(name=patch_name, path=target, source=record)

    def install_from_hub(
        self,
        ref: str,
        *,
        filename: str | None = None,
        revision: str | None = None,
        overwrite: bool = False,
        offline: bool = False,
    ) -> InstalledPatch:
        """Install from a Hugging Face repo reference.

        ``ref`` is ``owner/repo`` or ``owner/repo:path/to/file.brainpatch``.
        Only the patch artifact is downloaded -- never the base model.
        """
        match = _HUB_REF_RE.match(ref)
        if not match:
            raise RegistryError(
                f"{ref!r} is not a valid Hugging Face reference. "
                "Expected 'owner/repo' or 'owner/repo:file.brainpatch'."
            )
        repo = match.group("repo")
        wanted = filename or match.group("file")

        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise RegistryError(
                "installing from Hugging Face needs the 'huggingface_hub' package.\n"
                "  pip install 'brainpatch[hub]'"
            ) from exc

        if offline:
            raise RegistryError(
                "cannot install from Hugging Face in offline mode; "
                "download the .brainpatch file separately and install it by path"
            )

        self.ensure_dirs()

        if wanted is None:
            try:
                files = list_repo_files(repo, revision=revision)
            except Exception as exc:  # noqa: BLE001 - hub raises many types
                raise RegistryError(f"could not list files in {repo!r}: {exc}") from exc
            candidates = [f for f in files if f.endswith(SUFFIX)]
            if not candidates:
                raise RegistryError(
                    f"{repo!r} contains no {SUFFIX} artifact. "
                    "Specify one explicitly with 'owner/repo:path/to/file.brainpatch'."
                )
            if len(candidates) > 1:
                raise RegistryError(
                    f"{repo!r} contains several patches: {candidates}. "
                    "Choose one with 'owner/repo:<file>'."
                )
            wanted = candidates[0]

        try:
            downloaded = hf_hub_download(
                repo_id=repo,
                filename=wanted,
                revision=revision,
                cache_dir=str(self.cache_dir),
            )
        except Exception as exc:  # noqa: BLE001
            raise RegistryError(f"could not download {wanted!r} from {repo!r}: {exc}") from exc

        return self.install_file(
            downloaded,
            overwrite=overwrite,
            source={"kind": "huggingface", "repo": repo, "file": wanted, "revision": revision},
        )

    def install(self, ref: str, *, overwrite: bool = False, offline: bool = False) -> InstalledPatch:
        """Install from a path or a Hugging Face reference, whichever ``ref`` is."""
        candidate = Path(ref).expanduser()
        if candidate.is_file():
            return self.install_file(candidate, overwrite=overwrite)
        if _HUB_REF_RE.match(ref):
            return self.install_from_hub(ref, overwrite=overwrite, offline=offline)
        raise RegistryError(
            f"{ref!r} is neither an existing file nor an 'owner/repo' Hugging Face reference"
        )

    def uninstall(self, name: str) -> None:
        patch = self.get(name)
        patch.path.unlink()
        source = self._source_path(name)
        if source.exists():
            source.unlink()

    def clear_cache(self) -> int:
        """Delete the download cache. Returns bytes freed."""
        if not self.cache_dir.is_dir():
            return 0
        freed = sum(p.stat().st_size for p in self.cache_dir.rglob("*") if p.is_file())
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return freed


def default_registry() -> PatchRegistry:
    return PatchRegistry()
