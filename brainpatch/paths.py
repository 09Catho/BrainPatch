"""Canonical layout of the BrainPatch Modal Volume.

The Volume ``brainpatch-data`` is mounted at ``/vol`` inside Modal containers.
All expensive or large artifacts live there; none of it is ever copied to a
local machine.

::

    /vol/
    |-- hf-cache/                     Hugging Face model + dataset cache
    |-- datasets/                     preprocessed text corpora
    |-- activations/<experiment>/     immutable activation shards + manifest
    |-- sae/<experiment>/             SAE checkpoints + training metrics
    |-- feature-db/<experiment>/      per-feature statistics and contexts
    |-- patches/                      BrainPatch json files
    |-- experiments/<experiment>/     causal-validation artifacts
    `-- reports/                      generated markdown/html reports

This module is deliberately dependency-free so it can be used both locally
(to build CLI messages) and remotely (to actually touch the filesystem).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

#: Mount point of the ``brainpatch-data`` Volume inside Modal containers.
DEFAULT_VOLUME_ROOT = "/vol"

#: Name of the Modal Volume holding all persistent artifacts.
VOLUME_NAME = "brainpatch-data"

#: Name of the Modal Secret exposing ``HF_TOKEN``.
HF_SECRET_NAME = "huggingface-secret"

#: Modal environment this project is developed in.
MODAL_ENVIRONMENT = "brainpatch-dev"


@dataclass(frozen=True)
class VolumePaths:
    """Resolver for every path on the BrainPatch Volume.

    Parameters
    ----------
    root:
        Volume mount point. Defaults to ``/vol``.

    Notes
    -----
    Paths are :class:`~pathlib.PurePosixPath` because the Volume is always
    mounted in a Linux container, even when this code is *constructed* on
    Windows. Converting to :class:`pathlib.Path` only happens on the remote
    side, where the platform is known to be POSIX.
    """

    root: str = DEFAULT_VOLUME_ROOT

    # -- top-level directories ------------------------------------------------

    @property
    def base(self) -> PurePosixPath:
        return PurePosixPath(self.root)

    @property
    def hf_cache(self) -> PurePosixPath:
        return self.base / "hf-cache"

    @property
    def datasets(self) -> PurePosixPath:
        return self.base / "datasets"

    @property
    def activations_root(self) -> PurePosixPath:
        return self.base / "activations"

    @property
    def sae_root(self) -> PurePosixPath:
        return self.base / "sae"

    @property
    def feature_db_root(self) -> PurePosixPath:
        return self.base / "feature-db"

    @property
    def patches(self) -> PurePosixPath:
        return self.base / "patches"

    @property
    def experiments_root(self) -> PurePosixPath:
        return self.base / "experiments"

    @property
    def reports(self) -> PurePosixPath:
        return self.base / "reports"

    def all_top_level(self) -> tuple[PurePosixPath, ...]:
        """Every directory that should exist on a freshly initialised Volume."""
        return (
            self.hf_cache,
            self.datasets,
            self.activations_root,
            self.sae_root,
            self.feature_db_root,
            self.patches,
            self.experiments_root,
            self.reports,
        )

    # -- per-experiment directories -------------------------------------------

    def activations(self, experiment: str) -> PurePosixPath:
        """Directory holding activation shards for ``experiment``."""
        return self.activations_root / experiment

    def activation_manifest(self, experiment: str) -> PurePosixPath:
        return self.activations(experiment) / "manifest.json"

    def activation_examples(self, experiment: str) -> PurePosixPath:
        """JSONL of source examples; token metadata references these by index."""
        return self.activations(experiment) / "examples.jsonl"

    def activation_shard(self, experiment: str, index: int) -> PurePosixPath:
        """Immutable shard path. Shard names are stable and never rewritten."""
        return self.activations(experiment) / shard_filename(index)

    def sae(self, experiment: str) -> PurePosixPath:
        return self.sae_root / experiment

    def sae_checkpoint(self, experiment: str, name: str = "sae_latest.pt") -> PurePosixPath:
        return self.sae(experiment) / name

    def sae_config(self, experiment: str) -> PurePosixPath:
        return self.sae(experiment) / "config.json"

    def sae_metrics(self, experiment: str) -> PurePosixPath:
        return self.sae(experiment) / "metrics.jsonl"

    def feature_db(self, experiment: str) -> PurePosixPath:
        return self.feature_db_root / experiment

    def features_jsonl(self, experiment: str) -> PurePosixPath:
        return self.feature_db(experiment) / "features.jsonl"

    def feature_summary(self, experiment: str) -> PurePosixPath:
        return self.feature_db(experiment) / "summary.json"

    def experiment(self, experiment: str) -> PurePosixPath:
        return self.experiments_root / experiment

    def experiment_file(self, experiment: str, filename: str) -> PurePosixPath:
        return self.experiment(experiment) / filename

    def patch(self, name: str) -> PurePosixPath:
        return self.patches / f"{name}.json"


def shard_filename(index: int) -> str:
    """Return the immutable filename for activation shard ``index``.

    Shard names are zero-padded to six digits so lexical order matches numeric
    order, which lets a streaming reader glob-and-sort without parsing.

    >>> shard_filename(0)
    'shard_000000.safetensors'
    >>> shard_filename(42)
    'shard_000042.safetensors'
    """
    if index < 0:
        raise ValueError(f"shard index must be non-negative, got {index}")
    return f"shard_{index:06d}.safetensors"


def parse_shard_index(filename: str) -> int:
    """Inverse of :func:`shard_filename`.

    >>> parse_shard_index("shard_000042.safetensors")
    42
    """
    stem = filename.rsplit("/", 1)[-1]
    if not stem.startswith("shard_") or not stem.endswith(".safetensors"):
        raise ValueError(f"not a shard filename: {filename!r}")
    return int(stem[len("shard_") : -len(".safetensors")])
