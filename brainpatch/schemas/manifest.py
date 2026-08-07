"""Activation-extraction manifest.

The manifest is the single source of truth describing an activation corpus: what
model produced it, from which hook site, in what dtype, and which shards are
complete. Extraction is *resumable*: the manifest is rewritten after each shard
lands, and a restarted run reads it to decide where to continue.

Shards themselves are immutable. A shard file, once written and recorded in the
manifest, is never modified -- only appended to by new shards. This makes a
partially-failed extraction safe to resume without corrupting earlier work.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

MANIFEST_FORMAT_VERSION = "0.1"


@dataclass
class ShardRecord:
    """One immutable activation shard.

    Attributes
    ----------
    index:
        Zero-based shard number; determines the filename.
    filename:
        Basename on the Volume, e.g. ``shard_000003.safetensors``.
    num_tokens:
        Number of activation rows stored in this shard.
    first_example:
        Index (into ``examples.jsonl``) of the first example contributing rows.
    last_example:
        Index of the last example contributing rows, inclusive.
    sha256:
        Optional content hash, used to detect silent corruption on resume.
    bytes:
        On-disk size, used to compute real bytes-per-token for cost estimates.
    """

    index: int
    filename: str
    num_tokens: int
    first_example: int
    last_example: int
    sha256: str | None = None
    bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShardRecord":
        return cls(
            index=int(data["index"]),
            filename=str(data["filename"]),
            num_tokens=int(data["num_tokens"]),
            first_example=int(data["first_example"]),
            last_example=int(data["last_example"]),
            sha256=data.get("sha256"),
            bytes=data.get("bytes"),
        )


@dataclass
class ActivationManifest:
    """Complete description of one activation corpus.

    Everything needed to (a) resume extraction, (b) stream the corpus into SAE
    training, and (c) later verify that an SAE / patch is being applied to the
    same hook site it was trained on.
    """

    experiment: str
    model: str
    model_revision: str
    layer: int
    hook: str
    hidden_size: int
    dtype: str
    dataset: str
    dataset_split: str
    sequence_length: int
    requested_tokens: int
    completed_tokens: int = 0
    shard_size: int = 100_000
    seed: int = 0
    shards: list[ShardRecord] = field(default_factory=list)
    num_examples: int = 0
    created_at: str | None = None
    updated_at: str | None = None
    format_version: str = MANIFEST_FORMAT_VERSION
    #: Free-form provenance: package versions, GPU, git commit, timings.
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- derived properties ----------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True once at least ``requested_tokens`` activations have been stored."""
        return self.completed_tokens >= self.requested_tokens

    @property
    def next_shard_index(self) -> int:
        """Index the next shard should be written under."""
        return len(self.shards)

    @property
    def bytes_per_token(self) -> float | None:
        """Measured on-disk bytes per activation row, or ``None`` if unknown.

        Used to produce *real* (not guessed) storage estimates for larger runs.
        """
        sized = [s for s in self.shards if s.bytes is not None]
        if not sized:
            return None
        total_bytes = sum(s.bytes for s in sized)  # type: ignore[misc]
        total_tokens = sum(s.num_tokens for s in sized)
        if total_tokens == 0:
            return None
        return total_bytes / total_tokens

    # -- validation ------------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ValueError` if the manifest is internally inconsistent."""
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.layer < 0:
            raise ValueError(f"layer must be non-negative, got {self.layer}")
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")
        if self.shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {self.shard_size}")
        expected = sum(s.num_tokens for s in self.shards)
        if expected != self.completed_tokens:
            raise ValueError(
                "manifest is inconsistent: shards account for "
                f"{expected} tokens but completed_tokens={self.completed_tokens}"
            )
        indices = [s.index for s in self.shards]
        if indices != list(range(len(indices))):
            raise ValueError(f"shard indices must be contiguous from 0, got {indices}")

    # -- serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shards"] = [s.to_dict() for s in self.shards]
        return data

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActivationManifest":
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        kwargs = {k: v for k, v in data.items() if k in known and k != "shards"}
        kwargs["shards"] = [ShardRecord.from_dict(s) for s in data.get("shards", [])]
        return cls(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, text: str) -> "ActivationManifest":
        return cls.from_dict(json.loads(text))
