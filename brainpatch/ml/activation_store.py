"""Streaming reader over activation shards.

SAE training must never require the whole corpus in memory: a serious run is
500k+ activations of 1536 bf16 values, and that is only the smoke-scale
version of where this is going.

:class:`ActivationStream` therefore holds at most one shard plus a bounded
shuffle buffer. Shuffling matters because a shard is written in corpus order,
so consecutive rows come from the same document -- feeding those to an
optimizer in order gives strongly correlated gradients. The reservoir-style
buffer decorrelates them without ever materialising the corpus.

:class:`ActivationSubset` is the small-corpus convenience path: when the whole
thing genuinely fits (as at smoke scale), loading it once avoids re-reading
shards every epoch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from brainpatch.paths import VolumePaths
from brainpatch.schemas.manifest import ActivationManifest


def read_manifest(paths: VolumePaths, experiment: str) -> ActivationManifest:
    """Load and validate an activation manifest, with a clear error if absent."""
    path = Path(paths.activation_manifest(experiment))
    if not path.is_file():
        raise FileNotFoundError(
            f"no activation manifest at {path}. Run extraction for {experiment!r} first."
        )
    manifest = ActivationManifest.from_json(path.read_text(encoding="utf-8"))
    manifest.validate()
    return manifest


def load_shard(
    paths: VolumePaths, experiment: str, index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one shard as ``(activations, meta)``."""
    from safetensors.torch import load_file

    path = Path(paths.activation_shard(experiment, index))
    if not path.is_file():
        raise FileNotFoundError(f"activation shard not found: {path}")
    data = load_file(str(path))
    return data["activations"], data["meta"]


@dataclass
class ActivationStream:
    """Iterate activation rows in bounded memory, with optional shuffling.

    Parameters
    ----------
    shuffle_buffer:
        Rows held back for shuffling. ``0`` disables shuffling entirely, which
        is what validation and analysis want (deterministic corpus order).
    limit:
        Stop after this many rows. Used to carve a validation split off the
        front of the corpus without reading the rest.
    skip:
        Skip this many rows first. Paired with ``limit`` to make disjoint
        train/validation splits.
    """

    paths: VolumePaths
    experiment: str
    manifest: ActivationManifest
    batch_size: int = 512
    shuffle_buffer: int = 8192
    seed: int = 0
    limit: int | None = None
    skip: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    @classmethod
    def open(cls, paths: VolumePaths, experiment: str, **kwargs) -> "ActivationStream":
        return cls(paths=paths, experiment=experiment, manifest=read_manifest(paths, experiment), **kwargs)

    @property
    def total_tokens(self) -> int:
        """Rows this stream will yield, after ``skip`` and ``limit``."""
        available = max(0, self.manifest.completed_tokens - self.skip)
        return min(available, self.limit) if self.limit is not None else available

    def __len__(self) -> int:
        """Number of batches, counting a short final batch."""
        total = self.total_tokens
        return (total + self.batch_size - 1) // self.batch_size

    def iter_rows(self) -> Iterator[torch.Tensor]:
        """Yield individual activation rows in corpus order (no shuffling)."""
        remaining_skip = self.skip
        emitted = 0
        for shard in self.manifest.shards:
            if self.limit is not None and emitted >= self.limit:
                return
            if remaining_skip >= shard.num_tokens:
                remaining_skip -= shard.num_tokens
                continue
            activations, _ = load_shard(self.paths, self.experiment, shard.index)
            start = remaining_skip
            remaining_skip = 0
            for i in range(start, activations.shape[0]):
                if self.limit is not None and emitted >= self.limit:
                    return
                yield activations[i]
                emitted += 1

    def iter_batches(self) -> Iterator[torch.Tensor]:
        """Yield ``[batch_size, hidden]`` float tensors, shuffled if configured.

        Only one shard plus the shuffle buffer is resident at any time.
        """
        rng = random.Random(self.seed)
        buffer: list[torch.Tensor] = []
        batch: list[torch.Tensor] = []

        def emit(rows: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(rows).to(device=self.device, dtype=self.dtype)

        for row in self.iter_rows():
            if self.shuffle_buffer > 0:
                buffer.append(row)
                if len(buffer) < self.shuffle_buffer:
                    continue
                # Swap a random buffered row out, keeping the buffer full.
                idx = rng.randrange(len(buffer))
                buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                row = buffer.pop()

            batch.append(row)
            if len(batch) == self.batch_size:
                yield emit(batch)
                batch = []

        # Drain the shuffle buffer.
        rng.shuffle(buffer)
        for row in buffer:
            batch.append(row)
            if len(batch) == self.batch_size:
                yield emit(batch)
                batch = []
        if batch:
            yield emit(batch)

    def estimate_input_scale(self, sample_rows: int = 8192) -> float:
        """Measure the multiplier that normalizes ``E[||x||_2]`` to ``sqrt(d)``.

        Returns the scalar ``s`` such that ``s * x`` has expected L2 norm
        ``sqrt(hidden_size)``. Storing this alongside the SAE is what lets a
        strength value mean the same thing across SAEs and layers.
        """
        norms: list[float] = []
        for i, row in enumerate(self.iter_rows()):
            if i >= sample_rows:
                break
            norms.append(row.to(torch.float32).norm().item())
        if not norms:
            raise ValueError(f"activation corpus {self.experiment!r} is empty")
        mean_norm = sum(norms) / len(norms)
        if mean_norm == 0:
            raise ValueError("activations have zero mean norm; corpus is degenerate")
        return (self.manifest.hidden_size**0.5) / mean_norm


@dataclass
class ActivationSubset:
    """The whole corpus in memory. Only for corpora that genuinely fit.

    At smoke scale (20k x 1536 bf16 = ~61 MB) this is trivially affordable and
    removes shard I/O from the training loop. :meth:`from_stream` refuses
    anything above ``max_bytes`` so this cannot silently become the path a
    serious run takes.
    """

    activations: torch.Tensor
    meta: torch.Tensor

    @classmethod
    def load(
        cls,
        paths: VolumePaths,
        experiment: str,
        *,
        max_bytes: int = 2 * 1024**3,
        dtype: torch.dtype = torch.float32,
    ) -> "ActivationSubset":
        manifest = read_manifest(paths, experiment)
        estimated = manifest.completed_tokens * manifest.hidden_size * dtype.itemsize
        if estimated > max_bytes:
            raise MemoryError(
                f"corpus {experiment!r} would need {estimated / 1024**3:.1f} GB in {dtype}; "
                f"the in-memory path is capped at {max_bytes / 1024**3:.1f} GB. "
                "Use ActivationStream instead."
            )
        acts: list[torch.Tensor] = []
        metas: list[torch.Tensor] = []
        for shard in manifest.shards:
            a, m = load_shard(paths, experiment, shard.index)
            acts.append(a.to(dtype))
            metas.append(m)
        if not acts:
            raise ValueError(f"corpus {experiment!r} has no shards")
        return cls(activations=torch.cat(acts, dim=0), meta=torch.cat(metas, dim=0))

    def __len__(self) -> int:
        return int(self.activations.shape[0])

    def input_scale(self) -> float:
        """Same normalization measurement as :meth:`ActivationStream.estimate_input_scale`."""
        d = self.activations.shape[1]
        mean_norm = self.activations.norm(dim=1).mean().item()
        if mean_norm == 0:
            raise ValueError("activations have zero mean norm; corpus is degenerate")
        return (d**0.5) / mean_norm

    def split(self, val_fraction: float, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic (train, validation) split by shuffled row index."""
        n = len(self)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        n_val = max(1, int(n * val_fraction)) if val_fraction > 0 else 0
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        return self.activations[train_idx], self.activations[val_idx]
