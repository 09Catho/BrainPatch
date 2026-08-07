"""Activation extraction into immutable, resumable shards.

The output of a run is a directory on the Volume::

    /vol/activations/<experiment>/
        manifest.json                 provenance + shard index
        examples.jsonl                one row per token block (text stored once)
        shard_000000.safetensors      activations + token metadata
        shard_000001.safetensors
        ...

Two properties this module is built around:

**Immutability.** A shard, once written and recorded in the manifest, is never
touched again. A run that dies mid-shard leaves an unrecorded partial file that
the next run simply overwrites; everything already in the manifest is safe.

**No string duplication.** Storing the surrounding text next to every one of
hundreds of thousands of activations would multiply the corpus size by an order
of magnitude. Instead each activation row carries
``(example_index, token_position, token_id)`` as int32, and the text lives once
in ``examples.jsonl``. Recovering the context for a high-activating token is a
lookup, not a scan.

The first ``skip_first_n_tokens`` positions of each block are dropped by
default. Position 0 was measured to carry an extreme residual-stream activation
outlier -- norm 11052 against a corpus mean of ~70 at layer 18 of
Qwen2.5-1.5B-Instruct, a factor of 156. Including it distorts the input
normalization and spends dictionary capacity on a single positional artifact.

Outliers at the first token are commonly attributed to attention-sink
behaviour. That is plausible here but unverified: no attention weights were
measured, so the docstring records the outlier, not a mechanism.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from brainpatch.ml.corpus import CorpusConfig, TokenBlock, batched, iter_token_blocks
from brainpatch.ml.hooks import ResidualCapture
from brainpatch.ml.model import ModelBundle, validate_hook, validate_layer
from brainpatch.paths import VolumePaths, shard_filename
from brainpatch.schemas.manifest import ActivationManifest, ShardRecord


@dataclass
class ExtractionConfig:
    """Everything that determines what gets extracted and how."""

    experiment: str
    layer: int = 18
    hook: str = "residual_post"
    target_tokens: int = 20_000
    shard_size: int = 100_000
    batch_size: int = 8
    #: Drop this many leading positions per block (measured position-0 outlier).
    skip_first_n_tokens: int = 1
    #: Storage dtype. bfloat16 is lossless relative to a bf16 forward pass and
    #: halves the corpus size versus float32.
    store_dtype: str = "bfloat16"
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "layer": self.layer,
            "hook": self.hook,
            "target_tokens": self.target_tokens,
            "shard_size": self.shard_size,
            "batch_size": self.batch_size,
            "skip_first_n_tokens": self.skip_first_n_tokens,
            "store_dtype": self.store_dtype,
            "seed": self.seed,
        }


@dataclass
class ExtractionResult:
    """Measured outcome of an extraction run -- the basis for cost estimates."""

    manifest: ActivationManifest
    tokens_written: int
    seconds: float
    tokens_per_second: float
    bytes_per_token: float
    peak_vram_mb: float
    resumed_from: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.manifest.experiment,
            "tokens_written": self.tokens_written,
            "completed_tokens": self.manifest.completed_tokens,
            "num_shards": len(self.manifest.shards),
            "seconds": round(self.seconds, 3),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "bytes_per_token": round(self.bytes_per_token, 2),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "resumed_from_tokens": self.resumed_from,
        }


def load_manifest(paths: VolumePaths, experiment: str) -> ActivationManifest | None:
    """Read an existing manifest, or ``None`` if this is a fresh run."""
    path = Path(paths.activation_manifest(experiment))
    if not path.is_file():
        return None
    manifest = ActivationManifest.from_json(path.read_text(encoding="utf-8"))
    manifest.validate()
    return manifest


def _write_manifest(paths: VolumePaths, manifest: ActivationManifest) -> None:
    manifest.validate()
    path = Path(paths.activation_manifest(manifest.experiment))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")


def _write_shard(
    paths: VolumePaths,
    experiment: str,
    index: int,
    activations: torch.Tensor,
    meta: torch.Tensor,
) -> ShardRecord:
    """Write one immutable shard and return its manifest record."""
    from safetensors.torch import save_file

    path = Path(paths.activation_shard(experiment, index))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"activations": activations.contiguous(), "meta": meta.contiguous()},
        str(path),
        metadata={"format": "brainpatch-activations-v0.1"},
    )
    size = path.stat().st_size
    return ShardRecord(
        index=index,
        filename=shard_filename(index),
        num_tokens=int(activations.shape[0]),
        first_example=int(meta[0, 0].item()),
        last_example=int(meta[-1, 0].item()),
        bytes=size,
    )


def extract_activations(
    bundle: ModelBundle,
    corpus_cfg: CorpusConfig,
    cfg: ExtractionConfig,
    paths: VolumePaths,
    *,
    force: bool = False,
    commit: Callable[[], None] | None = None,
    provenance: dict[str, Any] | None = None,
) -> ExtractionResult:
    """Extract residual-stream activations into sharded storage.

    Parameters
    ----------
    force:
        Discard any existing manifest and start over. Without this, an existing
        run is resumed, which is the safe default for expensive GPU work.
    commit:
        Called after each shard lands, to flush the Modal Volume. Passing
        ``volume.commit`` makes the run durable against container loss.

    Returns
    -------
    ExtractionResult
        Real measurements -- tokens/sec, bytes/token, peak VRAM -- which are
        what later cost projections are built from.
    """
    layer = validate_layer(cfg.layer, bundle.num_layers)
    hook_name = validate_hook(cfg.hook)
    store_dtype = getattr(torch, cfg.store_dtype)

    manifest = None if force else load_manifest(paths, cfg.experiment)
    if manifest is not None:
        _assert_manifest_compatible(manifest, bundle, cfg, corpus_cfg, layer, hook_name)
        if manifest.is_complete:
            print(
                f"[extraction] {cfg.experiment}: already complete "
                f"({manifest.completed_tokens:,} tokens). Pass force=True to redo."
            )
            return ExtractionResult(
                manifest=manifest,
                tokens_written=0,
                seconds=0.0,
                tokens_per_second=0.0,
                bytes_per_token=manifest.bytes_per_token or 0.0,
                peak_vram_mb=0.0,
                resumed_from=manifest.completed_tokens,
            )
    else:
        manifest = ActivationManifest(
            experiment=cfg.experiment,
            model=bundle.model_id,
            model_revision=bundle.revision,
            layer=layer,
            hook=hook_name,
            hidden_size=bundle.hidden_size,
            dtype=cfg.store_dtype,
            dataset=corpus_cfg.dataset_id,
            dataset_split=corpus_cfg.split,
            sequence_length=corpus_cfg.sequence_length,
            requested_tokens=cfg.target_tokens,
            shard_size=cfg.shard_size,
            seed=cfg.seed,
            created_at=_now(),
            provenance=dict(provenance or {}),
        )

    resumed_from = manifest.completed_tokens
    start_example = manifest.num_examples
    remaining = cfg.target_tokens - manifest.completed_tokens

    examples_path = Path(paths.activation_examples(cfg.experiment))
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    if force and examples_path.exists():
        examples_path.unlink()

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    layer_module = bundle.layer_module(layer)
    capture = ResidualCapture(to_cpu=True, dtype=store_dtype)

    act_buffer: list[torch.Tensor] = []
    meta_buffer: list[torch.Tensor] = []
    buffered = 0
    shard_index = manifest.next_shard_index
    tokens_written = 0
    example_index = start_example

    print(
        f"[extraction] {cfg.experiment}: layer {layer} ({hook_name}), "
        f"target {cfg.target_tokens:,} tokens, resuming from {resumed_from:,}"
    )
    start_time = time.perf_counter()

    handle = capture.attach(layer_module)
    try:
        with open(examples_path, "a", encoding="utf-8") as examples_file:
            blocks = iter_token_blocks(bundle.tokenizer, corpus_cfg, start_example=start_example)
            for batch in batched(blocks, cfg.batch_size):
                if tokens_written >= remaining:
                    break

                acts, metas = _forward_batch(
                    bundle, capture, batch, cfg.skip_first_n_tokens, store_dtype
                )
                if acts.shape[0] == 0:
                    continue

                # Never overshoot the requested token count.
                budget = remaining - tokens_written
                if acts.shape[0] > budget:
                    acts = acts[:budget]
                    metas = metas[:budget]

                for block in batch:
                    examples_file.write(
                        json.dumps(
                            {
                                "index": block.example_index,
                                "source_doc": block.source_doc,
                                "char_offset": block.char_offset,
                                "num_tokens": len(block.input_ids),
                                "text": block.text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    example_index = max(example_index, block.example_index + 1)

                act_buffer.append(acts)
                meta_buffer.append(metas)
                buffered += acts.shape[0]
                tokens_written += acts.shape[0]

                if buffered >= cfg.shard_size:
                    examples_file.flush()
                    shard_index, buffered = _flush_shard(
                        paths, cfg, manifest, act_buffer, meta_buffer, shard_index, example_index
                    )
                    _write_manifest(paths, manifest)
                    if commit is not None:
                        commit()

            examples_file.flush()

        if buffered > 0:
            shard_index, buffered = _flush_shard(
                paths, cfg, manifest, act_buffer, meta_buffer, shard_index, example_index
            )
    finally:
        handle.remove()

    elapsed = time.perf_counter() - start_time
    manifest.num_examples = example_index
    manifest.updated_at = _now()
    manifest.provenance.update(provenance or {})
    manifest.provenance["extraction_seconds"] = round(elapsed, 3)
    _write_manifest(paths, manifest)
    if commit is not None:
        commit()

    peak_vram = (
        torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    )
    return ExtractionResult(
        manifest=manifest,
        tokens_written=tokens_written,
        seconds=elapsed,
        tokens_per_second=tokens_written / elapsed if elapsed > 0 else 0.0,
        bytes_per_token=manifest.bytes_per_token or 0.0,
        peak_vram_mb=peak_vram,
        resumed_from=resumed_from,
    )


def _flush_shard(
    paths: VolumePaths,
    cfg: ExtractionConfig,
    manifest: ActivationManifest,
    act_buffer: list[torch.Tensor],
    meta_buffer: list[torch.Tensor],
    shard_index: int,
    example_index: int,
) -> tuple[int, int]:
    """Concatenate the buffer into one shard, record it, clear the buffer."""
    activations = torch.cat(act_buffer, dim=0)
    meta = torch.cat(meta_buffer, dim=0)
    record = _write_shard(paths, cfg.experiment, shard_index, activations, meta)
    manifest.shards.append(record)
    manifest.completed_tokens += record.num_tokens
    manifest.num_examples = example_index
    manifest.updated_at = _now()
    act_buffer.clear()
    meta_buffer.clear()
    print(
        f"[extraction]   shard {shard_index:06d}: {record.num_tokens:,} tokens, "
        f"{record.bytes / 1024**2:.1f} MB (total {manifest.completed_tokens:,})"
    )
    return shard_index + 1, 0


@torch.inference_mode()
def _forward_batch(
    bundle: ModelBundle,
    capture: ResidualCapture,
    batch: list[TokenBlock],
    skip_first_n: int,
    store_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one batch and return (activations, metadata) for kept positions.

    Blocks in a batch may differ in length (the trailing block of a document is
    shorter), so they are right-padded and the padded positions are then
    excluded via the attention mask -- padded activations are meaningless and
    must never enter the corpus.
    """
    lengths = [len(b.input_ids) for b in batch]
    max_len = max(lengths)
    pad_id = bundle.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = bundle.tokenizer.eos_token_id or 0

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, block in enumerate(batch):
        n = len(block.input_ids)
        input_ids[i, :n] = torch.tensor(block.input_ids, dtype=torch.long)
        attention_mask[i, :n] = 1

    input_ids = input_ids.to(bundle.device)
    attention_mask = attention_mask.to(bundle.device)

    capture.activations = None
    bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hidden = capture.activations
    if hidden is None:
        raise RuntimeError("capture hook did not fire -- wrong module or model layout changed")

    keep_acts: list[torch.Tensor] = []
    keep_meta: list[torch.Tensor] = []
    for i, block in enumerate(batch):
        n = lengths[i]
        if n <= skip_first_n:
            continue
        positions = torch.arange(skip_first_n, n, dtype=torch.int32)
        keep_acts.append(hidden[i, skip_first_n:n, :].to(store_dtype))
        meta = torch.stack(
            [
                torch.full((positions.numel(),), block.example_index, dtype=torch.int32),
                positions,
                torch.tensor(block.input_ids[skip_first_n:n], dtype=torch.int32),
            ],
            dim=1,
        )
        keep_meta.append(meta)

    if not keep_acts:
        empty_dtype = store_dtype
        return (
            torch.empty((0, bundle.hidden_size), dtype=empty_dtype),
            torch.empty((0, 3), dtype=torch.int32),
        )
    return torch.cat(keep_acts, dim=0), torch.cat(keep_meta, dim=0)


def _assert_manifest_compatible(
    manifest: ActivationManifest,
    bundle: ModelBundle,
    cfg: ExtractionConfig,
    corpus_cfg: CorpusConfig,
    layer: int,
    hook: str,
) -> None:
    """Refuse to append activations from a different setup to an existing corpus.

    Silently mixing layer-17 and layer-18 activations would produce a corpus
    that trains an SAE on nothing coherent, and the failure would be invisible.
    """
    mismatches: list[str] = []
    if manifest.model != bundle.model_id:
        mismatches.append(f"model: manifest={manifest.model} run={bundle.model_id}")
    if manifest.layer != layer:
        mismatches.append(f"layer: manifest={manifest.layer} run={layer}")
    if manifest.hook != hook:
        mismatches.append(f"hook: manifest={manifest.hook} run={hook}")
    if manifest.hidden_size != bundle.hidden_size:
        mismatches.append(f"hidden_size: manifest={manifest.hidden_size} run={bundle.hidden_size}")
    if manifest.dtype != cfg.store_dtype:
        mismatches.append(f"dtype: manifest={manifest.dtype} run={cfg.store_dtype}")
    if manifest.sequence_length != corpus_cfg.sequence_length:
        mismatches.append(
            f"sequence_length: manifest={manifest.sequence_length} run={corpus_cfg.sequence_length}"
        )
    if manifest.dataset != corpus_cfg.dataset_id:
        mismatches.append(f"dataset: manifest={manifest.dataset} run={corpus_cfg.dataset_id}")
    if mismatches:
        raise ValueError(
            f"cannot resume extraction for {manifest.experiment!r}: configuration changed.\n  "
            + "\n  ".join(mismatches)
            + "\nUse a new experiment name, or pass force=True to discard the existing corpus."
        )


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
