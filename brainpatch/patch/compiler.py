"""Compiling research patches into self-contained runtime artifacts.

This is the bridge between the two halves of the project. A v0.1 research patch
says *"feature 727 of SAE smoke_v0, at strength 16"*; the runtime needs *"this
1536-dimensional vector, at this coefficient"*. Compilation resolves the former
into the latter, once, so that no user ever downloads a 72 MB SAE to apply three
directions.

The arithmetic
--------------
An SAE decoder column lives in normalised space: the SAE was trained on
activations scaled so ``E[||x||] = sqrt(d_in)``. Injecting into the raw residual
stream divides by that ``input_scale``. So a research edit of strength ``s`` on
feature ``f`` compiles to::

    vector = unit(W_dec[:, f]) / input_scale
    coefficient = s

Folding ``1/input_scale`` into the *vector* rather than the coefficient means the
runtime needs no scale metadata at all -- the vector is already in raw residual
units, and a coefficient of 1.0 means exactly what the research patch meant by
strength 1.0.

Several edits on the same layer are emitted as separate interventions rather
than pre-summed, so a user can still see and reason about the individual
directions in ``brainpatch inspect``.

This module needs torch (to read an SAE checkpoint) and therefore lives behind
the ``research`` extra. Nothing in the runtime path imports it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from brainpatch.patch import tensors as ts
from brainpatch.patch.format import (
    BaseModelSpec,
    Intervention,
    Manifest,
    PatchFormatError,
)
from brainpatch.patch.loader import save_patch
from brainpatch.schemas.patch import BrainPatchSpec

#: Storage dtype for compiled vectors. fp16 halves the artifact with no
#: meaningful loss: these are directions whose magnitude is set by a coefficient
#: the user controls at runtime anyway.
DEFAULT_VECTOR_DTYPE = "F16"


class CompileError(RuntimeError):
    """The research patch could not be resolved into runtime vectors."""


def compile_from_sae(
    spec: BrainPatchSpec,
    sae_checkpoint: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    base_model: BaseModelSpec | None = None,
    dtype: str = DEFAULT_VECTOR_DTYPE,
    evidence_level: str | None = None,
    compatibility: dict[str, dict[str, Any]] | None = None,
    readme: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Materialise a v0.1 research patch into a ``.brainpatch`` artifact.

    Parameters
    ----------
    spec:
        The research patch, naming SAE feature IDs and strengths.
    sae_checkpoint:
        Path to the ``sae_latest.pt`` those IDs index into.
    base_model:
        Architecture facts. Defaults are taken from the SAE config, which
        recorded the model it was fitted on.

    Returns
    -------
    Path
        The written artifact.
    """
    import torch

    from brainpatch.research.ml.sae import TopKSAE

    checkpoint_path = Path(sae_checkpoint)
    if not checkpoint_path.is_file():
        raise CompileError(f"SAE checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sae = TopKSAE.from_checkpoint(checkpoint, device="cpu")
    config = sae.config

    if spec.sae.d_sae != sae.d_sae:
        raise CompileError(
            f"patch expects a dictionary of size {spec.sae.d_sae} but the checkpoint "
            f"has {sae.d_sae} -- these are different SAEs and feature IDs are not comparable"
        )
    if spec.sae.d_in != sae.d_in:
        raise CompileError(
            f"patch expects hidden size {spec.sae.d_in} but the SAE was trained on {sae.d_in}"
        )

    input_scale = spec.sae.input_scale or config.input_scale
    if not input_scale:
        raise CompileError(
            "no input_scale is recorded in either the patch or the SAE checkpoint, so "
            "the compiled vector would have no defined magnitude"
        )

    resolved_model = base_model or BaseModelSpec(
        model_id=spec.base_model,
        architecture=_architecture_for(spec.base_model),
        hidden_size=sae.d_in,
        num_layers=_num_layers_for(spec.base_model),
        revision=spec.model_revision or (config.model_revision or None),
    )
    if resolved_model.num_layers <= 0:
        raise CompileError(
            f"could not determine the layer count for {resolved_model.model_id!r}; "
            "pass base_model explicitly"
        )

    vectors: dict[str, ts.Tensor] = {}
    interventions: list[Intervention] = []

    for edit in spec.features:
        if edit.mode != "add":
            raise CompileError(
                f"cannot compile intervention mode {edit.mode!r}: ablation depends on the "
                "SAE encoder at runtime and has no fixed-vector representation"
            )
        direction = sae.feature_direction(edit.feature_id, normalize=True)
        # Fold the scale into the vector so the runtime carries no SAE metadata.
        raw = (direction / float(input_scale)).to(torch.float32).tolist()

        key = f"f{edit.feature_id}"
        vectors[key] = ts.vector(raw, dtype=dtype)
        interventions.append(
            Intervention(
                layer=spec.sae.layer,
                vector=key,
                coefficient=float(edit.strength),
                hook=spec.sae.hook or "residual_post",
                id=f"sae-feature-{edit.feature_id}",
            )
        )

    manifest = Manifest(
        name=spec.name,
        base_model=resolved_model,
        interventions=interventions,
        description=spec.description,
        evidence_level=evidence_level or spec.evidence_level,
        evaluation=dict(spec.evaluation),
        compatibility=compatibility or {},
        provenance={
            "method": "sparse_autoencoder_decoder_direction",
            "sae_reference": spec.sae.reference,
            "sae_d_sae": spec.sae.d_sae,
            "sae_layer": spec.sae.layer,
            "sae_input_scale": float(input_scale),
            "source_format_version": spec.format_version,
            "source_feature_ids": [e.feature_id for e in spec.features],
            "compiled_by": "brainpatch.patch.compiler",
            "note": (
                "Vectors are unit decoder columns divided by input_scale, so they are "
                "already in raw residual-stream units. The runtime needs no SAE."
            ),
            **dict(spec.metadata),
        },
        max_abs_strength=max(8.0, max(abs(e.strength) for e in spec.features) * 2),
        default_strength=1.0,
        license=spec.license,
        authors=list(spec.authors),
        schedule=spec.schedule,
    )
    manifest.validate()

    return save_patch(manifest, vectors, output, readme=readme, overwrite=overwrite)


def compile_from_vectors(
    name: str,
    vectors: dict[str, list[float]],
    interventions: list[Intervention],
    base_model: BaseModelSpec,
    output: str | os.PathLike[str],
    *,
    dtype: str = DEFAULT_VECTOR_DTYPE,
    description: str = "",
    evidence_level: str = "none",
    provenance: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    compatibility: dict[str, dict[str, Any]] | None = None,
    readme: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Build an artifact from raw vectors, whatever method produced them.

    The runtime does not care whether a direction came from an SAE, a difference
    of means, PCA, or a learned controller -- this is the entry point for every
    method that is not the SAE path.
    """
    tensor_map = {key: ts.vector(values, dtype=dtype) for key, values in vectors.items()}
    manifest = Manifest(
        name=name,
        base_model=base_model,
        interventions=interventions,
        description=description,
        evidence_level=evidence_level,  # type: ignore[arg-type]
        evaluation=dict(evaluation or {}),
        compatibility=dict(compatibility or {}),
        provenance=dict(provenance or {}),
    )
    manifest.validate()
    return save_patch(manifest, tensor_map, output, readme=readme, overwrite=overwrite)


def _architecture_for(model_id: str) -> str:
    """Read the architecture string from a model config, if reachable."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)
        archs = getattr(config, "architectures", None) or []
        return archs[0] if archs else config.model_type
    except Exception:  # noqa: BLE001 - offline compile must still work
        return ""


def _num_layers_for(model_id: str) -> int:
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)
        return int(getattr(config, "num_hidden_layers", 0))
    except Exception:  # noqa: BLE001
        return 0


def export_llamacpp_control_vector(
    patch_path: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    strength: float = 1.0,
) -> Path:
    """Export a compiled patch as a llama.cpp control-vector GGUF.

    llama.cpp's control-vector format is a GGUF holding one tensor per layer,
    named ``direction.<layer>``, where layer indices are **1-based** -- unlike
    BrainPatch's 0-based decoder-block indexing. Getting that mapping wrong
    silently steers the wrong block, so it is done explicitly here and asserted
    in the integration test.

    Requires ``gguf``, which ships with llama.cpp's Python tooling.
    """
    from brainpatch.patch.loader import load_patch

    try:
        import gguf
    except ModuleNotFoundError as exc:  # pragma: no cover - optional tooling
        raise CompileError(
            "exporting a llama.cpp control vector needs the 'gguf' package.\n"
            "  pip install gguf"
        ) from exc

    import numpy as np

    loaded = load_patch(patch_path)
    hidden = loaded.manifest.base_model.hidden_size

    # Sum every intervention that targets the same layer: llama.cpp applies one
    # direction per layer, so multi-vector layers must be combined at export.
    per_layer: dict[int, list[float]] = {}
    for intervention in loaded.manifest.interventions:
        vector = loaded.vector_for(intervention.vector)
        scaled = [v * intervention.coefficient * strength for v in vector.data]
        acc = per_layer.setdefault(intervention.layer, [0.0] * hidden)
        for i, value in enumerate(scaled):
            acc[i] += value

    out = Path(output)
    if out.suffix != ".gguf":
        out = out.with_suffix(".gguf")
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(str(out), arch="controlvector")
    writer.add_string("controlvector.model_hint", _model_hint(loaded.manifest.base_model.architecture))
    writer.add_uint32("controlvector.layer_count", len(per_layer))
    for layer, values in sorted(per_layer.items()):
        # 0-based BrainPatch layer -> 1-based llama.cpp direction index.
        writer.add_tensor(f"direction.{layer + 1}", np.array(values, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return out


def _model_hint(architecture: str) -> str:
    """llama.cpp's short architecture hint for control vectors."""
    lowered = (architecture or "").lower()
    for needle, hint in (
        ("qwen2", "qwen2"),
        ("qwen", "qwen2"),
        ("llama", "llama"),
        ("mistral", "llama"),
        ("gemma", "gemma"),
        ("phi", "phi2"),
    ):
        if needle in lowered:
            return hint
    return "unknown"
