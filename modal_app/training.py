"""Modal entry point for SAE training.

Trains on one L4. The activation corpus is read from the Volume; checkpoints go
back to the Volume after every ``checkpoint_every`` steps, so a container loss
costs at most that many steps rather than the whole run.
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.resources import VOL_MOUNT, app, gpu_kwargs, volume


@app.function(**gpu_kwargs(timeout=60 * 45))
def train_sae(
    experiment: str = "smoke_v0",
    d_sae: int = 2048,
    k: int = 32,
    lr: float = 3e-4,
    batch_size: int = 512,
    epochs: int = 20,
    max_steps: int | None = None,
    auxk_alpha: float = 1.0 / 32.0,
    # Exposed because SAEConfig requires auxk_k <= d_sae: a smaller dictionary
    # cannot use the 256 default, and serious_v1.yaml specifies 512.
    auxk_k: int = 256,
    dead_feature_window: int = 20_000,
    val_fraction: float = 0.05,
    seed: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    """Train a Top-K SAE on a previously extracted activation corpus."""
    from brainpatch.ml.activation_store import ActivationSubset, read_manifest
    from brainpatch.ml.training import train_sae as run_training
    from brainpatch.paths import VolumePaths
    from brainpatch.schemas.sae import SAEConfig
    from modal_app.image import pinned_versions

    paths = VolumePaths(VOL_MOUNT)
    manifest = read_manifest(paths, experiment)
    print(
        f"[train] corpus {experiment}: {manifest.completed_tokens:,} tokens, "
        f"layer {manifest.layer}, hidden {manifest.hidden_size}, "
        f"{len(manifest.shards)} shard(s)"
    )

    subset = ActivationSubset.load(paths, experiment, dtype=__import__("torch").float32)
    config = SAEConfig(
        d_in=manifest.hidden_size,
        d_sae=d_sae,
        k=k,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        max_steps=max_steps,
        auxk_alpha=auxk_alpha,
        auxk_k=auxk_k,
        dead_feature_window=dead_feature_window,
        val_fraction=val_fraction,
        seed=seed,
        model=manifest.model,
        model_revision=manifest.model_revision,
        layer=manifest.layer,
        hook=manifest.hook,
        notes={"package_versions": pinned_versions(), "corpus_tokens": manifest.completed_tokens},
    )

    result = run_training(
        config,
        subset,
        paths,
        experiment,
        device="cuda",
        force=force,
        commit=volume.commit,
        provenance={"gpu": "L4", "package_versions": pinned_versions()},
    )
    volume.commit()

    payload = result.to_dict()
    payload["corpus_tokens"] = manifest.completed_tokens
    payload["expansion_factor"] = config.expansion_factor
    payload["num_parameters"] = config.num_parameters
    print(json.dumps(payload, indent=2, default=str))
    return payload

