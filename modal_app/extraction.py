"""Modal entry point for activation extraction.

Runs on one L4. The token-budget guard in
:func:`~modal_app.resources.assert_token_budget` is called here, at the entry
point, so that no code path can start a large extraction without it.
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.resources import VOL_MOUNT, app, assert_token_budget, gpu_kwargs, volume

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@app.function(**gpu_kwargs(timeout=60 * 45, secrets=True))
def extract_activations(
    experiment: str = "smoke_v0",
    model_id: str = DEFAULT_MODEL,
    revision: str | None = None,
    layer: int = 18,
    target_tokens: int = 20_000,
    sequence_length: int = 256,
    batch_size: int = 8,
    shard_size: int = 100_000,
    dataset: str = "Salesforce/wikitext",
    dataset_config: str | None = "wikitext-2-raw-v1",
    dataset_split: str = "train",
    seed: int = 0,
    force: bool = False,
    approved: bool = False,
) -> dict[str, Any]:
    """Extract residual-stream activations into immutable shards on the Volume.

    Parameters
    ----------
    force:
        Discard an existing corpus and re-extract. Off by default so an
        accidental re-run resumes instead of paying twice.
    approved:
        Required for runs above the unapproved token ceiling. This is a budget
        control, not a technical one.
    """
    assert_token_budget(target_tokens, approved=approved)

    from brainpatch.research.ml.corpus import CorpusConfig
    from brainpatch.research.ml.extraction import ExtractionConfig, extract_activations as run_extraction
    from brainpatch.research.ml.model import load_model
    from brainpatch.paths import VolumePaths
    from modal_app.image import pinned_versions

    paths = VolumePaths(VOL_MOUNT)
    bundle = load_model(model_id, revision=revision, dtype="bfloat16", device="cuda")
    print(f"[extract] loaded {bundle.model_id} @ {bundle.revision} "
          f"({bundle.num_layers} layers, hidden {bundle.hidden_size})")

    corpus_cfg = CorpusConfig(
        dataset=dataset,
        config=dataset_config,
        split=dataset_split,
        sequence_length=sequence_length,
        seed=seed,
    )
    cfg = ExtractionConfig(
        experiment=experiment,
        layer=layer,
        target_tokens=target_tokens,
        shard_size=shard_size,
        batch_size=batch_size,
        seed=seed,
    )

    provenance = {
        "gpu": "L4",
        "package_versions": pinned_versions(),
        "corpus": corpus_cfg.to_dict(),
        "extraction": cfg.to_dict(),
    }

    result = run_extraction(
        bundle,
        corpus_cfg,
        cfg,
        paths,
        force=force,
        commit=volume.commit,
        provenance=provenance,
    )
    volume.commit()

    payload = result.to_dict()
    payload["model"] = bundle.model_id
    payload["model_revision"] = bundle.revision
    payload["layer"] = result.manifest.layer
    payload["hidden_size"] = result.manifest.hidden_size
    print(json.dumps(payload, indent=2))
    return payload

