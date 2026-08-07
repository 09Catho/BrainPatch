"""Single entry point exposing every BrainPatch Modal Function.

Every workflow is reachable as::

    modal run modal_app/app.py::<function>

The individual modules also work as entry points (``modal run
modal_app/extraction.py::extract_activations``); this module just gathers them
so there is one place to look.

``smoke_pipeline`` is the full end-to-end run. It is a ``local_entrypoint``
rather than a remote Function so that each stage gets its own container with the
right resources -- GPU where needed, CPU where not -- instead of holding an L4
open through the CPU-bound analysis phase.
"""

from __future__ import annotations

import json

from modal_app.analysis import analyze_features, top_features, volume_report
from modal_app.extraction import extract_activations
from modal_app.gpu_info import cpu_smoke, gpu_info
from modal_app.intervention import (
    dynamic_steering_demo,
    intervention_experiment,
    intervention_smoke,
    sweep_strength,
)
from modal_app.model_cache import cache_model, model_architecture, verify_cached_model
from modal_app.publish import (
    publish_dataset_to_huggingface,
    publish_to_huggingface,
    sync_patches,
)
from modal_app.resources import app
from modal_app.training import train_sae
from modal_app.web import serve_demo

__all__ = [
    "analyze_features",
    "app",
    "cache_model",
    "cpu_smoke",
    "dynamic_steering_demo",
    "extract_activations",
    "gpu_info",
    "intervention_experiment",
    "intervention_smoke",
    "model_architecture",
    "publish_dataset_to_huggingface",
    "publish_release",
    "publish_to_huggingface",
    "serve_demo",
    "sync_patches",
    "smoke_pipeline",
    "sweep_strength",
    "top_features",
    "train_sae",
    "verify_cached_model",
    "volume_report",
]


@app.local_entrypoint()
def publish_release(
    experiment: str = "smoke_v0",
    intervention_run: str = "smoke_v0_intervention",
    repo_id: str = "",
    dataset_repo_id: str = "",
    dry_run: bool = True,
    with_dataset: bool = True,
) -> None:
    """Publish curated artifacts from the Volume to Hugging Face.

    This local entrypoint reads only *small text files* from the working tree --
    the patch JSONs and the model card, a few kilobytes total -- and hands them
    to Modal. The actual upload runs inside a Modal container and reads the
    heavy artifacts straight from ``/vol``::

        Modal Volume -> Hugging Face Hub          (what happens)
        Modal Volume -> local PC -> Hugging Face  (never)

    Defaults to ``dry_run=True``. Publishing is outward-facing and effectively
    irreversible, so it takes an explicit ``--no-dry-run``.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent

    patches = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted((repo_root / "patches").glob("*.json"))
    }
    if not patches:
        raise SystemExit("no patch files found in patches/")
    print(f"[publish] staging {len(patches)} patch file(s) onto the Volume")
    sync_patches.remote(patches)

    card_path = repo_root / "docs" / "model_card.md"
    if not card_path.is_file():
        raise SystemExit(f"model card not found at {card_path}")
    model_card = card_path.read_text(encoding="utf-8")

    print(f"[publish] model repository (dry_run={dry_run})")
    model_result = publish_to_huggingface.remote(
        experiment=experiment,
        repo_id=repo_id or None,
        intervention_run=intervention_run,
        dry_run=dry_run,
        model_card=model_card,
    )

    dataset_result = None
    if with_dataset:
        card = _dataset_card(experiment)
        print(f"[publish] dataset repository (dry_run={dry_run})")
        dataset_result = publish_dataset_to_huggingface.remote(
            experiment=experiment,
            repo_id=dataset_repo_id or None,
            dry_run=dry_run,
            dataset_card=card,
        )

    print("\n===== publication summary =====")
    print(json.dumps({"model": model_result, "dataset": dataset_result}, indent=2, default=str))


def _dataset_card(experiment: str) -> str:
    """Card for the feature-database dataset repository."""
    return f"""---
license: apache-2.0
tags:
  - mechanistic-interpretability
  - sparse-autoencoder
  - interpretability
---

# BrainPatch feature database — Qwen2.5-1.5B-Instruct

Per-feature statistics for a Top-K sparse autoencoder trained on layer 18
(`residual_post`) of a frozen `Qwen/Qwen2.5-1.5B-Instruct`
(revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`).

GitHub: **https://github.com/09Catho/BrainPatch**

## Contents

| File | What |
|---|---|
| `{experiment}/features.jsonl` | One record per feature: statistics and top-activating contexts |
| `{experiment}/summary.json` | Aggregate dictionary statistics |
| `{experiment}/activation_manifest.json` | Corpus provenance (metadata only) |

## Record schema

```json
{{
  "feature_id": 727,
  "stats": {{
    "fire_count": 0, "total_tokens": 20000,
    "mean_activation": 0.0, "max_activation": 0.0,
    "std_activation": 0.0, "decoder_norm": 1.0,
    "firing_rate": 0.0, "is_dead": false
  }},
  "top_contexts": [
    {{"example_index": 0, "token_position": 0, "token_id": 0,
      "token_text": "", "activation": 0.0,
      "context_before": "", "context_after": ""}}
  ],
  "hypothesis": null,
  "evidence_level": "none",
  "evidence_refs": []
}}
```

`mean_activation` and `std_activation` are computed over **firing tokens only**.
Averaging in a Top-K SAE's structural zeros would report `k/d_sae` times the
true magnitude.

## Dictionary statistics

| | |
|---|---|
| Features | 2048 (d_in 1536, expansion 1.33×) |
| Top-K | 32 (measured L0 exactly 32.0) |
| Tokens analysed | 20,000 |
| Alive / dead | 2048 / 0 |
| Mean firing rate (alive) | 0.015625 |
| Median firing rate (alive) | 0.01355 |
| Max firing rate | 0.0766 |
| Decoder norms | mean 1.0, min 0.9999992, max 1.0000007 |
| `input_scale` | 0.5610531069008018 |

## Scientific status

**Every `hypothesis` field is `null` and every `evidence_level` is `none`.**

Top-activating contexts are *correlational* evidence. A feature whose top
examples look thematically coherent is a feature that correlates with that theme
in this corpus — it is not "the X feature" until steering it changes behaviour
and scale-matched controls do not.

In the accompanying intervention experiment the controls **failed**: a
scale-matched random direction moved the model's output further from baseline
(0.847) than the tested feature direction did (0.710). Treat these statistics as
descriptive only.

The SAE is also deliberately undertrained (20k activations, train explained
variance 0.762 vs validation 0.658). See the research log on GitHub.

## Corpus and licensing

Statistics are derived from
[`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext)
(`wikitext-2-raw-v1`, train split), CC BY-SA 3.0, derived from Wikipedia.

The corpus is **not** redistributed here. `top_contexts` carry short attributed
snippets (a few tokens either side of the activating token) for human
inspection; everything else is derived numerical metadata.

This repository: Apache-2.0.
"""


@app.local_entrypoint()
def smoke_pipeline(
    experiment: str = "smoke_v0",
    layer: int = 18,
    target_tokens: int = 20_000,
    d_sae: int = 2048,
    k: int = 32,
    epochs: int = 20,
    strength: float = 4.0,
    force: bool = False,
) -> None:
    """Run the full smoke_v0 pipeline, stage by stage.

    ::

        Modal -> Qwen -> hook -> shards -> SAE -> features -> contexts
              -> intervention -> changed generation -> controls

    Each stage writes to the Volume and is independently resumable, so a
    failure part-way does not throw away the stages that succeeded.
    """
    print("\n===== stage 1/7: environment (CPU) =====")
    cpu_smoke.remote()

    print("\n===== stage 2/7: model cache (CPU) =====")
    cached = cache_model.remote()
    revision = cached["revision"]

    print("\n===== stage 3/7: architecture check (CPU) =====")
    arch = model_architecture.remote()
    if layer >= arch["num_hidden_layers"]:
        raise SystemExit(
            f"configured layer {layer} does not exist: model has "
            f"{arch['num_hidden_layers']} layers"
        )
    print(f"layer {layer} is valid (model has {arch['num_hidden_layers']} layers)")

    print("\n===== stage 4/7: activation extraction (L4) =====")
    extraction = extract_activations.remote(
        experiment=experiment,
        revision=revision,
        layer=layer,
        target_tokens=target_tokens,
        force=force,
    )

    print("\n===== stage 5/7: SAE training (L4) =====")
    training = train_sae.remote(
        experiment=experiment, d_sae=d_sae, k=k, epochs=epochs, force=force
    )

    print("\n===== stage 6/7: feature analysis (CPU) =====")
    features = analyze_features.remote(experiment=experiment)

    print("\n===== stage 7/7: intervention + controls (L4) =====")
    smoke = intervention_smoke.remote(experiment=experiment)
    if not smoke["zero_strength_identical_to_baseline"]:
        raise SystemExit(
            "ABORT: strength=0 did not reproduce baseline. The hook machinery is "
            "wrong and every downstream measurement would be invalid."
        )
    causal = intervention_experiment.remote(
        experiment=experiment, run_name=f"{experiment}_intervention", strength=strength
    )

    print("\n===== summary =====")
    print(
        json.dumps(
            {
                "extraction": extraction,
                "training": training,
                "features": {
                    key: features[key]
                    for key in (
                        "alive_features",
                        "dead_features",
                        "dead_fraction",
                        "mean_l0",
                        "num_tokens_analysed",
                    )
                },
                "intervention_smoke": {
                    key: smoke[key]
                    for key in (
                        "feature_id",
                        "zero_strength_identical_to_baseline",
                        "steered_differs_from_baseline",
                        "delta_norm_matches",
                    )
                },
                "causal_summary": causal["summary"]["effect_vs_controls"],
                "artifacts": causal["artifacts"],
            },
            indent=2,
            default=str,
        )
    )
