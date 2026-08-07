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
from modal_app.publish import publish_to_huggingface
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
    "publish_to_huggingface",
    "serve_demo",
    "smoke_pipeline",
    "sweep_strength",
    "top_features",
    "train_sae",
    "verify_cached_model",
    "volume_report",
]


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
