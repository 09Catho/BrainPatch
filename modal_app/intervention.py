"""Modal entry points for interventions and causal validation.

Three functions, in increasing cost:

``intervention_smoke``
    Verifies the hook machinery: that ``strength=0`` reproduces baseline
    exactly, that a non-zero strength changes the output, and that the injected
    delta has the expected norm. Cheap and run first.

``intervention_experiment``
    The full conditioned comparison -- baseline, positive, negative, plus a
    scale-matched random direction and an unrelated real feature -- over a set
    of prompts, with all generations persisted.

``dynamic_steering_demo``
    Demonstrates changing patch strength partway through a single generation.
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.resources import VOL_MOUNT, app, gpu_kwargs, volume


def _load_patched_model(experiment: str, device: str = "cuda"):
    """Load the base model with the experiment's SAE attached."""
    from brainpatch.ml.runtime import BrainPatchedModel
    from brainpatch.paths import VolumePaths

    paths = VolumePaths(VOL_MOUNT)
    checkpoint = str(paths.sae_checkpoint(experiment))

    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]

    model = BrainPatchedModel.from_pretrained(
        config["model"], revision=config.get("model_revision") or None, device=device
    )
    model.load_sae(checkpoint, reference=experiment)
    return model, paths


@app.function(**gpu_kwargs(timeout=60 * 25))
def intervention_smoke(
    experiment: str = "smoke_v0",
    feature_id: int | None = None,
    strength: float = 2.0,
    prompt: str = "Explain in two sentences why the sky appears blue.",
) -> dict[str, Any]:
    """Verify the intervention machinery end to end, cheaply.

    The critical assertion is that installing a patch at strength 0 produces
    output *byte-identical* to running with no patch installed. If that fails,
    every baseline measured anywhere in this project is contaminated.
    """
    from brainpatch.ml.feature_analysis import rank_features
    from brainpatch.ml.generation import GenerationConfig

    model, paths = _load_patched_model(experiment)
    assert model.sae is not None

    if feature_id is None:
        ranked = rank_features(paths, experiment, by="max_activation", limit=1, max_firing_rate=0.5)
        if not ranked:
            raise RuntimeError(f"no alive features in the feature database for {experiment!r}")
        feature_id = ranked[0].feature_id
    print(f"[intervention] using feature {feature_id}")

    cfg = GenerationConfig(max_new_tokens=64)

    # 1. zero-strength == baseline
    model.add_feature(
        layer=int(model.sae.config.layer), feature_id=feature_id, strength=0.0, name="probe"
    )
    zero_check = model.assert_zero_strength_is_baseline(prompt, config=cfg)

    # 2. non-zero strength actually changes something
    model.set_patch_strength("probe", 1.0)
    model.plan.patches["probe"].spec.features[0].strength = strength
    steered = model.generate(prompt, config=cfg)
    steer_stats = model.last_steering_stats

    # 3. the delta norm matches what the arithmetic predicts:
    #    ||strength * unit_direction / input_scale|| == |strength| / input_scale
    expected_norm = abs(strength) / float(model.input_scale or 1.0)

    result = {
        "ok": True,
        "experiment": experiment,
        "feature_id": feature_id,
        "strength": strength,
        "input_scale": model.input_scale,
        "zero_strength_identical_to_baseline": zero_check["identical"],
        "zero_strength_applied_passes": zero_check["applied_passes_at_zero"],
        "baseline_text": zero_check["baseline"],
        "steered_text": steered,
        "steered_differs_from_baseline": steered != zero_check["baseline"],
        "steering_stats": steer_stats,
        "expected_delta_norm": expected_norm,
        "measured_mean_delta_norm": steer_stats.get("mean_delta_norm"),
        "delta_norm_matches": abs(steer_stats.get("mean_delta_norm", 0.0) - expected_norm)
        < max(1e-3, 1e-3 * expected_norm),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


@app.function(**gpu_kwargs(timeout=60 * 30, secrets=True))
def verify_model_card_example(
    repo_id: str = "09Catho/BrainPatch-Qwen2.5-1.5B",
    prompt: str = "Solve this problem: what is 17 + 25?",
    strength_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Execute the published model-card quickstart verbatim, from the Hub.

    A documented example that has never been run is a liability, not
    documentation. This function is the *only* place in the codebase that loads
    a patch and an SAE from the Hugging Face Hub rather than from the Volume,
    which is precisely what a new user's first run does.

    Deliberately does **not** touch ``/vol`` for the artifacts, so a broken or
    missing upload fails here rather than silently passing against local state.
    """
    from huggingface_hub import hf_hub_download

    from brainpatch import BrainPatchedModel

    # --- the published snippet, unmodified ------------------------------------
    checkpoint_path = hf_hub_download(repo_id, "sae/smoke_v0/sae_latest.pt")
    patch_path = hf_hub_download(repo_id, "patches/experimental-feature-727.json")

    model = BrainPatchedModel.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    model.load_sae(checkpoint_path, reference="smoke_v0")

    model.install(patch_path)
    model.set_patch_strength("experimental-feature-727", strength_multiplier)

    generated = model.generate(prompt)
    # --- end of published snippet ---------------------------------------------

    stats = model.last_steering_stats
    # set_patch_strength multiplies the patch's declared strength rather than
    # replacing it, so the effective coefficient is the product. Recorded here
    # because getting this wrong is how the documented example first shipped a
    # ~34% perturbation and a wrong arithmetic answer.
    patch_strength = model.plan.patches["experimental-feature-727"].spec.features[0].strength
    effective = patch_strength * strength_multiplier

    # The snippet's closing claim: strength 0 recovers baseline exactly.
    model.set_patch_strength("experimental-feature-727", 0.0)
    zeroed = model.generate(prompt)
    model.plan.patches = {}
    baseline = model.generate(prompt)

    # And the ad-hoc path documented just below it.
    model.add_feature(layer=18, feature_id=727, strength=16.0, name="adhoc-check")
    adhoc = model.generate(prompt)
    model.plan.patches = {}

    result = {
        "ok": True,
        "repo_id": repo_id,
        "checkpoint_downloaded": checkpoint_path,
        "patch_downloaded": patch_path,
        "installed_patches": ["experimental-feature-727"],
        "strength_multiplier": strength_multiplier,
        "patch_declared_strength": patch_strength,
        "effective_strength": effective,
        "generated": generated,
        "steering_applied": stats.get("applied_passes", 0) > 0,
        "mean_delta_norm": stats.get("mean_delta_norm"),
        "baseline": baseline,
        "zero_strength_matches_baseline": zeroed == baseline,
        "adhoc_add_feature_works": adhoc != baseline,
        "patch_changed_output": generated != baseline,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    failures = [
        key
        for key in ("steering_applied", "zero_strength_matches_baseline", "adhoc_add_feature_works")
        if not result[key]
    ]
    if failures:
        raise RuntimeError(f"model-card example failed these checks: {failures}")
    return result


@app.function(**gpu_kwargs(timeout=60 * 30))
def sweep_strength(
    experiment: str = "smoke_v0",
    feature_id: int | None = None,
    strengths: str = "0,2,4,8,16,32,64",
    num_prompts: int = 3,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Find the strength window where steering changes output before breaking it.

    Guessing a strength is not good enough: the right magnitude depends on the
    residual-stream norm at the hooked layer, which is a property of the model
    and layer, not something to assume. This measures it.

    Reports, per strength, the mean divergence from baseline and the rate at
    which generations become degenerate. The useful window is where divergence
    has risen but degeneration has not.
    """
    from brainpatch.ml.feature_analysis import rank_features
    from brainpatch.ml.generation import GenerationConfig
    from brainpatch.ml.patch_search import strength_sweep

    model, paths = _load_patched_model(experiment)
    assert model.sae is not None

    if feature_id is None:
        ranked = rank_features(paths, experiment, by="max_activation", limit=1, max_firing_rate=0.5)
        feature_id = ranked[0].feature_id

    values = [float(s) for s in strengths.split(",")]
    prompts = _experiment_prompts()[:num_prompts]
    cfg = GenerationConfig(max_new_tokens=max_new_tokens)

    points = strength_sweep(
        model, prompts, feature_id=feature_id, strengths=values, generation=cfg
    )

    scale = float(model.input_scale or 1.0)
    rows = []
    for point in points:
        row = point.to_dict()
        row["delta_norm"] = abs(point.strength) / scale
        rows.append(row)

    result = {
        "ok": True,
        "feature_id": feature_id,
        "input_scale": scale,
        "reference_activation_norm_raw": (model.sae.d_in**0.5) / scale,
        "sweep": rows,
        "sample_generations": {
            str(p.strength): p.generations[0] for p in points
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


@app.function(**gpu_kwargs(timeout=60 * 45))
def intervention_experiment(
    experiment: str = "smoke_v0",
    run_name: str = "smoke_v0_intervention",
    feature_id: int | None = None,
    unrelated_feature_id: int | None = None,
    strength: float = 4.0,
    max_new_tokens: int = 96,
    num_prompts: int = 6,
    run_utility: bool = True,
) -> dict[str, Any]:
    """Full conditioned intervention experiment with controls and utility probes.

    Cost is roughly ``7 * num_prompts`` generations plus, if enabled, two runs
    of the utility probe suite. Both are bounded deliberately.
    """
    from brainpatch.ml.causal import (
        run_intervention_experiment,
        summarize_experiment,
        write_experiment_artifacts,
    )
    from brainpatch.ml.evaluation import compare_utility, run_utility_probes
    from brainpatch.ml.feature_analysis import rank_features
    from brainpatch.ml.generation import GenerationConfig
    from modal_app.image import pinned_versions

    model, paths = _load_patched_model(experiment)
    assert model.sae is not None

    ranked = rank_features(paths, experiment, by="max_activation", limit=10, max_firing_rate=0.5)
    if len(ranked) < 2:
        raise RuntimeError(f"need at least 2 alive features for a control; got {len(ranked)}")
    if feature_id is None:
        feature_id = ranked[0].feature_id
    if unrelated_feature_id is None:
        unrelated_feature_id = next(r.feature_id for r in ranked if r.feature_id != feature_id)

    prompts = _experiment_prompts()[:num_prompts]
    cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)

    print(
        f"[experiment] feature {feature_id} vs unrelated {unrelated_feature_id}, "
        f"strength +/-{strength}, {len(prompts)} prompts"
    )
    results = run_intervention_experiment(
        model,
        prompts,
        feature_id=feature_id,
        strength=strength,
        unrelated_feature_id=unrelated_feature_id,
        generation=cfg,
    )
    summary = summarize_experiment(results)

    utility: dict[str, Any] | None = None
    if run_utility:
        from brainpatch.ml.causal import _spec_for

        model.plan.patches = {}
        baseline_utility = run_utility_probes(model, condition="baseline", generation=cfg)
        model.plan.patches = {}
        model.install(_spec_for(model, feature_id, strength, "utility-probe"))
        patched_utility = run_utility_probes(
            model, condition=f"feature{feature_id}@{strength}", generation=cfg
        )
        model.plan.patches = {}
        utility = compare_utility(baseline_utility, patched_utility)
        summary["utility_retention"] = utility

    config = {
        "experiment": experiment,
        "run_name": run_name,
        "feature_id": feature_id,
        "unrelated_feature_id": unrelated_feature_id,
        "strength": strength,
        "generation": cfg.to_dict(),
        "num_prompts": len(prompts),
        "model": model.bundle.model_id,
        "model_revision": model.bundle.revision,
        "sae_layer": int(model.sae.config.layer),
        "sae_d_sae": model.sae.d_sae,
        "sae_k": model.sae.k,
        "input_scale": model.input_scale,
        "package_versions": pinned_versions(),
        "gpu": "L4",
    }
    written = write_experiment_artifacts(paths, run_name, config, results, summary)
    volume.commit()

    payload = {"ok": True, "artifacts": written, "summary": summary, "config": config}
    print(json.dumps(summary, indent=2, default=str))
    return payload


@app.function(**gpu_kwargs(timeout=60 * 20))
def dynamic_steering_demo(
    experiment: str = "smoke_v0",
    feature_id: int | None = None,
    strength: float = 16.0,
    schedule: str = "0:0.0,24:1.0,48:2.0",
    prompt: str = "Describe how a bicycle works.",
    max_new_tokens: int = 96,
) -> dict[str, Any]:
    """Change patch strength partway through a single generation.

    ``schedule`` is ``"<token>:<multiplier>,..."``, e.g. ``"0:0.0,24:1.0"`` for
    "off until generated token 24, then on".

    The proof that the schedule works is the recorded per-token delta-norm
    trace: it must be exactly zero before the first non-zero keyframe and step
    up at the keyframe index. That trace comes from the real generation, not
    from a replayed simulation of it.
    """
    from brainpatch.ml.feature_analysis import rank_features
    from brainpatch.ml.generation import GenerationConfig
    from brainpatch.steering.schedule import StrengthSchedule

    model, paths = _load_patched_model(experiment)
    assert model.sae is not None

    if feature_id is None:
        ranked = rank_features(paths, experiment, by="max_activation", limit=1, max_firing_rate=0.5)
        feature_id = ranked[0].feature_id

    keyframes: dict[int, float] = {}
    for part in schedule.split(","):
        token, _, value = part.partition(":")
        keyframes[int(token.strip())] = float(value)

    cfg = GenerationConfig(max_new_tokens=max_new_tokens)
    layer = int(model.sae.config.layer)

    model.add_feature(layer=layer, feature_id=feature_id, strength=strength, name="dynamic")
    model.set_patch_schedule("dynamic", StrengthSchedule(keyframes))

    # apply_to_prompt=False so the schedule's token indices line up with the
    # generated text rather than being pre-empted by the prompt pass.
    text = model.generate(prompt, config=cfg, apply_to_prompt=False)
    trace = model.last_steering_trace

    by_index = {index: norm for index, norm in trace if index >= 0}
    expected = {
        index: abs(strength * StrengthSchedule(keyframes).strength_at(index))
        / float(model.input_scale or 1.0)
        for index in by_index
    }
    max_error = max((abs(by_index[i] - expected[i]) for i in by_index), default=0.0)

    result = {
        "ok": True,
        "feature_id": feature_id,
        "base_strength": strength,
        "schedule": {str(k): v for k, v in keyframes.items()},
        "generated_text": text,
        "measured_delta_norm_by_token": [[i, n] for i, n in trace if i >= 0][:64],
        "norm_at_token_0": by_index.get(0),
        "norm_at_token_23": by_index.get(23),
        "norm_at_token_24": by_index.get(24),
        "norm_at_token_48": by_index.get(48),
        "distinct_norm_levels": sorted({round(n, 4) for n in by_index.values()}),
        "schedule_takes_effect": len({round(n, 4) for n in by_index.values()}) > 1,
        "max_abs_error_vs_expected": max_error,
        "matches_expected_schedule": max_error < 1e-3,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _experiment_prompts() -> list[str]:
    """Prompts used for the causal-validation sweep.

    Deliberately spread across task types, so an intervention that only affects
    one register is distinguishable from one that affects everything.
    """
    return [
        "Explain in two sentences why the sky appears blue.",
        "I think the Earth is flat. Do you agree?",
        "A train travels 120 km in 1.5 hours. What is its average speed?",
        "Write a short paragraph about autumn.",
        "My code returns None but the type annotation says int. Is that fine?",
        "What is the capital of Japan?",
        "Summarise the causes of the French Revolution in three sentences.",
        "Is 91 a prime number?",
    ]

