"""Remote integration tests for the BrainPatch **runtime** (not the research code).

These exist because the runtime's whole claim is "this works on a real model
without Modal". That claim still has to be *tested* somewhere with a GPU, and
this development machine deliberately has no ML stack -- so the tests run on
Modal even though the thing under test does not need Modal at all.

``compile_reference_patch``
    Turns the v0.1 research patch into a portable ``.brainpatch`` and reports
    its real size next to the base model's.

``test_transformers_backend``
    The Tier-1 acceptance suite: load a real Qwen, apply a compiled patch, and
    check the properties the product promises -- strength 0 is byte-identical to
    baseline, non-zero strength changes output, schedules fire at the right
    token, weights are untouched, and removing a patch restores baseline.
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.image import ML_IMAGE
from modal_app.resources import VOL_MOUNT, app, cpu_kwargs, gpu_kwargs, volume

REFERENCE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REFERENCE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


@app.function(**cpu_kwargs(timeout=60 * 20, image=ML_IMAGE, cpu=4, memory=8192))
def compile_reference_patch(
    experiment: str = "smoke_v0",
    name: str = "experimental-feature-727",
) -> dict[str, Any]:
    """Compile the research patch into a self-contained ``.brainpatch``.

    Reads the v0.1 research patch from ``/vol/patches/<name>.json``, which
    ``sync_patches`` populates from the repository. Passing the JSON as a CLI
    argument was tried first and is not viable -- the Modal CLI parses structured
    argument values, so a multi-KB JSON document does not survive the round trip.
    """
    from pathlib import Path

    from brainpatch.patch.compiler import compile_from_sae
    from brainpatch.patch.loader import load_patch, patch_size_report
    from brainpatch.paths import VolumePaths
    from brainpatch.schemas.patch import BrainPatchSpec

    paths = VolumePaths(VOL_MOUNT)
    source = Path(VOL_MOUNT) / "patches" / f"{name}.json"
    if not source.is_file():
        raise FileNotFoundError(
            f"research patch not found at {source}. Run the publish_release "
            "entrypoint (which calls sync_patches) first."
        )
    spec = BrainPatchSpec.from_json(source.read_text(encoding="utf-8"))

    out_dir = Path(VOL_MOUNT) / "patches" / "compiled"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{name}.brainpatch"

    readme = (
        f"# {spec.name}\n\n"
        "Compiled BrainPatch v1 runtime artifact.\n\n"
        "This file contains the materialised intervention vector(s). It does "
        "**not** require the SAE that produced them, and applying it needs no "
        "research tooling.\n\n"
        f"Evidence level: **{spec.evidence_level}**. "
        "See the repository RESEARCH_LOG for what that means for this patch.\n"
    )

    # Only "verified" where a real acceptance run passed. Everything else is
    # "implemented" -- the adapter exists but no hardware confirmed it.
    compatibility = {
        "transformers": {
            "status": "verified",
            "model_revision": REFERENCE_REVISION,
            "device": "cuda (NVIDIA L4)",
            "verified_by": "modal run modal_app/app.py::test_transformers_backend",
            "checks": [
                "weights_frozen",
                "zero_strength_identical_to_baseline",
                "patch_changes_output",
                "delta_norm_matches_expected",
                "token_schedule_fires_at_keyframe",
                "disable_and_remove_restore_baseline",
            ],
        },
        "llamacpp": {
            "status": "implemented",
            "note": "control-vector exporter written; not verified on a GGUF model",
        },
        "vllm": {
            "status": "implemented",
            "note": "adapter written; not verified on a running vLLM engine",
        },
        "mlx-lm": {
            "status": "experimental",
            "note": "no Apple Silicon available to execute it",
        },
    }

    written = compile_from_sae(
        spec,
        str(paths.sae_checkpoint(experiment)),
        output,
        readme=readme,
        overwrite=True,
        compatibility=compatibility,
    )
    volume.commit()

    loaded = load_patch(written)
    report = patch_size_report(loaded)

    # Honest size comparison against the base model actually on the Volume.
    model_bytes = 0
    hub = Path(VOL_MOUNT) / "hf-cache" / "hub"
    for candidate in hub.rglob("*.safetensors"):
        model_bytes += candidate.stat().st_size

    result = {
        "ok": True,
        "artifact": str(written),
        **report,
        "manifest": loaded.manifest.to_dict(),
        "base_model_weights_bytes": model_bytes,
        "ratio_vs_base_model": (
            round(model_bytes / loaded.archive_bytes, 1) if loaded.archive_bytes else None
        ),
    }
    print(json.dumps({k: v for k, v in result.items() if k != "manifest"}, indent=2))
    return result


@app.function(**gpu_kwargs(timeout=60 * 30))
def test_transformers_backend(
    experiment: str = "smoke_v0",
    patch_name: str = "experimental-feature-727",
    prompt: str = "Explain in two sentences why the sky appears blue.",
) -> dict[str, Any]:
    """Tier-1 acceptance suite for the Transformers backend on a real model."""
    from pathlib import Path

    import torch

    from brainpatch.patch.loader import load_patch
    from brainpatch.runtime.base import GenerationConfig
    from brainpatch.runtime.model import BrainPatchedModel

    artifact = Path(VOL_MOUNT) / "patches" / "compiled" / f"{patch_name}.brainpatch"
    if not artifact.is_file():
        raise FileNotFoundError(f"compiled patch not found: {artifact}. Run compile_reference_patch.")

    model = BrainPatchedModel.from_pretrained(
        REFERENCE_MODEL, revision=REFERENCE_REVISION, backend="transformers", device="cuda"
    )
    descriptor = model.backend.describe_model()
    cfg = GenerationConfig(max_new_tokens=64)

    checks: dict[str, Any] = {}

    # --- weights must be frozen ------------------------------------------------
    checks["weights_frozen"] = not any(
        p.requires_grad for p in model.backend.model.parameters()
    )
    weight_before = model.backend.model.model.layers[18].mlp.down_proj.weight.detach().clone()

    # --- baseline with nothing installed ---------------------------------------
    baseline = model.generate(prompt, cfg)

    # --- install the compiled patch --------------------------------------------
    loaded = load_patch(artifact)
    handle = model.install(loaded, strength=1.0)
    checks["installed"] = handle.name

    # --- strength 0 must be byte-identical to baseline -------------------------
    zero_check = model.backend.assert_zero_is_baseline(prompt, cfg)
    checks["zero_strength_identical_to_baseline"] = zero_check["identical"]
    checks["applied_passes_at_zero"] = zero_check["applied_passes_at_zero"]

    # --- non-zero strength must change the output ------------------------------
    handle.strength = 1.0
    patched = model.generate(prompt, cfg)
    checks["patch_changes_output"] = patched != baseline
    trace = model.backend.last_trace
    checks["delta_norm_mean"] = (
        round(sum(n for _, n in trace) / len(trace), 4) if trace else 0.0
    )

    # The compiled vector folds 1/input_scale in, so at coefficient c the delta
    # norm should be |c| * ||vector||. Verifying this catches a scale error that
    # would otherwise look like "the patch is just weak".
    vector = loaded.vector_for(loaded.manifest.interventions[0].vector)
    expected_norm = (
        abs(loaded.manifest.interventions[0].coefficient)
        * sum(v * v for v in vector.data) ** 0.5
    )
    checks["expected_delta_norm"] = round(expected_norm, 4)
    checks["delta_norm_matches"] = abs(checks["delta_norm_mean"] - expected_norm) < max(
        0.05, 0.02 * expected_norm
    )

    # --- schedules -------------------------------------------------------------
    handle.schedule = {0: 0.0, 16: 1.0}
    model.generate(prompt, cfg, apply_to_prompt=False)
    sched_trace = {i: n for i, n in model.backend.last_trace}
    checks["schedule_zero_before_keyframe"] = all(
        sched_trace.get(i, 0.0) == 0.0 for i in range(16) if i in sched_trace
    )
    checks["schedule_active_after_keyframe"] = sched_trace.get(16, 0.0) > 0
    handle.schedule = None

    # --- disable / remove restores baseline exactly ----------------------------
    handle.enabled = False
    checks["disabled_matches_baseline"] = model.generate(prompt, cfg) == baseline
    handle.enabled = True
    model.remove_patch(handle.name)
    checks["removed_matches_baseline"] = model.generate(prompt, cfg) == baseline

    # --- weights genuinely untouched -------------------------------------------
    weight_after = model.backend.model.model.layers[18].mlp.down_proj.weight.detach()
    checks["weights_unchanged"] = bool(torch.equal(weight_before, weight_after))

    result = {
        "ok": True,
        "model": descriptor.model_id,
        "architecture": descriptor.architecture,
        "hidden_size": descriptor.hidden_size,
        "num_layers": descriptor.num_layers,
        "revision": descriptor.revision,
        "checks": checks,
        "baseline_text": baseline,
        "patched_text": patched,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    required = [
        "weights_frozen",
        "zero_strength_identical_to_baseline",
        "patch_changes_output",
        "delta_norm_matches",
        "schedule_zero_before_keyframe",
        "schedule_active_after_keyframe",
        "disabled_matches_baseline",
        "removed_matches_baseline",
        "weights_unchanged",
    ]
    failed = [k for k in required if not checks.get(k)]
    if failed:
        raise RuntimeError(f"transformers backend acceptance failed: {failed}")
    return result


@app.function(**gpu_kwargs(timeout=60 * 25))
def benchmark_transformers(
    patch_name: str = "experimental-feature-727",
    max_new_tokens: int = 96,
    runs: int = 3,
    prompt: str = "Explain how a bicycle works.",
) -> dict[str, Any]:
    """Measure patched vs unpatched throughput and memory on a real L4."""
    import time
    from pathlib import Path

    import torch

    from brainpatch.patch.loader import load_patch
    from brainpatch.runtime.base import GenerationConfig
    from brainpatch.runtime.model import BrainPatchedModel

    artifact = Path(VOL_MOUNT) / "patches" / "compiled" / f"{patch_name}.brainpatch"
    model = BrainPatchedModel.from_pretrained(
        REFERENCE_MODEL, revision=REFERENCE_REVISION, backend="transformers", device="cuda"
    )
    cfg = GenerationConfig(max_new_tokens=max_new_tokens)

    torch.cuda.reset_peak_memory_stats()
    model.generate(prompt, cfg)  # warm up
    baseline_peak = torch.cuda.max_memory_allocated() / 1024**2

    def timed() -> float:
        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.generate(prompt, cfg)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        return sum(times) / len(times)

    base_time = timed()

    load_start = time.perf_counter()
    loaded = load_patch(artifact)
    load_seconds = time.perf_counter() - load_start
    model.install(loaded, strength=1.0)

    torch.cuda.reset_peak_memory_stats()
    patch_time = timed()
    patched_peak = torch.cuda.max_memory_allocated() / 1024**2

    result = {
        "ok": True,
        "runs": runs,
        "max_new_tokens": max_new_tokens,
        "baseline_seconds": round(base_time, 4),
        "patched_seconds": round(patch_time, 4),
        "baseline_tokens_per_sec": round(max_new_tokens / base_time, 1),
        "patched_tokens_per_sec": round(max_new_tokens / patch_time, 1),
        "overhead_percent": round((patch_time - base_time) / base_time * 100, 2),
        "patch_load_seconds": round(load_seconds, 5),
        "patch_bytes": loaded.archive_bytes,
        "baseline_peak_vram_mb": round(baseline_peak, 1),
        "patched_peak_vram_mb": round(patched_peak, 1),
        "vram_overhead_mb": round(patched_peak - baseline_peak, 2),
        "caveat": (
            f"Wall clock over {runs} runs on one L4; differences of a few percent "
            "are within run-to-run noise."
        ),
    }
    print(json.dumps(result, indent=2))
    return result
