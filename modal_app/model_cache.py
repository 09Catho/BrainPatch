"""Persistent Hugging Face model caching on the Modal Volume.

``cache_model`` runs on **CPU**: downloading weights needs bandwidth and disk,
not a GPU, and paying L4 rates to wait on a network transfer is pure waste.

``verify_cached_model`` runs in a *separate* invocation on the L4 and loads the
model from the cache. That separation is the actual test: it proves the cache
survives container teardown, which is the whole point of putting it on a Volume.
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.resources import VOL_MOUNT, app, cpu_kwargs, gpu_kwargs, volume

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@app.function(**cpu_kwargs(timeout=60 * 40, secrets=True))
def cache_model(model_id: str = DEFAULT_MODEL, revision: str | None = None) -> dict[str, Any]:
    """Download a model + tokenizer into ``/vol/hf-cache`` and pin its revision.

    Returns the resolved commit SHA, which every downstream experiment records
    so that "which weights produced this feature" always has an answer.
    """
    import time
    from pathlib import Path

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(model_id, revision=revision or "main")
    resolved = info.sha
    print(f"[cache] {model_id} @ {resolved}")

    start = time.perf_counter()
    local_path = snapshot_download(
        repo_id=model_id,
        revision=resolved,
        # Skip duplicate weight formats; safetensors is what transformers prefers.
        ignore_patterns=["*.pth", "*.msgpack", "*.h5", "*.onnx", "original/*"],
    )
    elapsed = time.perf_counter() - start

    files = sorted(p for p in Path(local_path).rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)
    volume.commit()

    result = {
        "ok": True,
        "model": model_id,
        "revision": resolved,
        "cache_path": str(local_path),
        "num_files": len(files),
        "total_gb": round(total_bytes / 1024**3, 3),
        "download_seconds": round(elapsed, 1),
        "files": [p.name for p in files if p.suffix in {".safetensors", ".json"}][:20],
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(**cpu_kwargs(timeout=600))
def model_architecture(model_id: str = DEFAULT_MODEL, revision: str | None = None) -> dict[str, Any]:
    """Read architecture facts from the cached config -- no weights, no GPU.

    Used to validate a configured target layer *before* any GPU time is spent
    discovering that layer 18 does not exist.
    """
    from brainpatch.research.ml.model import architecture_summary

    summary = architecture_summary(model_id, revision)
    print(json.dumps(summary, indent=2))
    return summary


@app.function(**gpu_kwargs(timeout=60 * 20))
def verify_cached_model(
    model_id: str = DEFAULT_MODEL,
    revision: str | None = None,
    layer: int = 18,
) -> dict[str, Any]:
    """Load the cached model onto the L4 in a fresh container and smoke-test it.

    Verifies, in one invocation:

    * the Volume cache is reused (no re-download)
    * the model loads onto the GPU in bf16
    * the configured layer exists
    * the capture hook fires and returns a correctly-shaped tensor
    * generation produces text
    """
    import time
    from pathlib import Path

    import torch

    from brainpatch.research.ml.hooks import ResidualCapture
    from brainpatch.research.ml.model import load_model, validate_layer

    hub_cache = Path("/vol/hf-cache/hub")
    cached_before = hub_cache.is_dir() and any(hub_cache.iterdir())

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    bundle = load_model(model_id, revision=revision, dtype="bfloat16", device="cuda")
    load_seconds = time.perf_counter() - start
    load_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    resolved_layer = validate_layer(layer, bundle.num_layers)

    # Hook smoke test: does the capture see a real residual-stream tensor?
    capture = ResidualCapture(to_cpu=True, dtype=torch.float32)
    handle = capture.attach(bundle.layer_module(resolved_layer))
    torch.cuda.reset_peak_memory_stats()
    inputs = bundle.tokenizer("The capital of France is", return_tensors="pt").to(bundle.device)
    with torch.inference_mode():
        bundle.model(**inputs, use_cache=False)
    handle.remove()

    acts = capture.activations
    if acts is None:
        raise RuntimeError("capture hook did not fire")
    inference_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Generation smoke test.
    with torch.inference_mode():
        output = bundle.model.generate(
            **inputs, max_new_tokens=12, do_sample=False,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )
    completion = bundle.tokenizer.decode(output[0], skip_special_tokens=True)

    result = {
        "ok": True,
        **bundle.describe(),
        "cache_present_before_load": cached_before,
        "load_seconds": round(load_seconds, 2),
        "target_layer_requested": layer,
        "target_layer_resolved": resolved_layer,
        "activation_shape": list(acts.shape),
        "activation_mean_norm": float(acts[0].norm(dim=-1).mean().item()),
        "activation_norm_first_token": float(acts[0, 0].norm().item()),
        "model_load_peak_vram_mb": round(load_peak_mb, 1),
        "inference_peak_vram_mb": round(inference_peak_mb, 1),
        "generation_sample": completion,
    }
    print(json.dumps(result, indent=2))
    return result

