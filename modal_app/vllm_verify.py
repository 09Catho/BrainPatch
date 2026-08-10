"""Real vLLM integration test: engine, server, concurrency, throughput.

This moves the vLLM backend from *implemented* to *verified*, or proves it
cannot be. The central claim being tested is that the intervention runs **inside
vLLM's own inference path** -- not in a Transformers model quietly substituted
behind the scenes.

That claim is verified two ways, not one:

1. The worker RPC returns the model class, layer count and active hook count
   from *inside the vLLM worker process*. If hooks had failed to install, or if
   some other engine were doing the work, this report would say so.
2. Output changes only when those hooks are present, and reverts exactly when
   they are removed.

Concurrency is tested against a real HTTP server with two simultaneous
requests, because "patch state does not leak between requests" is a claim about
batched execution, and serial calls cannot test it.
"""

from __future__ import annotations

import json
from typing import Any

import modal

from modal_app.image import PYTHON_VERSION, _SHARED_ENV
from modal_app.resources import VOL_MOUNT, app, gpu_kwargs, volume

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

#: vLLM pulls its own torch; letting it resolve avoids a version conflict that
#: would silently fall back to a CPU build.
VLLM_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        # vLLM pins torch, pydantic and fastapi tightly (0.11.0 needs
        # pydantic>=2.11.7). Pinning those ourselves produced a
        # ResolutionImpossible, so only vLLM is pinned and the rest is left to
        # its resolver -- the exact resolved versions are reported by the test.
        "vllm==0.11.0",
        # vLLM 0.11.0 predates transformers v5, which removed
        # `all_special_tokens_extended`; leaving this unpinned resolves to v5 and
        # crashes inside vLLM's tokenizer wrapper before the model ever loads.
        "transformers>=4.55,<5",
        "huggingface_hub",
        "typer",
        "rich",
        "uvicorn",
    )
    .env({**_SHARED_ENV, "VLLM_LOGGING_LEVEL": "WARNING"})
    .add_local_python_source("brainpatch", "modal_app")
)


@app.function(
    image=VLLM_IMAGE,
    volumes={VOL_MOUNT: volume},
    gpu="L4",
    timeout=60 * 40,
    retries=0,
    scaledown_window=60,
)
def verify_vllm(
    patch_name: str = "experimental-feature-727",
    strength: float = 1.0,
    max_new_tokens: int = 48,
) -> dict[str, Any]:
    """Engine-level acceptance: hooks inside vLLM, scale 0, non-zero, batching."""
    from pathlib import Path

    import vllm

    from brainpatch.patch.loader import load_patch
    from brainpatch.runtime.base import GenerationConfig
    from brainpatch.backends.vllm_backend import VLLMBackend

    artifact = Path(VOL_MOUNT) / "patches" / "compiled" / f"{patch_name}.brainpatch"
    if not artifact.is_file():
        raise FileNotFoundError(f"compiled patch not found: {artifact}")

    backend = VLLMBackend()
    backend.load_model(MODEL, revision=REVISION, max_model_len=1024)
    descriptor = backend.describe_model()
    cfg = GenerationConfig(max_new_tokens=max_new_tokens)

    prompts = [
        "Explain in one sentence why the sky is blue.",
        "What is the capital of Japan?",
    ]

    checks: dict[str, Any] = {}
    checks["vllm_version"] = vllm.__version__
    checks["worker_before_patch"] = backend.worker_state()

    # --- baseline --------------------------------------------------------------
    baseline = backend.generate_batch(prompts, cfg)

    # --- install, and prove hooks landed inside the worker ---------------------
    loaded = load_patch(artifact)
    backend.install_patch(loaded, strength=strength)
    report = backend.last_hook_report
    checks["worker_hook_report"] = report
    checks["hooks_installed_in_vllm_worker"] = bool(report) and all(
        r.get("num_hooks", 0) > 0 for r in report
    )
    checks["hooked_layers"] = report[0].get("hooked_layers") if report else []
    checks["worker_model_class"] = report[0].get("model_class") if report else None

    patched = backend.generate_batch(prompts, cfg)
    checks["patch_changes_output"] = patched != baseline

    # --- scale 0 must reproduce baseline ---------------------------------------
    backend.set_strength(loaded.manifest.name, 0.0)
    checks["worker_hooks_at_zero"] = backend.last_hook_report[0].get("num_hooks")
    zeroed = backend.generate_batch(prompts, cfg)
    checks["scale_zero_matches_baseline"] = zeroed == baseline

    # --- back on, then removed --------------------------------------------------
    backend.set_strength(loaded.manifest.name, strength)
    checks["reenabled_matches_patched"] = backend.generate_batch(prompts, cfg) == patched
    backend.remove_patch(loaded.manifest.name)
    checks["removed_matches_baseline"] = backend.generate_batch(prompts, cfg) == baseline

    # --- batched consistency: same prompt twice in one batch -------------------
    backend.install_patch(loaded, strength=strength)
    duplicated = backend.generate_batch([prompts[0], prompts[0]], cfg)
    checks["identical_prompts_in_one_batch_agree"] = duplicated[0] == duplicated[1]
    checks["batched_matches_single"] = duplicated[0] == patched[0]

    result = {
        "ok": True,
        "vllm_version": vllm.__version__,
        "model": descriptor.model_id,
        "architecture": descriptor.architecture,
        "hidden_size": descriptor.hidden_size,
        "num_layers": descriptor.num_layers,
        "strength": strength,
        "checks": checks,
        "baseline": baseline,
        "patched": patched,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False)[:6000])

    required = [
        "hooks_installed_in_vllm_worker",
        "patch_changes_output",
        "scale_zero_matches_baseline",
        "reenabled_matches_patched",
        "removed_matches_baseline",
        "identical_prompts_in_one_batch_agree",
        "batched_matches_single",
    ]
    failed = [k for k in required if not checks.get(k)]
    result["verified"] = not failed
    result["failed_checks"] = failed
    if failed:
        print(f"\n[verify] NOT VERIFIED -- failed: {failed}")
    return result


@app.function(
    image=VLLM_IMAGE,
    volumes={VOL_MOUNT: volume},
    gpu="L4",
    timeout=60 * 40,
    retries=0,
    scaledown_window=60,
)
def verify_vllm_server(
    patch_name: str = "experimental-feature-727",
    strength: float = 1.0,
    port: int = 8111,
) -> dict[str, Any]:
    """Serve OpenAI-compatible HTTP over vLLM and hit it concurrently."""
    import threading
    import time
    import urllib.error
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import uvicorn
    import vllm

    from brainpatch.patch.loader import load_patch
    from brainpatch.runtime.model import BrainPatchedModel
    from brainpatch.backends.vllm_backend import VLLMBackend
    from brainpatch.server.app import build_app

    artifact = Path(VOL_MOUNT) / "patches" / "compiled" / f"{patch_name}.brainpatch"

    backend = VLLMBackend()
    backend.load_model(MODEL, revision=REVISION, max_model_len=1024)
    model = BrainPatchedModel(backend)
    model.install(load_patch(artifact), strength=strength)

    api = build_app(model, served_model_name="qwen2.5-1.5b-brainpatch")
    config = uvicorn.Config(api, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"

    def wait_ready(timeout: int = 120) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
                    if response.status == 200:
                        return True
            except Exception:  # noqa: BLE001
                time.sleep(1)
        return False

    if not wait_ready():
        raise RuntimeError("server did not become ready")

    def post(path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        request = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def chat(content: str, extra: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": "qwen2.5-1.5b-brainpatch",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 40,
            "temperature": 0.0,
        }
        if extra:
            payload.update(extra)
        return post("/v1/chat/completions", payload)

    checks: dict[str, Any] = {}

    with urllib.request.urlopen(f"{base}/health", timeout=10) as response:
        health = json.loads(response.read())
    checks["health"] = health
    checks["server_reports_patch"] = patch_name in health.get("patches", {})

    with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as response:
        models = json.loads(response.read())
    checks["models_endpoint_openai_shaped"] = models.get("object") == "list"

    status, body = chat("What is the capital of Japan?")
    checks["chat_status"] = status
    if status != 200 or "choices" not in body:
        # Surface the server's own error rather than dying on a KeyError, which
        # tells you nothing about why the request failed.
        raise RuntimeError(f"chat request failed: HTTP {status}: {json.dumps(body)[:1500]}")
    checks["chat_openai_shaped"] = (
        body.get("object") == "chat.completion"
        and isinstance(body.get("choices"), list)
        and "content" in body["choices"][0]["message"]
    )
    serial_answer = body["choices"][0]["message"]["content"]

    # --- two simultaneous requests --------------------------------------------
    prompts = ["What is the capital of Japan?", "Name one primary colour."]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(chat, p) for p in prompts]
        concurrent = [f.result() for f in futures]

    checks["both_concurrent_requests_succeeded"] = all(s == 200 for s, _ in concurrent)
    concurrent_texts = [b["choices"][0]["message"]["content"] for _, b in concurrent]
    checks["concurrent_texts"] = concurrent_texts
    # The shared prompt must produce the same answer whether issued alone or
    # alongside another request: that is what "no state leak" means here.
    checks["concurrent_matches_serial"] = concurrent_texts[0] == serial_answer
    checks["concurrent_requests_differ_from_each_other"] = (
        concurrent_texts[0] != concurrent_texts[1]
    )

    # --- per-request override must be refused, not silently ignored ------------
    status_bad, body_bad = chat("hello", {"brainpatch": {patch_name: 7.5}})
    checks["mismatched_per_request_strength_rejected"] = status_bad == 400
    checks["rejection_explains_why"] = "per-request" in json.dumps(body_bad).lower()

    status_ok, _ = chat("hello", {"brainpatch": {patch_name: strength}})
    checks["matching_per_request_strength_accepted"] = status_ok == 200

    # --- throughput ------------------------------------------------------------
    def timed(n: int = 4) -> float:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(lambda i: chat(f"Count to three. Request {i}."), range(n)))
        return time.perf_counter() - start

    patched_seconds = timed()
    backend.end_serving()
    model.backend.set_enabled(patch_name, False)
    backend.begin_serving()
    baseline_seconds = timed()

    server.should_exit = True
    thread.join(timeout=30)

    result = {
        "ok": True,
        "vllm_version": vllm.__version__,
        "model": MODEL,
        "checks": checks,
        "throughput": {
            "concurrent_requests": 4,
            "baseline_seconds": round(baseline_seconds, 3),
            "patched_seconds": round(patched_seconds, 3),
            "overhead_percent": round(
                (patched_seconds - baseline_seconds) / baseline_seconds * 100, 2
            ),
            "caveat": "4 concurrent requests, single measurement; noise dominates small differences.",
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False)[:6000])

    required = [
        "server_reports_patch",
        "models_endpoint_openai_shaped",
        "chat_openai_shaped",
        "both_concurrent_requests_succeeded",
        "concurrent_matches_serial",
        "mismatched_per_request_strength_rejected",
        "matching_per_request_strength_accepted",
    ]
    failed = [k for k in required if not checks.get(k)]
    result["verified"] = not failed
    result["failed_checks"] = failed
    if failed:
        print(f"\n[verify] SERVER NOT VERIFIED -- failed: {failed}")
    return result
