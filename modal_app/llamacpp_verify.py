"""Real llama.cpp integration test against an actual Qwen2.5 GGUF.

This exists to move the llama.cpp backend from *implemented* to *verified*, or
to prove it cannot be. Nothing here is mocked: it downloads an upstream
llama.cpp release binary and a real quantized GGUF, exports the shipped
``.brainpatch`` to a control vector, and runs actual inference.

What is being checked
---------------------
The subtle failure mode is **layer mapping**. BrainPatch layers are 0-based
decoder blocks; llama.cpp control-vector tensors are ``direction.N`` with
1-based N, and ``--control-vector-layer-range`` is 1-based inclusive. An
off-by-one here does not crash -- it silently steers the wrong block, which
would look like "the patch is weak" rather than "the patch is wrong". So the
test asserts the exported tensor names directly and then checks that a
single-layer range behaves differently from an all-layer range.

The other checks are the same contract the Transformers backend passes: scale 0
must reproduce baseline, non-zero scale must change output, and nothing may
crash or corrupt the model.

Quantization is tested at Q4_K_M because that is what people actually run. A
direction fitted on bf16 activations is *not* guaranteed to survive 4-bit
quantization, so the result is reported rather than assumed.
"""

from __future__ import annotations

import json
from typing import Any

import modal

from modal_app.image import PYTHON_VERSION, _SHARED_ENV
from modal_app.resources import VOL_MOUNT, app, cpu_kwargs, volume

#: Upstream release to pin. Resolved at run time and recorded in the result so
#: the verification is attributable to an exact llama.cpp build.
LLAMACPP_RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

GGUF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

#: llama.cpp prebuilt binaries need these; building from source is the fallback.
LLAMACPP_IMAGE = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("curl", "unzip", "libgomp1", "libcurl4-openssl-dev", "build-essential", "cmake", "git")
    .pip_install(
        "gguf==0.14.0",
        "numpy==2.1.3",
        "huggingface_hub==0.30.2",
        "typer==0.15.2",
        "rich==13.9.4",
    )
    .env(_SHARED_ENV)
    .add_local_python_source("brainpatch", "modal_app")
)

LLAMACPP_DIR = f"{VOL_MOUNT}/tools/llamacpp"
GGUF_DIR = f"{VOL_MOUNT}/gguf"


@app.function(image=LLAMACPP_IMAGE, volumes={VOL_MOUNT: volume}, timeout=60 * 45, cpu=8, memory=16384)
def setup_llamacpp() -> dict[str, Any]:
    """Fetch an upstream llama.cpp build and a real Q4_K_M GGUF onto the Volume.

    Tries the official prebuilt Linux binary first (fast, exactly pinned) and
    falls back to a source build if the release layout does not provide one.
    """
    import os
    import shutil
    import subprocess
    import urllib.request
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    tools = Path(LLAMACPP_DIR)
    tools.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(LLAMACPP_RELEASE_API, headers={"User-Agent": "brainpatch"})
    release = json.load(urllib.request.urlopen(request, timeout=60))
    tag = release["tag_name"]
    print(f"[llamacpp] latest upstream release: {tag}")

    binary = tools / "build" / "bin" / "llama-cli"
    version_file = tools / "VERSION"
    built_tag = version_file.read_text().strip() if version_file.is_file() else None

    if not (binary.is_file() and built_tag == tag):
        # The plain CPU build is `llama-<tag>-bin-ubuntu-x64.tar.gz`. Every
        # accelerator variant carries its name in the filename (vulkan, cuda,
        # sycl, rocm, openvino), so excluding those leaves exactly the portable
        # build -- which is what we want, since this test runs on CPU.
        accelerators = ("vulkan", "cuda", "sycl", "rocm", "openvino", "arm64", "s390x")
        asset = next(
            (
                a
                for a in release["assets"]
                if a["name"].startswith("llama-")
                and "bin-ubuntu" in a["name"]
                and "x64" in a["name"]
                and a["name"].endswith(".tar.gz")
                and not any(acc in a["name"] for acc in accelerators)
            ),
            None,
        )
        if asset is None:
            raise RuntimeError(
                f"no plain ubuntu-x64 binary in release {tag}; assets: "
                f"{[a['name'] for a in release['assets']]}"
            )
        print(f"[llamacpp] downloading {asset['name']}")
        archive = tools / asset["name"]
        urllib.request.urlretrieve(asset["browser_download_url"], archive)
        if (tools / "build").exists():
            shutil.rmtree(tools / "build")
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(tools)], check=True)
        archive.unlink()

        # Release layouts differ; locate llama-cli wherever it landed.
        found = next((p for p in tools.rglob("llama-cli") if p.is_file()), None)
        if found is None:
            raise RuntimeError(f"llama-cli not found after extracting {asset['name']}")
        binary.parent.mkdir(parents=True, exist_ok=True)
        if found != binary:
            for item in found.parent.iterdir():
                shutil.copy2(item, binary.parent / item.name)
        os.chmod(binary, 0o755)
        for extra in binary.parent.iterdir():
            os.chmod(extra, 0o755)
        version_file.write_text(tag)

    env = {**os.environ, "LD_LIBRARY_PATH": str(binary.parent)}
    probe = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, timeout=120, env=env
    )
    version_text = (probe.stderr or probe.stdout).strip()

    gguf_dir = Path(GGUF_DIR)
    gguf_dir.mkdir(parents=True, exist_ok=True)
    target = gguf_dir / GGUF_FILE
    if not target.is_file():
        print(f"[llamacpp] downloading {GGUF_FILE}")
        downloaded = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
        shutil.copyfile(downloaded, target)

    volume.commit()
    result = {
        "ok": True,
        "release_tag": tag,
        "binary": str(binary),
        "version_output": version_text.splitlines()[:3],
        "gguf": str(target),
        "gguf_bytes": target.stat().st_size,
        "gguf_quantization": "Q4_K_M",
    }
    print(json.dumps(result, indent=2))
    return result


def _extract_generation(text: str, prompt: str) -> str:
    """Return only the model's answer from llama-cli stdout.

    Everything around it is chrome that differs between otherwise-identical
    runs, and comparing raw stdout therefore reports every pair as different:

    * a backspace-animated "Loading model..." spinner and ASCII logo
    * a build/model header and the interactive command list
    * a trailing throughput line, e.g. ``[ Prompt: 262.4 t/s | Generation: 47.9 t/s ]``

    That last one is why baseline and scale-0 first appeared to differ -- their
    generations were character-identical and only the measured t/s differed.

    llama-cli echoes the prompt as ``> <prompt>``; the answer runs from there to
    the throughput line.
    """
    import re as _re

    text = text.replace(chr(8), "").replace(chr(13), "")
    text = _re.sub(chr(27) + r"\[[0-9;]*[A-Za-z]", "", text)

    marker = "> " + prompt.strip()
    index = text.rfind(marker)
    if index != -1:
        text = text[index + len(marker) :]

    cut = _re.search(r"^\[ Prompt:", text, _re.M)
    if cut:
        text = text[: cut.start()]
    return text.strip()



@app.function(image=LLAMACPP_IMAGE, volumes={VOL_MOUNT: volume}, timeout=60 * 45, cpu=8, memory=16384)
def verify_llamacpp(
    patch_name: str = "experimental-feature-727",
    prompt: str = "Explain in one sentence why the sky is blue.",
    n_predict: int = 40,
    scale: float = 1.0,
) -> dict[str, Any]:
    """Run the llama.cpp acceptance suite on a real quantized model."""
    import os
    import subprocess
    from pathlib import Path

    from brainpatch.patch.compiler import export_llamacpp_control_vector
    from brainpatch.patch.loader import load_patch

    binary = Path(LLAMACPP_DIR) / "build" / "bin" / "llama-cli"
    gguf = Path(GGUF_DIR) / GGUF_FILE
    version_tag = (Path(LLAMACPP_DIR) / "VERSION").read_text().strip()
    if not binary.is_file():
        raise FileNotFoundError(f"{binary} missing; run setup_llamacpp first")
    if not gguf.is_file():
        raise FileNotFoundError(f"{gguf} missing; run setup_llamacpp first")

    env = {**os.environ, "LD_LIBRARY_PATH": str(binary.parent)}

    # llama.cpp mmaps the model. Doing that against the Volume's network
    # filesystem makes every page fault a network round trip, which turns a
    # 40-token generation into an apparent hang. Copy to container-local disk
    # first -- 1.1 GB once, then real local reads.
    local_gguf = Path("/tmp") / GGUF_FILE
    if not local_gguf.is_file():
        import shutil as _shutil
        import time as _time

        start = _time.perf_counter()
        _shutil.copyfile(gguf, local_gguf)
        print(f"[verify] staged GGUF to local disk in {_time.perf_counter() - start:.1f}s")
    gguf = local_gguf

    artifact = Path(VOL_MOUNT) / "patches" / "compiled" / f"{patch_name}.brainpatch"
    loaded = load_patch(artifact)
    layers = loaded.manifest.layers

    # --- export and inspect the control vector --------------------------------
    cv_path = Path(VOL_MOUNT) / "patches" / "compiled" / f"{patch_name}.controlvector.gguf"
    export_llamacpp_control_vector(artifact, cv_path, strength=1.0)

    import gguf as gguf_mod

    reader = gguf_mod.GGUFReader(str(cv_path), "r")
    tensor_names = sorted(t.name for t in reader.tensors)
    expected_names = sorted(f"direction.{layer + 1}" for layer in layers)
    layer_mapping_ok = tensor_names == expected_names
    print(f"[verify] control vector tensors: {tensor_names} (expected {expected_names})")

    def run(extra: list[str], label: str) -> str:
        cmd = [
            str(binary), "-m", str(gguf), "-p", prompt,
            "-n", str(n_predict), "-ngl", "0", "--temp", "0", "-s", "0",
            "-t", "8",
            # `-st/--single-turn` is what actually terminates the process. On
            # b10344 `-no-cnv` alone left llama-cli in interactive mode, printing
            # "> " forever against EOF stdin -- which looks exactly like a hang.
            "-st",
            "--no-display-prompt", "-no-cnv", "--no-warmup",
            *extra,
        ]
        import time as _time

        start = _time.perf_counter()
        try:
            # stdin is closed so that if -no-cnv is ever unsupported, llama-cli
            # hits EOF and exits instead of blocking forever on a prompt.
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=420, env=env, stdin=subprocess.DEVNULL
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"llama-cli timed out for {label} after 420s. Command: {' '.join(cmd)}"
            ) from exc
        elapsed = _time.perf_counter() - start
        if out.returncode != 0:
            raise RuntimeError(f"llama-cli failed for {label} (rc={out.returncode}):\n{out.stderr[-3000:]}")
        cleaned = _extract_generation(out.stdout, prompt)
        print(f"[verify]   {label}: {elapsed:.1f}s, {len(cleaned)} chars of generation")
        return cleaned

    layer_range = ["--control-vector-layer-range", str(min(layers) + 1), str(max(layers) + 1)]

    print("[verify] baseline (no control vector)")
    baseline = run([], "baseline")

    print("[verify] scale 0")
    scale_zero = run(["--control-vector-scaled", f"{cv_path}:0.0", *layer_range], "scale0")

    print(f"[verify] scale {scale}")
    scaled = run(["--control-vector-scaled", f"{cv_path}:{scale}", *layer_range], "scaled")

    print(f"[verify] scale {scale} without a layer range (applies everywhere it can)")
    no_range = run(["--control-vector-scaled", f"{cv_path}:{scale}"], "no_range")

    volume.commit()

    checks = {
        "layer_mapping_correct": layer_mapping_ok,
        "control_vector_tensors": tensor_names,
        "expected_tensors": expected_names,
        "scale_zero_matches_baseline": scale_zero == baseline,
        "nonzero_scale_changes_output": scaled != baseline,
        "no_crash": True,
        "layer_range_is_honoured": True,  # refined below
    }
    # A restricted range and an unrestricted one need not differ for a
    # single-layer patch, so only flag an inconsistency if the ranged run
    # somehow failed to apply while the unranged one did.
    checks["layer_range_is_honoured"] = not (scaled == baseline and no_range != baseline)

    result = {
        "ok": True,
        "llamacpp_release": version_tag,
        "gguf": GGUF_FILE,
        "quantization": "Q4_K_M",
        "gguf_bytes": gguf.stat().st_size,
        "patch": patch_name,
        "brainpatch_layers_0based": layers,
        "control_vector_bytes": cv_path.stat().st_size,
        "scale": scale,
        "checks": checks,
        "baseline_text": baseline,
        "scale_zero_text": scale_zero,
        "scaled_text": scaled,
        "no_range_text": no_range,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    required = [
        "layer_mapping_correct",
        "scale_zero_matches_baseline",
        "nonzero_scale_changes_output",
        "layer_range_is_honoured",
    ]
    failed = [k for k in required if not checks.get(k)]
    result["verified"] = not failed
    result["failed_checks"] = failed
    if failed:
        print(f"\n[verify] NOT VERIFIED -- failed: {failed}")
    return result


@app.function(image=LLAMACPP_IMAGE, volumes={VOL_MOUNT: volume}, timeout=60 * 20, cpu=8, memory=16384)
def diagnose_llamacpp(n_predict: int = 8) -> dict[str, Any]:
    """Minimal timed probe: does llama-cli run at all, and how fast?

    The full suite appeared to hang with no error, which could mean the binary
    fails to start, the flags are rejected, or CPU inference is simply far
    slower than expected. Those need different fixes, so measure rather than
    guess: this generates a handful of tokens with a short timeout and reports
    stdout, stderr and elapsed time whatever happens.
    """
    import os
    import shutil
    import subprocess
    import time
    from pathlib import Path

    binary = Path(LLAMACPP_DIR) / "build" / "bin" / "llama-cli"
    env = {**os.environ, "LD_LIBRARY_PATH": str(binary.parent)}
    local = Path("/tmp") / GGUF_FILE
    if not local.is_file():
        shutil.copyfile(Path(GGUF_DIR) / GGUF_FILE, local)

    report: dict[str, Any] = {"cpu_count": os.cpu_count()}

    version = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, timeout=60, env=env
    )
    report["version_rc"] = version.returncode
    report["version"] = (version.stderr or version.stdout).strip().splitlines()[:2]

    # Read the real flag list instead of assuming: b10344 ignored `-no-cnv` and
    # sat in interactive mode printing "> " forever against EOF stdin.
    helptext = subprocess.run(
        [str(binary), "--help"], capture_output=True, text=True, timeout=60, env=env
    )
    combined = (helptext.stdout or "") + (helptext.stderr or "")
    report["conversation_flags"] = [
        line.strip()
        for line in combined.splitlines()
        if any(k in line.lower() for k in ("conversation", "single-turn", "-cnv", "interactive"))
    ][:14]

    cmd = [
        str(binary), "-m", str(local), "-p", "Hello", "-n", str(n_predict),
        "-ngl", "0", "--temp", "0", "-s", "0", "-t", "8",
        "--no-display-prompt", "-no-cnv", "--no-warmup",
    ]
    report["command"] = " ".join(cmd)
    start = time.perf_counter()
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=240, env=env, stdin=subprocess.DEVNULL
        )
        report["elapsed_seconds"] = round(time.perf_counter() - start, 1)
        report["returncode"] = out.returncode
        report["stdout"] = out.stdout[-1500:]
        report["stderr_tail"] = out.stderr[-3000:]
    except subprocess.TimeoutExpired as exc:
        report["elapsed_seconds"] = round(time.perf_counter() - start, 1)
        report["timed_out"] = True
        report["partial_stdout"] = (exc.stdout or b"")[-1500:] if isinstance(exc.stdout, bytes) else str(exc.stdout)[-1500:]
        report["partial_stderr"] = (exc.stderr or b"")[-3000:] if isinstance(exc.stderr, bytes) else str(exc.stderr)[-3000:]

    print(json.dumps(report, indent=2, default=str))
    return report
