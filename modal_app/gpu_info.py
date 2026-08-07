"""Remote environment probes.

``cpu_smoke`` is the cheapest possible round trip: it proves auth, image build,
Volume mount and code upload all work, without touching a GPU.

``gpu_info`` is the L4 probe. It reports the card, its memory, the CUDA and
PyTorch versions, bf16 support, and runs a real matmul so that "the GPU is
visible" and "the GPU computes correctly" are separately verified.

Neither function keeps a container warm.
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.resources import VOL_MOUNT, app, cpu_kwargs, gpu_kwargs, volume


@app.function(**cpu_kwargs(timeout=300))
def cpu_smoke() -> dict[str, Any]:
    """Cheapest end-to-end check: container, code upload, Volume mount.

    Also creates the canonical Volume directory layout if it is missing, so
    this doubles as `initialise the volume`.
    """
    import platform
    import sys
    from pathlib import Path

    from brainpatch import __version__
    from brainpatch.paths import VolumePaths

    paths = VolumePaths(VOL_MOUNT)
    created: list[str] = []
    for directory in paths.all_top_level():
        p = Path(directory)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created.append(str(directory))
    volume.commit()

    existing = sorted(p.name for p in Path(VOL_MOUNT).iterdir() if p.is_dir())

    result = {
        "ok": True,
        "brainpatch_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "volume_mounted": Path(VOL_MOUNT).is_dir(),
        "volume_dirs_created": created,
        "volume_dirs_present": existing,
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(**gpu_kwargs(timeout=600))
def gpu_info() -> dict[str, Any]:
    """Report L4 capabilities and run a CUDA correctness smoke test."""
    import time

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in a GPU container -- image or driver problem")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    # Correctness smoke test: compare a GPU matmul against a CPU reference.
    torch.manual_seed(0)
    a = torch.randn(512, 512, dtype=torch.float32)
    b = torch.randn(512, 512, dtype=torch.float32)
    expected = a @ b
    got = (a.to(device) @ b.to(device)).cpu()
    max_abs_err = (expected - got).abs().max().item()

    # bf16 support is what lets us hold activations at half the memory.
    bf16_supported = torch.cuda.is_bf16_supported()
    bf16_ok = None
    if bf16_supported:
        x = torch.randn(256, 256, dtype=torch.bfloat16, device=device)
        y = (x @ x).float()
        bf16_ok = bool(torch.isfinite(y).all().item())

    # Rough throughput reference so later cost estimates have a baseline.
    torch.cuda.reset_peak_memory_stats()
    big = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        big = big @ big.T / 2048.0
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tflops = (50 * 2 * 2048**3) / elapsed / 1e12

    result = {
        "ok": True,
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "vram_total_gb": round(total_bytes / 1024**3, 2),
        "vram_free_gb": round(free_bytes / 1024**3, 2),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "bf16_supported": bf16_supported,
        "bf16_smoke_finite": bf16_ok,
        "matmul_max_abs_err_vs_cpu": max_abs_err,
        "bf16_matmul_tflops": round(tflops, 2),
        "peak_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }
    print(json.dumps(result, indent=2))
    return result

