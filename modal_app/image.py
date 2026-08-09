"""Modal Image definitions.

Two images, so that cheap work never pays for the heavy dependency set:

``CPU_IMAGE``
    Tiny. Standard library plus the BrainPatch core (pyyaml/typer/rich). Used
    for volume bookkeeping, feature-database post-processing, report
    generation, and Hugging Face uploads.

``ML_IMAGE``
    torch + transformers + datasets. Used only for GPU work: model caching,
    activation extraction, SAE training, and interventions.

Dependencies are baked into the image, never `pip install`ed per invocation.
Versions are pinned so that a rebuilt image reproduces a previous experiment.
"""

from __future__ import annotations

import modal

#: Python version for all BrainPatch containers.
PYTHON_VERSION = "3.11"

#: Pinned dependency versions. Recorded in every experiment's provenance block.
TORCH_VERSION = "2.6.0"
TRANSFORMERS_VERSION = "4.51.3"
DATASETS_VERSION = "3.5.0"
ACCELERATE_VERSION = "1.6.0"
SAFETENSORS_VERSION = "0.5.3"
HUGGINGFACE_HUB_VERSION = "0.30.2"
NUMPY_VERSION = "2.1.3"

#: Local Python sources copied into every container at start-up.
#: Added as a mount rather than an image layer so editing code does not
#: trigger a rebuild.
_LOCAL_SOURCES = ("brainpatch", "modal_app")

#: Environment shared by both images. Points every Hugging Face cache at the
#: Volume so a model is downloaded exactly once, ever.
_SHARED_ENV = {
    "HF_HOME": "/vol/hf-cache",
    "HF_HUB_CACHE": "/vol/hf-cache/hub",
    "HF_DATASETS_CACHE": "/vol/hf-cache/datasets",
    # hf_transfer is deliberately OFF. Its parallel range-writes fail against
    # the Volume's network filesystem under gVisor ("An error occurred while
    # downloading using hf_transfer"), and a model is downloaded exactly once
    # ever, so the throughput gain is worth nothing next to the reliability loss.
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}

#: Parquet writer for the published feature table. Kept in the CPU image only --
#: building a dataset release is I/O and schema work, not GPU work.
PYARROW_VERSION = "19.0.1"

_CPU_PACKAGES = (
    "pyyaml==6.0.2",
    "typer==0.15.2",
    "rich==13.9.4",
    f"huggingface_hub=={HUGGINGFACE_HUB_VERSION}",
    f"numpy=={NUMPY_VERSION}",
    f"safetensors=={SAFETENSORS_VERSION}",
    f"pyarrow=={PYARROW_VERSION}",
)

_ML_PACKAGES = (
    f"torch=={TORCH_VERSION}",
    f"transformers=={TRANSFORMERS_VERSION}",
    f"accelerate=={ACCELERATE_VERSION}",
    f"datasets=={DATASETS_VERSION}",
    f"safetensors=={SAFETENSORS_VERSION}",
    f"huggingface_hub=={HUGGINGFACE_HUB_VERSION}",
    f"numpy=={NUMPY_VERSION}",
    "pyyaml==6.0.2",
)

_WEB_PACKAGES = ("gradio==5.23.3", "fastapi==0.115.12")


def _build(*packages: str, local_dirs: tuple[tuple[str, str], ...] = ()) -> modal.Image:
    """Assemble an image with local sources added LAST.

    Modal forbids any build step after ``add_local_*``, because local files are
    attached at container start rather than baked into a layer. Composing a new
    image by appending ``pip_install`` to an existing one therefore fails at
    run time, not at import time -- so images are built from their package list
    here rather than by extending one another.

    ``local_dirs`` are ``(local_path, remote_path)`` pairs attached after the
    Python sources. Consecutive ``add_local_*`` calls are fine; only a *build*
    step after one is forbidden.
    """
    image = (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install(*packages)
        .env(_SHARED_ENV)
        .add_local_python_source(*_LOCAL_SOURCES)
    )
    for local_path, remote_path in local_dirs:
        image = image.add_local_dir(local_path, remote_path=remote_path)
    return image


CPU_IMAGE = _build(*_CPU_PACKAGES)
ML_IMAGE = _build(*_ML_PACKAGES)

#: Image for the Gradio demo: the full ML stack plus the UI layer. The demo runs
#: real generation, so it needs torch; keeping it a separate image stops the UI
#: dependencies from bloating every training container.
WEB_IMAGE = _build(*_ML_PACKAGES, *_WEB_PACKAGES)

#: Image for the remote pytest suite (``tests/remote/``). Carries the ML stack
#: plus pytest, and mounts the test directory. CPU only -- the SAE regression
#: tests need torch maths, not a GPU.
TEST_IMAGE = _build(
    *_ML_PACKAGES,
    "pytest==8.3.5",
    local_dirs=(("tests", "/root/tests"),),
)


def pinned_versions() -> dict[str, str]:
    """Dependency pins, for recording in experiment provenance."""
    return {
        "python": PYTHON_VERSION,
        "torch": TORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
        "datasets": DATASETS_VERSION,
        "accelerate": ACCELERATE_VERSION,
        "safetensors": SAFETENSORS_VERSION,
        "huggingface_hub": HUGGINGFACE_HUB_VERSION,
        "numpy": NUMPY_VERSION,
        "pyarrow": PYARROW_VERSION,
    }
