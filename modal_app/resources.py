"""Shared Modal resources: the App, the Volume, the Secret, and cost guards.

Cost policy
-----------
This project runs under a hard ~$10 budget, so the defaults here are
deliberately conservative:

* one GPU, always ``L4`` (:data:`DEFAULT_GPU`)
* short timeouts, so a hung job cannot quietly burn an hour
* ``retries=0`` on GPU functions -- an automatic retry of an expensive job is a
  silent doubling of cost; failures should be read and fixed, not re-rolled
* ``scaledown_window`` kept small so containers do not idle warm

:func:`assert_token_budget` is the tripwire for the "no experiment above 50k
activation tokens without approval" rule. It is called by the extraction entry
point rather than buried in a helper, so it is impossible to bypass by
accident.
"""

from __future__ import annotations

import modal

from brainpatch.paths import HF_SECRET_NAME, VOLUME_NAME
from modal_app.image import CPU_IMAGE, ML_IMAGE

APP_NAME = "brainpatch"

app = modal.App(APP_NAME)

#: Persistent storage for models, activations, checkpoints and experiments.
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

#: Hugging Face credentials. Exposes ``HF_TOKEN``; never logged, never written
#: to the Volume, never serialized into an artifact.
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

#: Where the Volume is mounted inside every container.
VOL_MOUNT = "/vol"
VOLUMES = {VOL_MOUNT: volume}

#: The only GPU this project uses. Escalation requires explicit human approval.
DEFAULT_GPU = "L4"

#: GPUs that must never be requested without a human decision, because they
#: cost several times an L4 per hour.
FORBIDDEN_GPUS = frozenset({"A100", "A100-40GB", "A100-80GB", "H100", "H200", "B200", "L40S"})

#: Above this, an extraction run needs explicit approval (see project budget).
MAX_UNAPPROVED_TOKENS = 50_000

# -- standard function keyword arguments --------------------------------------

#: CPU work: preprocessing, analysis, reports, publishing. No GPU cost at all.
CPU_DEFAULTS: dict = {
    "image": CPU_IMAGE,
    "volumes": VOLUMES,
    "timeout": 60 * 30,
    "retries": 0,
}

#: GPU work. One L4, no automatic retries, aggressive scaledown.
GPU_DEFAULTS: dict = {
    "image": ML_IMAGE,
    "volumes": VOLUMES,
    "gpu": DEFAULT_GPU,
    "timeout": 60 * 45,
    "retries": 0,
    "scaledown_window": 60,
}


def gpu_kwargs(
    *,
    gpu: str = DEFAULT_GPU,
    timeout: int = 60 * 45,
    secrets: bool = False,
    **extra,
) -> dict:
    """Build ``@app.function`` kwargs for a GPU function, with cost guards.

    Raises
    ------
    ValueError
        If an expensive GPU is requested, or more than one is asked for. Both
        are budget decisions that belong to a human, not to code.
    """
    normalized = gpu.split(":")[0].upper()
    if normalized in FORBIDDEN_GPUS:
        raise ValueError(
            f"GPU {gpu!r} is not permitted under the project budget. "
            f"Use {DEFAULT_GPU!r}, or obtain explicit approval to escalate."
        )
    if ":" in gpu and int(gpu.split(":", 1)[1]) > 1:
        raise ValueError(f"multi-GPU request {gpu!r} is not permitted under the project budget")

    kwargs = dict(GPU_DEFAULTS)
    kwargs["gpu"] = gpu
    kwargs["timeout"] = timeout
    if secrets:
        kwargs["secrets"] = [hf_secret]
    kwargs.update(extra)
    return kwargs


def cpu_kwargs(*, timeout: int = 60 * 30, secrets: bool = False, **extra) -> dict:
    """Build ``@app.function`` kwargs for a CPU function."""
    kwargs = dict(CPU_DEFAULTS)
    kwargs["timeout"] = timeout
    if secrets:
        kwargs["secrets"] = [hf_secret]
    kwargs.update(extra)
    return kwargs


def assert_token_budget(num_tokens: int, *, approved: bool = False) -> None:
    """Refuse an extraction larger than the unapproved ceiling.

    Parameters
    ----------
    num_tokens:
        Activation tokens the run intends to collect.
    approved:
        Set True only when a human has explicitly authorised the larger run.

    Raises
    ------
    ValueError
        If the run exceeds :data:`MAX_UNAPPROVED_TOKENS` without approval.
    """
    if num_tokens > MAX_UNAPPROVED_TOKENS and not approved:
        raise ValueError(
            f"requested {num_tokens:,} activation tokens, which exceeds the "
            f"unapproved ceiling of {MAX_UNAPPROVED_TOKENS:,}. "
            "Re-run with approved=True only after a human has authorised the spend."
        )
