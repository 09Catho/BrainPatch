"""Backend discovery, selection, and the ``doctor`` report.

Every backend is imported lazily and probed defensively: ``brainpatch doctor``
has to work on a machine with none of the engines installed, and its whole job
is to say *which* are missing and what to do about it.
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass
from typing import Any

from brainpatch.runtime.base import BrainPatchBackend
from brainpatch.runtime.capabilities import Capabilities

#: backend name -> module path. Order is preference order for ``backend="auto"``.
BACKEND_MODULES: dict[str, str] = {
    "transformers": "brainpatch.backends.transformers_backend",
    "llamacpp": "brainpatch.backends.llamacpp",
    "vllm": "brainpatch.backends.vllm_backend",
    "mlx": "brainpatch.backends.mlx_backend",
}

#: Friendly aliases users actually type.
BACKEND_ALIASES: dict[str, str] = {
    "hf": "transformers",
    "torch": "transformers",
    "pytorch": "transformers",
    "llama.cpp": "llamacpp",
    "llama_cpp": "llamacpp",
    "gguf": "llamacpp",
    "mlx-lm": "mlx",
    "mlx_lm": "mlx",
}


class BackendNotAvailable(RuntimeError):
    """The requested backend cannot run in this environment."""


def normalize_backend_name(name: str) -> str:
    key = name.strip().lower()
    return BACKEND_ALIASES.get(key, key)


def backend_class(name: str) -> type[BrainPatchBackend]:
    """Import and return a backend class by name."""
    key = normalize_backend_name(name)
    if key not in BACKEND_MODULES:
        raise BackendNotAvailable(
            f"unknown backend {name!r}. Known: {', '.join(sorted(BACKEND_MODULES))}"
        )
    module = importlib.import_module(BACKEND_MODULES[key])
    cls = getattr(module, "BACKEND", None)
    if cls is None:  # pragma: no cover - guards a malformed backend module
        raise BackendNotAvailable(f"backend module for {key!r} exposes no BACKEND class")
    return cls


@dataclass
class BackendStatus:
    """One row of the doctor report."""

    name: str
    available: bool
    detail: str
    capabilities: Capabilities | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "available": self.available,
            "detail": self.detail,
            "capabilities": self.capabilities.to_dict() if self.capabilities else None,
        }


def probe_backend(name: str) -> BackendStatus:
    """Check one backend without raising, whatever is or is not installed."""
    key = normalize_backend_name(name)
    try:
        cls = backend_class(key)
    except BackendNotAvailable as exc:
        return BackendStatus(name=key, available=False, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - a broken backend must not break doctor
        return BackendStatus(name=key, available=False, detail=f"import failed: {exc}")

    try:
        available, detail = cls.is_available()
    except Exception as exc:  # noqa: BLE001
        return BackendStatus(name=key, available=False, detail=f"probe failed: {exc}")

    try:
        caps = cls.capabilities()
    except Exception:  # noqa: BLE001
        caps = None
    return BackendStatus(name=key, available=available, detail=detail, capabilities=caps)


def available_backends() -> list[BackendStatus]:
    """Probe every known backend, in preference order."""
    return [probe_backend(name) for name in BACKEND_MODULES]


def select_backend(preferred: str = "auto") -> type[BrainPatchBackend]:
    """Resolve ``preferred`` to a usable backend class.

    ``"auto"`` picks the first available backend in preference order. An
    explicitly named backend that is unavailable raises with the reason, rather
    than silently falling back to a different engine -- a silent substitution
    would make "I tested it on vLLM" untrue.
    """
    if preferred != "auto":
        key = normalize_backend_name(preferred)
        cls = backend_class(key)
        ok, detail = cls.is_available()
        if not ok:
            raise BackendNotAvailable(f"backend {key!r} is not available: {detail}")
        return cls

    problems: list[str] = []
    for name in BACKEND_MODULES:
        status = probe_backend(name)
        if status.available:
            return backend_class(name)
        problems.append(f"  {name}: {status.detail}")
    raise BackendNotAvailable(
        "no inference backend is available. Install one:\n"
        "  pip install 'brainpatch[transformers]'\n\n"
        "Probed:\n" + "\n".join(problems)
    )


def environment_report() -> dict[str, Any]:
    """Everything ``brainpatch doctor`` prints."""
    from brainpatch import __version__
    from brainpatch.patch.registry import default_registry

    registry = default_registry()
    try:
        installed = [p.name for p in registry.list_patches()]
    except OSError:
        installed = []

    return {
        "brainpatch_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "is_apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
        "registry_home": str(registry.home),
        "installed_patches": installed,
        "backends": [s.to_dict() for s in available_backends()],
    }
