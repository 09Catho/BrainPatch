"""llama.cpp backend, via upstream control vectors.

Approach
--------
llama.cpp already implements exactly the operation a BrainPatch performs: add a
per-layer direction to the residual stream. It exposes this as *control
vectors*, with CLI flags on ``llama-cli`` and ``llama-server``::

    --control-vector-scaled FILE SCALE
    --control-vector-layer-range START END

So this backend does **not** fork llama.cpp or reimplement inference. It
compiles the patch to a control-vector GGUF (see
:func:`brainpatch.patch.compiler.export_llamacpp_control_vector`) and drives the
upstream binary. Upstream stays upstream; BrainPatch supplies the vector.

Two details that are easy to get silently wrong
-----------------------------------------------
**Layer indexing.** BrainPatch layers are 0-based decoder blocks. llama.cpp
control-vector tensors are named ``direction.N`` with **1-based** N, and
``--control-vector-layer-range`` is also 1-based and inclusive. The exporter
does the ``+1``; this backend passes the matching range so a single-layer patch
applies to that layer and not to every layer.

**Quantization.** A direction fitted on bf16 activations is not guaranteed to
behave identically on a Q4 model. Nothing here assumes it does, and the
capability table reports only quantizations that were actually exercised.

Known capability gap
--------------------
Token-level schedules are not supported. A control vector is bound for the whole
run; changing it between decode steps would need a persistent libllama process
with per-step control, which the CLI does not expose. This is reported as
``dynamic_schedule=False`` rather than emulated badly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator

from brainpatch.patch.validation import ModelDescriptor
from brainpatch.runtime.base import BrainPatchBackend, GenerationConfig
from brainpatch.runtime.capabilities import Capabilities

#: Binaries searched on PATH, plus ``BRAINPATCH_LLAMACPP_BIN`` if set.
CLI_NAMES = ("llama-cli", "llama")
SERVER_NAMES = ("llama-server",)
ENV_BIN = "BRAINPATCH_LLAMACPP_BIN"


def _find_binary(names: tuple[str, ...]) -> str | None:
    override = os.environ.get(ENV_BIN)
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return str(candidate)
        if candidate.is_dir():
            for name in names:
                found = candidate / name
                if found.is_file():
                    return str(found)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


class LlamaCppBackend(BrainPatchBackend):
    """Run GGUF models through upstream llama.cpp with a compiled control vector."""

    name = "llamacpp"

    def __init__(self) -> None:
        super().__init__()
        self.model_path: Path | None = None
        self.binary: str | None = None
        self.n_gpu_layers: int = 0
        self.extra_args: list[str] = []
        self._gguf_meta: dict[str, Any] = {}
        self._vector_files: dict[str, Path] = {}
        self._tmpdir: tempfile.TemporaryDirectory | None = None

    # -- availability ----------------------------------------------------------

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        binary = _find_binary(CLI_NAMES)
        if binary is None:
            return False, (
                "llama-cli not found on PATH -- install llama.cpp, or set "
                f"{ENV_BIN} to its binary or bin directory"
            )
        try:
            out = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=20
            )
            version = (out.stderr or out.stdout).strip().splitlines()
            detail = version[0] if version else "unknown version"
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"llama-cli found at {binary} but could not run: {exc}"
        return True, f"{binary} ({detail})"

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            name=cls.name,
            static_intervention=True,
            dynamic_schedule=False,
            multiple_patches=True,
            streaming=True,
            cpu=True,
            cuda=True,
            apple_silicon=True,
            server=True,
            concurrent_requests=True,
            per_request_strength=False,
            quantization=(),
            notes={
                "dynamic_schedule": (
                    "A control vector is bound for the whole run; llama.cpp's CLI "
                    "exposes no per-decode-step control. Use the transformers "
                    "backend for token-level schedules."
                ),
                "per_request_strength": (
                    "Control-vector scale is fixed at process start, so it is "
                    "server-wide rather than per request."
                ),
                "quantization": (
                    "GGUF quantizations load, but a direction fitted on bf16 is not "
                    "guaranteed to behave identically on Q4. Verified quantizations "
                    "are listed per patch in its compatibility block."
                ),
            },
        )

    # -- model -----------------------------------------------------------------

    def load_model(
        self,
        model: str,
        *,
        n_gpu_layers: int = 0,
        extra_args: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Register a GGUF path. llama.cpp loads per invocation, not here."""
        path = Path(model).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GGUF model not found: {path}")
        binary = _find_binary(CLI_NAMES)
        if binary is None:
            raise RuntimeError("llama-cli is not available; run `brainpatch doctor`")

        self.model_path = path
        self.binary = binary
        self.n_gpu_layers = n_gpu_layers
        self.extra_args = list(extra_args or [])
        self._gguf_meta = read_gguf_metadata(path)
        self._tmpdir = tempfile.TemporaryDirectory(prefix="brainpatch-cv-")

    def describe_model(self) -> ModelDescriptor:
        if self.model_path is None:
            raise RuntimeError("no model loaded; call load_model() first")
        meta = self._gguf_meta
        return ModelDescriptor(
            model_id=meta.get("model_id") or self.model_path.stem,
            hidden_size=int(meta.get("hidden_size", 0)),
            num_layers=int(meta.get("num_layers", 0)),
            architecture=str(meta.get("architecture", "")),
            revision=None,
        )

    # -- control vectors -------------------------------------------------------

    def _on_patches_changed(self) -> None:
        self._vector_files.clear()

    def _control_vector_args(self) -> list[str]:
        """Build ``--control-vector-scaled`` args for every enabled patch."""
        from brainpatch.patch.compiler import export_llamacpp_control_vector

        if self._tmpdir is None:
            raise RuntimeError("no model loaded")

        args: list[str] = []
        layers: set[int] = set()
        for name, active in self.patches.items():
            multiplier = active.multiplier_at(0)
            if multiplier == 0.0:
                continue  # disabled or zeroed: emit nothing at all
            path = self._vector_files.get(name)
            if path is None:
                if active.patch.source is None:
                    raise RuntimeError(
                        f"patch {name!r} has no source file; llama.cpp export needs one"
                    )
                path = Path(self._tmpdir.name) / f"{name}.gguf"
                export_llamacpp_control_vector(active.patch.source, path, strength=1.0)
                self._vector_files[name] = path
            args += ["--control-vector-scaled", str(path), f"{multiplier:.6f}"]
            layers.update(active.manifest.layers)

        if layers:
            # 1-based inclusive, matching the exporter's direction.N naming.
            args += [
                "--control-vector-layer-range",
                str(min(layers) + 1),
                str(max(layers) + 1),
            ]
        return args

    # -- generation ------------------------------------------------------------

    def _build_command(self, prompt: str, cfg: GenerationConfig) -> list[str]:
        assert self.binary and self.model_path
        cmd = [
            self.binary,
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(cfg.max_new_tokens),
            "-ngl", str(self.n_gpu_layers),
            "--no-display-prompt",
            "-no-cnv",
        ]
        if cfg.temperature <= 0:
            cmd += ["--temp", "0"]
        else:
            cmd += ["--temp", str(cfg.temperature), "--top-p", str(cfg.top_p), "-s", str(cfg.seed)]
            if cfg.top_k > 0:
                cmd += ["--top-k", str(cfg.top_k)]
        cmd += self._control_vector_args()
        cmd += self.extra_args
        return cmd

    def generate(
        self, prompt: str, config: GenerationConfig | None = None, *, timeout: int = 600, **kwargs: Any
    ) -> str:
        if self.model_path is None:
            raise RuntimeError("no model loaded; call load_model() first")
        cfg = config or GenerationConfig()
        cmd = self._build_command(prompt, cfg)
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"llama.cpp timed out after {timeout}s") from exc
        if out.returncode != 0:
            raise RuntimeError(
                f"llama-cli exited {out.returncode}:\n{out.stderr[-2000:]}"
            )
        return out.stdout.strip()

    def stream(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any
    ) -> Iterator[str]:
        if self.model_path is None:
            raise RuntimeError("no model loaded; call load_model() first")
        cfg = config or GenerationConfig()
        cmd = self._build_command(prompt, cfg)
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                yield line
        finally:
            process.terminate()
            process.wait(timeout=10)

    def server_command(self, *, port: int = 8000, host: str = "127.0.0.1") -> list[str]:
        """Full ``llama-server`` command line, control vectors included."""
        binary = _find_binary(SERVER_NAMES)
        if binary is None:
            raise RuntimeError("llama-server not found on PATH")
        if self.model_path is None:
            raise RuntimeError("no model loaded")
        return [
            binary,
            "-m", str(self.model_path),
            "--host", host,
            "--port", str(port),
            "-ngl", str(self.n_gpu_layers),
            *self._control_vector_args(),
            *self.extra_args,
        ]

    def unload(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
        self._vector_files.clear()


def read_gguf_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read architecture facts from a GGUF header.

    Uses the ``gguf`` package when present. Returns ``{}`` rather than raising
    when it is not, so a missing optional dependency degrades to "unknown model
    geometry" instead of blocking the backend.
    """
    try:
        import gguf
    except ModuleNotFoundError:
        return {}

    try:
        reader = gguf.GGUFReader(str(path), "r")
    except Exception:  # noqa: BLE001 - a malformed GGUF must not crash doctor
        return {}

    fields = reader.fields
    arch = _gguf_str(fields.get("general.architecture"))
    meta: dict[str, Any] = {
        "architecture": arch or "",
        "model_id": _gguf_str(fields.get("general.name")) or "",
    }
    if arch:
        meta["hidden_size"] = _gguf_int(fields.get(f"{arch}.embedding_length")) or 0
        meta["num_layers"] = _gguf_int(fields.get(f"{arch}.block_count")) or 0
    meta["quantization"] = _gguf_int(fields.get("general.file_type"))
    return meta


def _gguf_str(field: Any) -> str | None:
    if field is None or not getattr(field, "parts", None):
        return None
    try:
        return str(bytes(field.parts[field.data[0]]), encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None


def _gguf_int(field: Any) -> int | None:
    if field is None or not getattr(field, "parts", None):
        return None
    try:
        return int(field.parts[field.data[0]][0])
    except Exception:  # noqa: BLE001
        return None


def describe_export(patch_path: str | os.PathLike[str]) -> str:
    """Human-readable summary of what an export would produce."""
    from brainpatch.patch.loader import load_patch

    loaded = load_patch(patch_path)
    layers = loaded.manifest.layers
    return json.dumps(
        {
            "patch": loaded.manifest.name,
            "brainpatch_layers_0based": layers,
            "llamacpp_direction_indices_1based": [layer + 1 for layer in layers],
            "layer_range_flag": ["--control-vector-layer-range", min(layers) + 1, max(layers) + 1],
            "hidden_size": loaded.manifest.base_model.hidden_size,
        },
        indent=2,
    )


BACKEND = LlamaCppBackend
