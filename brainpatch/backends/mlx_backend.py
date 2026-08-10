"""MLX-LM backend for Apple Silicon.

Status: **experimental, unverified on hardware.**

This adapter is implemented against the MLX-LM API but has **not** been executed
on an Apple Silicon machine, because none was available during development and
Modal offers no Apple runners. It is shipped as ``experimental`` rather than
advertised as supported, and :meth:`MLXBackend.capabilities` says so in the
notes that ``brainpatch doctor`` prints.

If you run it on real hardware, the integration checklist is in
``integration_tests/README.md``; a report either way is genuinely useful.

Implementation note
-------------------
MLX has no ``register_forward_hook``. Instead this wraps the target decoder
block's ``__call__`` with a closure that adds the delta to the block's output --
the same intervention, expressed the way MLX modules allow. The original method
is restored on removal, so the model object is left as it was found.
"""

from __future__ import annotations

import platform
from typing import Any

from brainpatch.patch.validation import ModelDescriptor
from brainpatch.runtime.base import BrainPatchBackend, GenerationConfig
from brainpatch.runtime.capabilities import Capabilities

_LAYER_PATHS = ("model.layers", "layers", "transformer.h")


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


class MLXBackend(BrainPatchBackend):
    """Apply BrainPatches to an MLX-LM model on Apple Silicon."""

    name = "mlx"

    def __init__(self) -> None:
        super().__init__()
        self.model: Any = None
        self.tokenizer: Any = None
        self.model_id: str = ""
        self._originals: dict[int, Any] = {}
        self._vector_cache: dict[tuple[str, str], Any] = {}

    # -- availability ----------------------------------------------------------

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if not _is_apple_silicon():
            return False, "requires Apple Silicon (darwin/arm64)"
        try:
            import mlx_lm  # noqa: F401
        except ModuleNotFoundError:
            return False, "mlx-lm not installed -- pip install 'brainpatch[mlx]'"
        try:
            import mlx.core as mx

            return True, f"mlx-lm on {mx.default_device()}"
        except Exception as exc:  # noqa: BLE001
            return False, f"mlx present but unusable: {exc}"

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            name=cls.name,
            static_intervention=True,
            dynamic_schedule=False,
            multiple_patches=True,
            streaming=True,
            cpu=False,
            cuda=False,
            apple_silicon=True,
            server=False,
            concurrent_requests=False,
            per_request_strength=False,
            quantization=(),
            notes={
                "static_intervention": (
                    "EXPERIMENTAL: implemented against the MLX-LM API but never "
                    "executed on Apple Silicon hardware. Treat as unverified."
                ),
                "dynamic_schedule": "Not implemented; would need per-step control in the decode loop.",
                "server": "Not implemented for this backend.",
            },
        )

    # -- model -----------------------------------------------------------------

    def load_model(self, model: str, **kwargs: Any) -> None:
        if not _is_apple_silicon():
            raise RuntimeError(
                "the MLX backend requires Apple Silicon. On other hardware use "
                "--backend transformers or --backend llamacpp."
            )
        from mlx_lm import load

        self.model, self.tokenizer = load(model, **kwargs)
        self.model_id = model
        self._vector_cache.clear()

    def _decoder_layers(self) -> Any:
        for path in _LAYER_PATHS:
            node: Any = self.model
            for part in path.split("."):
                node = getattr(node, part, None)
                if node is None:
                    break
            if node is not None and hasattr(node, "__len__") and len(node) > 0:
                return node
        raise RuntimeError(f"could not locate decoder blocks on {type(self.model).__name__}")

    def describe_model(self) -> ModelDescriptor:
        if self.model is None:
            raise RuntimeError("no model loaded; call load_model() first")
        args = getattr(self.model, "args", None)
        hidden = int(getattr(args, "hidden_size", 0)) if args else 0
        return ModelDescriptor(
            model_id=self.model_id,
            hidden_size=hidden,
            num_layers=len(self._decoder_layers()),
            architecture=str(getattr(args, "model_type", "")) if args else "",
        )

    # -- intervention ----------------------------------------------------------

    def _array_for(self, patch_name: str, key: str) -> Any:
        cache_key = (patch_name, key)
        cached = self._vector_cache.get(cache_key)
        if cached is None:
            import mlx.core as mx

            cached = mx.array(self.vector_values(patch_name, key))
            self._vector_cache[cache_key] = cached
        return cached

    def _on_patches_changed(self) -> None:
        self._vector_cache.clear()
        if self.model is not None:
            self._install_wrappers()

    def _install_wrappers(self) -> None:
        """Wrap each patched block's ``__call__`` to add the delta."""
        self._remove_wrappers()
        layers = self._decoder_layers()
        for layer_index in sorted(
            {i.layer for p in self.patches.values() for i in p.manifest.interventions}
        ):
            if layer_index >= len(layers):
                continue
            block = layers[layer_index]
            original = block.__call__
            self._originals[layer_index] = original
            block.__call__ = self._make_wrapper(layer_index, original)  # type: ignore[method-assign]

    def _make_wrapper(self, layer_index: int, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            output = original(*args, **kwargs)
            edits = self.resolve_edits(0, layer=layer_index)
            if not edits:
                return output
            delta = None
            for edit in edits:
                contribution = self._array_for(edit.patch_name, edit.vector_key) * edit.coefficient
                delta = contribution if delta is None else delta + contribution
            if isinstance(output, tuple):
                return (output[0] + delta, *output[1:])
            return output + delta

        return wrapped

    def _remove_wrappers(self) -> None:
        if self.model is None:
            self._originals.clear()
            return
        layers = self._decoder_layers()
        for layer_index, original in self._originals.items():
            if layer_index < len(layers):
                layers[layer_index].__call__ = original  # type: ignore[method-assign]
        self._originals.clear()

    # -- generation ------------------------------------------------------------

    def generate(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any
    ) -> str:
        from mlx_lm import generate as mlx_generate

        if self.model is None:
            raise RuntimeError("no model loaded; call load_model() first")
        cfg = config or GenerationConfig()
        return mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=cfg.max_new_tokens,
            verbose=False,
            **kwargs,
        )

    def unload(self) -> None:
        self._remove_wrappers()
        self._vector_cache.clear()
        self.model = None
        self.tokenizer = None


BACKEND = MLXBackend
