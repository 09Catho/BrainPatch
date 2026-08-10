"""vLLM backend.

What this actually does, precisely
----------------------------------
It loads the model **through vLLM** and registers forward hooks on the decoder
blocks of the model object vLLM itself instantiated. The intervention therefore
runs inside vLLM's own forward pass, under vLLM's scheduler, batching and KV
cache. It does **not** launch a shadow Transformers model and pretend -- that
would make "verified on vLLM" a false statement.

The honest caveat: reaching the model object requires vLLM internals
(``llm_engine.model_executor.driver_worker.model_runner.model``). vLLM does not
currently expose a public activation-hook API, so this path is version-sensitive
and is probed defensively at load time with a clear error if the internals move.
That is a real fragility and is documented rather than hidden.

Request isolation
-----------------
Patch state is **immutable while serving**. Strength is fixed when the engine is
configured, so every request in a batch sees identical model behaviour and no
state can leak between concurrent users. Per-request strength is deliberately
*not* offered: with continuous batching, requests share one forward pass, so a
per-request coefficient would require per-sequence scaling inside the hook and
vLLM exposes no supported way to attribute rows of a batch to requests. Claiming
it without that would be a correctness bug affecting other users' outputs.
"""

from __future__ import annotations

import threading
from typing import Any, Iterator

from brainpatch.patch.validation import ModelDescriptor
from brainpatch.runtime.base import BrainPatchBackend, GenerationConfig
from brainpatch.runtime.capabilities import Capabilities

#: Internal paths tried, in order, to reach the torch model vLLM loaded.
_MODEL_PATHS = (
    "llm_engine.model_executor.driver_worker.model_runner.model",
    "llm_engine.model_executor.driver_worker.worker.model_runner.model",
    "llm_engine.model_executor.driver_worker.model_runner.model.model",
)

_LAYER_PATHS = ("model.layers", "layers", "transformer.h", "model.decoder.layers")


class VLLMBackend(BrainPatchBackend):
    """Apply BrainPatches inside vLLM's inference path."""

    name = "vllm"

    def __init__(self) -> None:
        super().__init__()
        self.llm: Any = None
        self.model_id: str = ""
        self.revision: str | None = None
        self._torch_model: Any = None
        self._handles: list[Any] = []
        self._vector_cache: dict[tuple[str, str], Any] = {}
        self._config: Any = None
        #: Guards patch mutation against in-flight requests.
        self._lock = threading.RLock()
        self._serving = False

    # -- availability ----------------------------------------------------------

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import vllm
        except ModuleNotFoundError:
            return False, "vLLM not installed -- pip install 'brainpatch[vllm]'"
        try:
            import torch

            if not torch.cuda.is_available():
                return False, f"vLLM {vllm.__version__} installed but no CUDA device is visible"
            return True, f"vLLM {vllm.__version__}, CUDA: {torch.cuda.get_device_name(0)}"
        except Exception as exc:  # noqa: BLE001
            return False, f"vLLM present but unusable: {exc}"

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            name=cls.name,
            static_intervention=True,
            dynamic_schedule=False,
            multiple_patches=True,
            streaming=False,
            cpu=False,
            cuda=True,
            server=True,
            concurrent_requests=True,
            per_request_strength=False,
            quantization=(),
            notes={
                "dynamic_schedule": (
                    "Continuous batching means one forward pass serves many sequences "
                    "at different generation positions, so a single token index is not "
                    "well defined. Use the transformers backend for schedules."
                ),
                "per_request_strength": (
                    "Patch state is fixed at engine configuration time so concurrent "
                    "requests cannot affect each other. Per-request strength would "
                    "need per-sequence scaling inside the batched forward pass, which "
                    "vLLM exposes no supported way to attribute."
                ),
                "concurrent_requests": (
                    "Safe because patch state is immutable while serving; mutation "
                    "raises if attempted mid-serve."
                ),
                "cpu": "vLLM requires a CUDA device for the supported path here.",
            },
        )

    # -- model -----------------------------------------------------------------

    def load_model(
        self,
        model: str,
        *,
        revision: str | None = None,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        enforce_eager: bool = True,
        **kwargs: Any,
    ) -> None:
        """Load through vLLM and locate the decoder blocks it instantiated.

        ``enforce_eager=True`` by default: CUDA graph capture replays a recorded
        graph, and a Python forward hook added afterwards would not participate.
        Eager mode costs some throughput and is what makes the intervention
        actually run. Override only if you have verified hook execution.
        """
        from vllm import LLM

        self.llm = LLM(
            model=model,
            revision=revision,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            **kwargs,
        )
        self.model_id = model
        self.revision = revision
        self._torch_model = self._locate_model()
        self._config = getattr(self._torch_model, "config", None)
        self._install_hooks()

    def _locate_model(self) -> Any:
        """Reach vLLM's instantiated torch model, with a clear error if moved."""
        tried: list[str] = []
        for path in _MODEL_PATHS:
            node: Any = self.llm
            ok = True
            for part in path.split("."):
                node = getattr(node, part, None)
                if node is None:
                    ok = False
                    break
            tried.append(path)
            if ok and node is not None:
                return node
        raise RuntimeError(
            "could not reach vLLM's internal model object. vLLM does not expose a "
            "public activation-hook API, so this backend depends on internals that "
            f"appear to have changed in this version. Tried: {tried}"
        )

    def _decoder_layers(self) -> Any:
        import torch.nn as nn

        for path in _LAYER_PATHS:
            node: Any = self._torch_model
            for part in path.split("."):
                node = getattr(node, part, None)
                if node is None:
                    break
            if isinstance(node, nn.ModuleList) and len(node) > 0:
                return node
        raise RuntimeError(
            f"could not locate decoder blocks on vLLM model {type(self._torch_model).__name__}"
        )

    def describe_model(self) -> ModelDescriptor:
        if self.llm is None:
            raise RuntimeError("no model loaded; call load_model() first")
        config = self._config
        hidden = int(getattr(config, "hidden_size", 0)) if config else 0
        archs = list(getattr(config, "architectures", []) or []) if config else []
        return ModelDescriptor(
            model_id=self.model_id,
            hidden_size=hidden,
            num_layers=len(self._decoder_layers()),
            architecture=archs[0] if archs else type(self._torch_model).__name__,
            revision=self.revision,
        )

    # -- intervention ----------------------------------------------------------

    def _tensor_for(self, patch_name: str, key: str) -> Any:
        cache_key = (patch_name, key)
        cached = self._vector_cache.get(cache_key)
        if cached is None:
            import torch

            device = next(self._torch_model.parameters()).device
            cached = torch.tensor(
                self.vector_values(patch_name, key), dtype=torch.float32, device=device
            )
            self._vector_cache[cache_key] = cached
        return cached

    def _on_patches_changed(self) -> None:
        if self._serving:
            raise RuntimeError(
                "refusing to mutate patch state while the vLLM server is running: "
                "in-flight batched requests would observe an inconsistent model. "
                "Restart the server to change patches."
            )
        self._vector_cache.clear()
        if self.llm is not None:
            self._install_hooks()

    def _install_hooks(self) -> None:
        """One hook per patched layer, applied to every position in the batch."""
        self._remove_hooks()
        if self._torch_model is None:
            return
        layers = self._decoder_layers()
        for layer_index in sorted(
            {i.layer for p in self.patches.values() for i in p.manifest.interventions}
        ):
            if layer_index >= len(layers):
                continue
            self._handles.append(
                layers[layer_index].register_forward_hook(self._make_hook(layer_index))
            )

    def _make_hook(self, layer_index: int) -> Any:
        def hook(module: Any, args: Any, output: Any) -> Any:
            # token_index 0: schedules are unsupported here, so the coefficient
            # is constant for every position in the batch. That constancy is
            # exactly what makes concurrent batching safe.
            edits = self.resolve_edits(0, layer=layer_index)
            if not edits:
                return output

            import torch

            delta: Any = None
            for edit in edits:
                contribution = self._tensor_for(edit.patch_name, edit.vector_key) * edit.coefficient
                delta = contribution if delta is None else delta + contribution

            if isinstance(output, tuple):
                hidden = output[0]
                if not isinstance(hidden, torch.Tensor):
                    return output
                return (hidden + delta.to(hidden.dtype), *output[1:])
            if isinstance(output, torch.Tensor):
                return output + delta.to(output.dtype)
            return output

        return hook

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # -- generation ------------------------------------------------------------

    def generate(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any
    ) -> str:
        results = self.generate_batch([prompt], config, **kwargs)
        return results[0]

    def generate_batch(
        self, prompts: list[str], config: GenerationConfig | None = None, **kwargs: Any
    ) -> list[str]:
        """Batched generation -- the reason to use vLLM at all."""
        from vllm import SamplingParams

        if self.llm is None:
            raise RuntimeError("no model loaded; call load_model() first")
        cfg = config or GenerationConfig()
        params = SamplingParams(
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k if cfg.top_k > 0 else -1,
            repetition_penalty=cfg.repetition_penalty,
            seed=cfg.seed if cfg.do_sample else None,
            stop=cfg.stop or None,
        )
        with self._lock:
            outputs = self.llm.generate(prompts, params, **kwargs)
        return [o.outputs[0].text for o in outputs]

    def stream(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any
    ) -> Iterator[str]:
        self.capabilities().require("streaming")
        yield self.generate(prompt, config, **kwargs)  # pragma: no cover

    # -- serving ---------------------------------------------------------------

    def begin_serving(self) -> None:
        """Freeze patch state for the lifetime of a server."""
        self._serving = True

    def end_serving(self) -> None:
        self._serving = False

    def unload(self) -> None:
        self._remove_hooks()
        self._vector_cache.clear()
        self.llm = None
        self._torch_model = None


BACKEND = VLLMBackend
