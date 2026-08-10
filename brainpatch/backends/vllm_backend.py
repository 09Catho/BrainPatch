"""vLLM backend.

How the intervention reaches vLLM
---------------------------------
vLLM's V1 engine runs the model in a **separate worker process**, so reaching it
by walking attributes off the ``LLM`` object does not work -- the model simply
is not in the caller's address space. The supported way in is
``worker_extension_cls`` plus ``LLM.collective_rpc``: vLLM mixes
:class:`~brainpatch.backends.vllm_worker.BrainPatchWorkerExtension` into every
worker, and the RPC then invokes its methods **by name**.

Calling by name matters. V1's RPC channel serializes with msgpack and refuses to
ship a callable, suggesting ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` instead --
which would turn the engine's control channel into an arbitrary-code path for
our convenience. The extension class avoids that entirely.

The consequence is that the intervention genuinely runs inside vLLM's forward
pass, under its scheduler, batching and KV cache. There is no shadow
Transformers model anywhere in this file.

The RPC payload is a plain ``{layer: [floats]}`` map of *already-scaled* deltas.
That suffices precisely because this backend supports neither token schedules
nor per-request strength, so the delta is constant for a whole forward pass.

Request isolation
-----------------
Patch state is fixed while serving. With continuous batching one forward pass
serves many sequences, so a per-request coefficient would alter *other users'*
output; the backend therefore refuses to mutate patches mid-serve rather than
offering an unsafe knob. Two concurrent requests provably see identical model
behaviour, which is what the integration test checks.
"""

from __future__ import annotations

import threading
from typing import Any, Iterator

from brainpatch.patch.validation import ModelDescriptor
from brainpatch.runtime.base import BrainPatchBackend, GenerationConfig
from brainpatch.runtime.capabilities import Capabilities

#: Import path vLLM loads into each worker process.
WORKER_EXTENSION = "brainpatch.backends.vllm_worker.BrainPatchWorkerExtension"


class VLLMBackend(BrainPatchBackend):
    """Apply BrainPatches inside vLLM's inference path."""

    name = "vllm"

    def __init__(self) -> None:
        super().__init__()
        self.llm: Any = None
        self.model_id: str = ""
        self.revision: str | None = None
        self.vllm_version: str = ""
        self._geometry: dict[str, Any] = {}
        self._last_rpc: list[dict[str, Any]] = []
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
                    "Continuous batching means one forward pass serves sequences at "
                    "different generation positions, so a single token index is not "
                    "well defined. Use the transformers backend for schedules."
                ),
                "per_request_strength": (
                    "Would require per-sequence scaling inside a batched forward pass; "
                    "vLLM exposes no supported way to attribute rows of a batch to "
                    "requests, so offering it would corrupt other users' output."
                ),
                "concurrent_requests": (
                    "Safe because patch state is immutable while serving; mutation "
                    "raises if attempted mid-serve."
                ),
                "cpu": "This backend requires a CUDA device.",
            },
        )

    # -- model -----------------------------------------------------------------

    def load_model(
        self,
        model: str,
        *,
        revision: str | None = None,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.80,
        max_model_len: int | None = 2048,
        enforce_eager: bool = True,
        **kwargs: Any,
    ) -> None:
        """Load through vLLM.

        ``enforce_eager=True`` by default: CUDA graph capture replays a recorded
        graph, and a Python forward hook registered afterwards would not
        participate. Eager costs throughput and is what makes the intervention
        actually execute. Do not disable it without re-verifying that hooks run.
        """
        import vllm
        from vllm import LLM

        self.vllm_version = vllm.__version__
        # Supported extension point: vLLM mixes this class into each worker, so
        # collective_rpc can call its methods BY NAME with msgpack-safe args.
        # Passing a callable instead would require
        # VLLM_ALLOW_INSECURE_SERIALIZATION=1, which we deliberately do not use.
        self.llm = LLM(
            model=model,
            worker_extension_cls=WORKER_EXTENSION,
            revision=revision,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            **kwargs,
        )
        self.model_id = model
        self.revision = revision
        self._geometry = self._rpc("bp_probe")[0]

    def _rpc(self, method: str, *args: Any) -> list[dict[str, Any]]:
        """Call a worker-extension method by name in every vLLM worker."""
        if self.llm is None:
            raise RuntimeError("no model loaded; call load_model() first")
        rpc = getattr(self.llm, "collective_rpc", None)
        if rpc is None:
            raise RuntimeError(
                f"vLLM {self.vllm_version} has no LLM.collective_rpc, which this "
                "backend needs to reach the worker process. Supported: vLLM with "
                "collective_rpc and worker_extension_cls (verified on 0.11.0)."
            )
        return list(rpc(method, args=args) if args else rpc(method))

    def describe_model(self) -> ModelDescriptor:
        if self.llm is None:
            raise RuntimeError("no model loaded; call load_model() first")
        geometry = self._geometry
        archs = geometry.get("architectures") or []
        return ModelDescriptor(
            model_id=self.model_id,
            hidden_size=int(geometry.get("hidden_size", 0)),
            num_layers=int(geometry.get("num_layers", 0)),
            architecture=archs[0] if archs else geometry.get("model_class", ""),
            revision=self.revision,
        )

    # -- intervention ----------------------------------------------------------

    def _deltas_by_layer(self) -> dict[int, list[float]]:
        """Collapse all enabled patches into one already-scaled vector per layer."""
        hidden = int(self._geometry.get("hidden_size", 0))
        deltas: dict[int, list[float]] = {}
        for edit in self.resolve_edits(0):
            values = self.vector_values(edit.patch_name, edit.vector_key)
            acc = deltas.setdefault(edit.layer, [0.0] * (hidden or len(values)))
            for i, value in enumerate(values):
                acc[i] += value * edit.coefficient
        return deltas

    def _on_patches_changed(self) -> None:
        if self._serving:
            raise RuntimeError(
                "refusing to mutate patch state while the vLLM server is running: "
                "in-flight batched requests would observe an inconsistent model. "
                "Restart the server to change patches."
            )
        if self.llm is None:
            return
        self._last_rpc = self._rpc("bp_apply_deltas", self._deltas_by_layer())

    @property
    def last_hook_report(self) -> list[dict[str, Any]]:
        """What the workers reported after the last hook installation.

        Exposed so an integration test can *prove* the hooks landed inside vLLM
        rather than inferring it from output changes.
        """
        return list(self._last_rpc)

    def worker_state(self) -> list[dict[str, Any]]:
        """Live probe of every worker, including active hook count."""
        return self._rpc("bp_probe")

    # -- generation ------------------------------------------------------------

    def generate(self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any) -> str:
        return self.generate_batch([prompt], config, **kwargs)[0]

    def generate_batch(
        self,
        prompts: list[str],
        config: GenerationConfig | None = None,
        *,
        use_chat_template: bool = True,
        system: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Batched generation -- the reason to use vLLM at all.

        ``system`` and ``use_chat_template`` are part of the cross-backend
        generate() contract, so they are consumed here rather than forwarded:
        passing them through to ``llm.generate`` raises a TypeError, which is
        how the OpenAI server first failed against this backend.
        """
        from vllm import SamplingParams

        # Only vLLM's own arguments may reach it.
        kwargs.pop("apply_to_prompt", None)

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
        rendered = (
            [self._render(p, system) for p in prompts] if use_chat_template else list(prompts)
        )
        with self._lock:
            outputs = self.llm.generate(rendered, params, **kwargs)
        return [o.outputs[0].text for o in outputs]

    def _render(self, prompt: str, system: str | None = None) -> str:
        tokenizer = self.llm.get_tokenizer()
        if getattr(tokenizer, "chat_template", None) is None:
            return prompt
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def stream(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any
    ) -> Iterator[str]:
        self.capabilities().require("streaming")
        yield ""  # pragma: no cover - unreachable; require() raises

    # -- serving ---------------------------------------------------------------

    def begin_serving(self) -> None:
        """Freeze patch state for the lifetime of a server."""
        self._serving = True

    def end_serving(self) -> None:
        self._serving = False

    def unload(self) -> None:
        if self.llm is not None:
            try:
                self._rpc("bp_apply_deltas", {})
            except Exception:  # noqa: BLE001 - teardown must not mask a real error
                pass
        self.llm = None


BACKEND = VLLMBackend
