"""Hugging Face Transformers backend -- the reference implementation.

Runs anywhere PyTorch runs: CUDA, CPU, or Apple MPS. No Modal, no Volume, no
network beyond whatever ``from_pretrained`` needs to fetch the base model once.

How the intervention works
--------------------------
A forward hook on decoder block *L* adds a vector to its output -- the residual
stream after that block has written to it, which is where the contribution is
visible and from which it propagates to every later block.

Two properties this file exists to guarantee:

**Weights are never modified.** ``requires_grad_(False)`` plus ``eval()``, and
the hook adds to activations rather than folding anything into parameters.
Removing a patch restores the original behaviour exactly, because nothing was
changed to begin with.

**Strength 0 is bit-identical to baseline.** When the resolved edit list is
empty the hook returns the output object untouched -- not ``output + 0.0``. There
is no arithmetic to round, so a zeroed patch and an uninstalled patch produce
the same bytes. :meth:`TransformersBackend.assert_zero_is_baseline` checks this
empirically rather than trusting the argument.
"""

from __future__ import annotations

import platform
from typing import Any, Callable, Iterator

from brainpatch.patch.validation import ModelDescriptor
from brainpatch.runtime.base import BrainPatchBackend, GenerationConfig
from brainpatch.runtime.capabilities import Capabilities

#: Layouts to try when locating the decoder-block list, in order.
_LAYER_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers")


class TransformersBackend(BrainPatchBackend):
    """Apply BrainPatches to a PyTorch Transformers causal LM."""

    name = "transformers"

    def __init__(self) -> None:
        super().__init__()
        self.model: Any = None
        self.tokenizer: Any = None
        self.model_id: str = ""
        self.revision: str | None = None
        self.device: Any = None
        self._handles: list[Any] = []
        self._vector_cache: dict[tuple[str, str], Any] = {}
        self._trace: list[tuple[int, float]] = []
        self._apply_to_prompt = True
        self._first_layer = -1
        self._pass_counter = 0
        self._current_pass = 0

    # -- availability ----------------------------------------------------------

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            return False, "PyTorch not installed -- pip install 'brainpatch[transformers]'"
        try:
            import transformers
        except ModuleNotFoundError:
            return False, "transformers not installed -- pip install 'brainpatch[transformers]'"

        import torch

        bits = [f"transformers {transformers.__version__}", f"torch {torch.__version__}"]
        if torch.cuda.is_available():
            bits.append(f"CUDA: {torch.cuda.get_device_name(0)}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            bits.append("Apple MPS")
        else:
            bits.append("CPU only")
        return True, ", ".join(bits)

    @classmethod
    def capabilities(cls) -> Capabilities:
        cuda = mps = False
        try:
            import torch

            cuda = torch.cuda.is_available()
            mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        except Exception:  # noqa: BLE001 - capabilities must work without torch
            pass
        return Capabilities(
            name=cls.name,
            static_intervention=True,
            dynamic_schedule=True,
            multiple_patches=True,
            streaming=True,
            cpu=True,
            cuda=cuda,
            mps=mps,
            apple_silicon=platform.system() == "Darwin" and platform.machine() == "arm64",
            server=True,
            concurrent_requests=False,
            per_request_strength=False,
            quantization=(),
            notes={
                "concurrent_requests": (
                    "Patch state is per-process and mutable; serve one request at a "
                    "time or use the vLLM backend for concurrency."
                ),
                "quantization": (
                    "bitsandbytes-quantized models load, but no quantization has been "
                    "verified end to end for patch behaviour."
                ),
            },
        )

    # -- model loading ---------------------------------------------------------

    def load_model(
        self,
        model: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> None:
        """Load a frozen causal LM.

        ``device="auto"`` prefers CUDA, then MPS, then CPU. ``dtype="auto"``
        picks bfloat16 on CUDA, float16 on MPS, float32 on CPU -- CPU bf16 is
        widely unsupported and silently slow.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_device = _resolve_device(device)
        torch_dtype = _resolve_dtype(dtype, resolved_device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        self.model.to(resolved_device)
        # Frozen for the process lifetime: every effect comes from activations.
        self.model.eval()
        self.model.requires_grad_(False)

        self.model_id = model
        self.revision = revision or _cached_revision(self.model)
        self.device = resolved_device
        self._vector_cache.clear()

    def describe_model(self) -> ModelDescriptor:
        if self.model is None:
            raise RuntimeError("no model loaded; call load_model() first")
        config = self.model.config
        hidden = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))
        architectures = getattr(config, "architectures", None) or []
        return ModelDescriptor(
            model_id=self.model_id,
            hidden_size=hidden,
            num_layers=len(self._decoder_layers()),
            architecture=architectures[0] if architectures else type(self.model).__name__,
            revision=self.revision,
        )

    def _decoder_layers(self) -> Any:
        """Locate the decoder-block ModuleList without hardcoding one layout."""
        import torch.nn as nn

        for path in _LAYER_PATHS:
            node: Any = self.model
            for part in path.split("."):
                node = getattr(node, part, None)
                if node is None:
                    break
            if isinstance(node, nn.ModuleList) and len(node) > 0:
                return node

        best = None
        for module in self.model.modules():
            if isinstance(module, nn.ModuleList) and len(module) > 1:
                if len({type(m) for m in module}) == 1:
                    if best is None or len(module) > len(best):
                        best = module
        if best is None:
            raise RuntimeError(
                f"could not locate decoder blocks on {type(self.model).__name__}"
            )
        return best

    # -- intervention ----------------------------------------------------------

    def _tensor_for(self, patch_name: str, key: str) -> Any:
        """Cached on-device float32 tensor for a patch vector."""
        cache_key = (patch_name, key)
        cached = self._vector_cache.get(cache_key)
        if cached is None:
            import torch

            values = self.vector_values(patch_name, key)
            cached = torch.tensor(values, dtype=torch.float32, device=self.device)
            self._vector_cache[cache_key] = cached
        return cached

    def _on_patches_changed(self) -> None:
        self._vector_cache.clear()

    def _delta_for(
        self, layer: int, token_index: int, is_prompt_pass: bool | None = None
    ) -> Any | None:
        """Combined delta for one layer, or None when nothing applies."""
        edits = self.resolve_edits(token_index, layer=layer, is_prompt_pass=is_prompt_pass)
        if not edits:
            return None
        import torch

        delta: Any = None
        for edit in edits:
            vector = self._tensor_for(edit.patch_name, edit.vector_key)
            contribution = vector * edit.coefficient
            delta = contribution if delta is None else delta + contribution
        return delta

    def _hooked_layers(self) -> list[int]:
        return sorted(
            {i.layer for p in self.patches.values() for i in p.manifest.interventions}
        )

    def _install_hooks(self) -> None:
        """Attach one forward hook per layer any installed patch touches."""
        self._remove_hooks()
        layers = self._decoder_layers()
        hooked = self._hooked_layers()
        # The lowest hooked layer is reached first in every forward pass, so it
        # is the reliable place to advance the pass counter exactly once.
        self._first_layer = hooked[0] if hooked else -1
        self._pass_counter = 0
        self._current_pass = 0
        for layer_index in hooked:
            handle = layers[layer_index].register_forward_hook(self._make_hook(layer_index))
            self._handles.append(handle)

    def _make_hook(self, layer_index: int) -> Callable[..., Any]:
        def hook(module: Any, args: Any, output: Any) -> Any:
            if layer_index == self._first_layer:
                self._current_pass = self._pass_counter
                self._pass_counter += 1

            # Pass 0 processes the whole prompt; pass n>0 emits generated token
            # n-1. So the generated-token index is pass - 1, and the prompt pass
            # is not counted -- which is what a user means by "token 20".
            is_prompt_pass = self._current_pass == 0
            token_index = 0 if is_prompt_pass else self._current_pass - 1

            if is_prompt_pass and not self._apply_to_prompt:
                return output

            delta = self._delta_for(layer_index, token_index, is_prompt_pass)
            if delta is None:
                # Untouched: identical to running with no hook at all.
                return output

            hidden, rebuild = _split_output(output)
            if layer_index == self._first_layer:
                self._trace.append((token_index, float(delta.norm().item())))
            return rebuild(hidden + delta.to(dtype=hidden.dtype, device=hidden.device))

        return hook

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # -- generation ------------------------------------------------------------

    def _prepare(self, prompt: str, use_chat_template: bool, system: str | None) -> str:
        if not use_chat_template or getattr(self.tokenizer, "chat_template", None) is None:
            return prompt
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        use_chat_template: bool = True,
        system: str | None = None,
        apply_to_prompt: bool = True,
        **kwargs: Any,
    ) -> str:
        import torch

        if self.model is None:
            raise RuntimeError("no model loaded; call load_model() first")

        cfg = config or GenerationConfig()
        text = self._prepare(prompt, use_chat_template, system)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if cfg.do_sample:
            gen_kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)
            if cfg.top_k > 0:
                gen_kwargs["top_k"] = cfg.top_k
            torch.manual_seed(cfg.seed)
        if cfg.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = cfg.repetition_penalty
        gen_kwargs.update(kwargs)

        self._trace = []
        self._apply_to_prompt = apply_to_prompt
        self._install_hooks()
        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs, **gen_kwargs)
        finally:
            self._remove_hooks()

        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def stream(
        self, prompt: str, config: GenerationConfig | None = None, **kwargs: Any
    ) -> Iterator[str]:
        """Token-by-token streaming via ``TextIteratorStreamer``."""
        import threading

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        thread = threading.Thread(
            target=self.generate, args=(prompt, config), kwargs={**kwargs, "streamer": streamer}
        )
        thread.start()
        try:
            for chunk in streamer:
                yield chunk
        finally:
            thread.join()

    # -- diagnostics -----------------------------------------------------------

    @property
    def last_trace(self) -> list[tuple[int, float]]:
        """``(generated_token_index, delta_norm)`` for the last generation."""
        return list(self._trace)

    def assert_zero_is_baseline(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> dict[str, Any]:
        """Empirically verify that zeroed patches reproduce baseline exactly.

        If this ever fails, every baseline measured with this backend is
        contaminated, so it is worth checking rather than assuming.
        """
        cfg = config or GenerationConfig(max_new_tokens=48)
        saved = {name: p.strength for name, p in self.patches.items()}

        installed = dict(self._patches)
        self._patches = {}
        baseline = self.generate(prompt, cfg)

        self._patches = installed
        for name in self._patches:
            self._patches[name].strength = 0.0
        self._on_patches_changed()
        zeroed = self.generate(prompt, cfg)
        trace = self.last_trace

        for name, strength in saved.items():
            self._patches[name].strength = strength
        self._on_patches_changed()

        return {
            "identical": baseline == zeroed,
            "baseline": baseline,
            "zero_strength": zeroed,
            "applied_passes_at_zero": len(trace),
            "num_patches": len(saved),
        }

    def unload(self) -> None:
        self._remove_hooks()
        self._vector_cache.clear()
        self.model = None
        self.tokenizer = None


def _split_output(output: Any) -> tuple[Any, Callable[[Any], Any]]:
    """Extract hidden states and a rebuilder for a decoder block's output."""
    import torch

    if isinstance(output, torch.Tensor):
        return output, lambda t: t
    if isinstance(output, tuple):
        if not output or not isinstance(output[0], torch.Tensor):
            raise TypeError(f"unexpected decoder output tuple: {type(output)}")
        rest = output[1:]
        return output[0], lambda t: (t, *rest)
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):

        def rebuild(t: Any, _out: Any = output) -> Any:
            _out.last_hidden_state = t
            return _out

        return hidden, rebuild
    raise TypeError(f"cannot locate hidden states in output of type {type(output)}")


def _resolve_device(device: str) -> Any:
    import torch

    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(dtype: str, device: Any) -> Any:
    import torch

    if dtype != "auto":
        return getattr(torch, dtype)
    kind = device.type
    if kind == "cuda":
        return torch.bfloat16
    if kind == "mps":
        return torch.float16
    return torch.float32


def _cached_revision(model: Any) -> str | None:
    """Best-effort recovery of the commit SHA transformers actually loaded."""
    name_or_path = str(getattr(model.config, "_name_or_path", "") or "")
    parts = name_or_path.replace("\\", "/").split("/")
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    commit = getattr(model.config, "_commit_hash", None)
    return str(commit) if commit else None


BACKEND = TransformersBackend
