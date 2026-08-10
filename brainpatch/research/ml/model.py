"""Loading the frozen base model and discovering its architecture.

Two principles here:

1. **Nothing is hardcoded that can be discovered.** Layer count, hidden size and
   the list of decoder blocks are read off the loaded model, so the same code
   works if the base model is swapped. A configured target layer is validated
   against the real depth rather than assumed to exist.

2. **The model is frozen.** ``requires_grad_(False)`` plus ``eval()`` on every
   load path. BrainPatch never fine-tunes; every behavioural change comes from
   activations, not weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

#: Hook site naming. ``residual_post`` is the output of decoder block *i*,
#: i.e. the residual stream after that block has written to it. This is the
#: standard site for SAE work because it is where a block's contribution is
#: visible and where an injection propagates to every later block.
HOOK_RESIDUAL_POST = "residual_post"
SUPPORTED_HOOKS = (HOOK_RESIDUAL_POST,)


@dataclass
class ModelBundle:
    """A loaded, frozen model with its tokenizer and discovered architecture."""

    model: Any
    tokenizer: Any
    model_id: str
    revision: str
    hidden_size: int
    num_layers: int
    dtype: torch.dtype
    device: torch.device

    def decoder_layers(self) -> Any:
        """The ``nn.ModuleList`` of decoder blocks."""
        return _find_decoder_layers(self.model)

    def layer_module(self, layer: int) -> Any:
        """The decoder block at ``layer``, validated against the real depth."""
        validate_layer(layer, self.num_layers)
        return self.decoder_layers()[layer]

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "revision": self.revision,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dtype": str(self.dtype).replace("torch.", ""),
            "device": str(self.device),
        }


def validate_layer(layer: int, num_layers: int) -> int:
    """Raise if ``layer`` is not a real decoder block index.

    Negative indices count from the end, matching Python convention, and are
    resolved to a positive index.
    """
    resolved = layer if layer >= 0 else num_layers + layer
    if not 0 <= resolved < num_layers:
        raise ValueError(
            f"layer {layer} does not exist: this model has {num_layers} decoder "
            f"blocks (valid indices 0..{num_layers - 1})"
        )
    return resolved


def validate_hook(hook: str) -> str:
    if hook not in SUPPORTED_HOOKS:
        raise ValueError(f"unsupported hook site {hook!r}; supported: {SUPPORTED_HOOKS}")
    return hook


def _find_decoder_layers(model: Any) -> Any:
    """Locate the decoder-block ``ModuleList`` without hardcoding a path.

    Tries the common Hugging Face layouts in order, then falls back to scanning
    for the largest ``ModuleList`` of identically-typed modules.
    """
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        node: Any = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None and isinstance(node, torch.nn.ModuleList) and len(node) > 0:
            return node

    best: Any = None
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 1:
            types = {type(m) for m in module}
            if len(types) == 1 and (best is None or len(module) > len(best)):
                best = module
    if best is None:
        raise RuntimeError(
            f"could not locate decoder blocks on {type(model).__name__}; "
            "add its layout to _find_decoder_layers"
        )
    return best


def resolve_revision(model_id: str, revision: str | None = None) -> str:
    """Resolve a branch name to an immutable commit SHA.

    Pinning the SHA matters: feature directions are properties of a specific set
    of weights, so an experiment that only records "main" is not reproducible.
    """
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, revision=revision or "main")
    return info.sha


def load_model(
    model_id: str = DEFAULT_MODEL,
    *,
    revision: str | None = None,
    dtype: str = "bfloat16",
    device: str = "cuda",
    trust_remote_code: bool = False,
) -> ModelBundle:
    """Load a frozen causal LM from the Volume-backed Hugging Face cache.

    The cache location comes from ``HF_HOME`` / ``HF_HUB_CACHE``, which the
    Modal image points at ``/vol/hf-cache``. Nothing is downloaded twice and
    nothing lands on ephemeral container storage.
    """
    torch_dtype = getattr(torch, dtype)
    resolved_device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(resolved_device)

    # Frozen for the entire lifetime of the process. BrainPatch never trains
    # the base model; every effect must come from activations.
    model.eval()
    model.requires_grad_(False)

    config = model.config
    hidden_size = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))
    if hidden_size <= 0:
        raise RuntimeError(f"could not discover hidden size for {model_id}")
    num_layers = len(_find_decoder_layers(model))

    effective_revision = revision or _commit_hash_from_cache(model) or "unknown"

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        revision=effective_revision,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dtype=torch_dtype,
        device=resolved_device,
    )


def _commit_hash_from_cache(model: Any) -> str | None:
    """Best-effort recovery of the commit SHA transformers actually loaded."""
    name_or_path = getattr(model.config, "_name_or_path", "") or ""
    # Cached snapshots live at .../snapshots/<sha>/...
    parts = str(name_or_path).replace("\\", "/").split("/")
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    commit = getattr(model.config, "_commit_hash", None)
    return str(commit) if commit else None


def architecture_summary(model_id: str = DEFAULT_MODEL, revision: str | None = None) -> dict[str, Any]:
    """Read architecture facts from the config alone -- no weights loaded.

    Cheap enough to run on CPU, which is how the layer choice is validated
    before any GPU time is spent.
    """
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    return {
        "model": model_id,
        "model_type": config.model_type,
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(getattr(config, "num_key_value_heads", config.num_attention_heads)),
        "intermediate_size": int(config.intermediate_size),
        "vocab_size": int(config.vocab_size),
        "max_position_embeddings": int(config.max_position_embeddings),
        "torch_dtype": str(getattr(config, "torch_dtype", "unknown")),
    }


def hf_token_present() -> bool:
    """Whether an HF token is available, without ever revealing it."""
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
