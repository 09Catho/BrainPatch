"""Worker-side extension that installs BrainPatch hooks inside vLLM.

vLLM's V1 engine runs the model in a separate process, and its RPC channel
serializes with msgpack -- it will **not** ship a callable, and says so:

    TypeError: Object of type <class 'function'> is not serializable
    Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow fallback to pickle

Enabling pickle would work and is the wrong answer: it turns the engine's
control channel into an arbitrary-code path for the sake of our convenience.
The supported mechanism is ``worker_extension_cls`` -- vLLM mixes this class
into the worker, and ``collective_rpc("method_name", args=...)`` then calls it
by **name** with plain msgpack-serializable arguments.

So the payload here is a ``{layer_index: [floats]}`` map of already-scaled
deltas. That suffices because this backend supports neither token schedules nor
per-request strength, so the delta is constant across a forward pass -- which is
also exactly what makes concurrent batching safe.

Method names are prefixed ``bp_`` to avoid colliding with vLLM's own worker API.
"""

from __future__ import annotations

from typing import Any

_LAYER_PATHS = ("model.layers", "layers", "transformer.h", "model.decoder.layers")

#: Set on the worker-side model so repeated calls replace rather than stack hooks.
_HOOK_ATTR = "_brainpatch_handles"


class BrainPatchWorkerExtension:
    """Mixed into the vLLM worker; ``self`` is the worker instance."""

    def _bp_model(self) -> Any:
        for path in ("model_runner.model", "worker.model_runner.model"):
            node: Any = self
            for part in path.split("."):
                node = getattr(node, part, None)
                if node is None:
                    break
            if node is not None:
                return node
        raise RuntimeError(
            f"could not locate the model on vLLM worker {type(self).__name__}"
        )

    def _bp_layers(self, model: Any) -> Any:
        import torch.nn as nn

        for path in _LAYER_PATHS:
            node: Any = model
            for part in path.split("."):
                node = getattr(node, part, None)
                if node is None:
                    break
            if isinstance(node, nn.ModuleList) and len(node) > 0:
                return node
        raise RuntimeError(f"could not locate decoder blocks on {type(model).__name__}")

    def bp_probe(self) -> dict[str, Any]:
        """Report worker-side model geometry and live hook count."""
        model = self._bp_model()
        layers = self._bp_layers(model)
        config = getattr(model, "config", None)
        return {
            "model_class": type(model).__name__,
            "num_layers": len(layers),
            "hidden_size": int(getattr(config, "hidden_size", 0)) if config else 0,
            "architectures": list(getattr(config, "architectures", []) or []) if config else [],
            "active_hooks": len(getattr(model, _HOOK_ATTR, [])),
        }

    def bp_apply_deltas(self, deltas: dict[int, list[float]]) -> dict[str, Any]:
        """Install (or replace) residual-stream hooks. Empty dict removes all.

        Returns a report so the caller can *verify* the hooks landed inside the
        worker rather than infer it from output changes -- a silently failed RPC
        and a patch with no effect look identical from outside.
        """
        import torch

        model = self._bp_model()
        layers = self._bp_layers(model)

        for handle in getattr(model, _HOOK_ATTR, []):
            handle.remove()
        setattr(model, _HOOK_ATTR, [])

        if not deltas:
            return {**self.bp_probe(), "num_hooks": 0}

        parameter = next(model.parameters())
        device, dtype = parameter.device, parameter.dtype

        handles = []
        # msgpack may deliver dict keys as strings; normalise before indexing.
        for raw_layer, values in sorted(((int(k), v) for k, v in deltas.items())):
            if raw_layer >= len(layers):
                raise RuntimeError(f"layer {raw_layer} out of range ({len(layers)} blocks)")
            vector = torch.tensor(values, dtype=dtype, device=device)
            handles.append(layers[raw_layer].register_forward_hook(_make_hook(vector)))

        setattr(model, _HOOK_ATTR, handles)
        return {
            **self.bp_probe(),
            "num_hooks": len(handles),
            "hooked_layers": sorted(int(k) for k in deltas),
            "device": str(device),
            "dtype": str(dtype),
        }


def _make_hook(vector: Any) -> Any:
    """Forward hook adding ``vector`` to a decoder block's hidden states."""
    import torch

    def hook(module: Any, args: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            if not isinstance(hidden, torch.Tensor):
                return output
            return (hidden + vector, *output[1:])
        if isinstance(output, torch.Tensor):
            return output + vector
        return output

    return hook
