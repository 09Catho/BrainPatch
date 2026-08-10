"""Residual-stream capture and injection hooks.

Both directions of the pipeline attach to the *output* of a decoder block:

* :class:`ResidualCapture` records what the block wrote, which is what the SAE
  is trained on.
* :class:`ResidualInjector` adds a vector to the same tensor, which is how a
  BrainPatch takes effect.

Using the identical site for both is not a convenience -- it is a correctness
requirement. An SAE decoder direction is only meaningful in the coordinate
system it was fitted in, so injecting at any other site would be adding a
vector that means nothing there.

Decoder blocks in transformers return either a bare tensor or a tuple whose
first element is the hidden state, depending on version and config. Both shapes
are handled, and the tuple is rebuilt rather than mutated so that nothing
downstream sees a half-modified structure.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def _split_output(output: Any) -> tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
    """Extract the hidden-state tensor and a rebuilder for the block output.

    Returns
    -------
    (hidden_states, rebuild)
        ``rebuild(new_tensor)`` reconstructs the original container type with
        the tensor replaced.
    """
    if isinstance(output, torch.Tensor):
        return output, lambda t: t
    if isinstance(output, tuple):
        if not output or not isinstance(output[0], torch.Tensor):
            raise TypeError(f"unexpected decoder block output tuple: {type(output)}")
        rest = output[1:]
        return output[0], lambda t: (t, *rest)
    # Some versions return a ModelOutput-like object with .last_hidden_state.
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):

        def rebuild(t: torch.Tensor, _out: Any = output) -> Any:
            _out.last_hidden_state = t
            return _out

        return hidden, rebuild
    raise TypeError(f"cannot locate hidden states in decoder block output of type {type(output)}")


class ResidualCapture:
    """Capture the residual stream at one decoder block.

    Usage::

        capture = ResidualCapture()
        handle = capture.attach(bundle.layer_module(18))
        with torch.inference_mode():
            model(**batch)
        acts = capture.activations   # [batch, seq, hidden]
        handle.remove()

    The captured tensor is *detached* and optionally moved off-GPU immediately,
    so a long extraction run does not accumulate VRAM.
    """

    def __init__(self, *, to_cpu: bool = True, dtype: torch.dtype | None = None) -> None:
        self.to_cpu = to_cpu
        self.dtype = dtype
        self.activations: torch.Tensor | None = None
        self._handle: Any = None

    def __call__(self, module: Any, args: Any, output: Any) -> None:
        hidden, _ = _split_output(output)
        tensor = hidden.detach()
        if self.dtype is not None:
            tensor = tensor.to(self.dtype)
        if self.to_cpu:
            tensor = tensor.to("cpu")
        self.activations = tensor

    def attach(self, module: Any) -> Any:
        """Register on ``module`` and return the removable handle."""
        self._handle = module.register_forward_hook(self)
        return self._handle

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "ResidualCapture":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove()


class ResidualInjector:
    """Add a per-token vector to the residual stream at one decoder block.

    The injected vector is supplied by a callback rather than fixed at
    construction, because in dynamic steering it changes with the generated
    token index. The callback receives the current hidden states and returns
    either a broadcastable tensor to add, or ``None`` for "do nothing".

    Returning ``None`` is the mechanism that makes ``strength=0`` *identical*
    to baseline rather than approximately equal: no arithmetic is performed on
    the tensor at all, so there is not even a float round-trip to differ on.
    """

    def __init__(
        self,
        delta_fn: Callable[[torch.Tensor], torch.Tensor | None],
        *,
        name: str = "injector",
    ) -> None:
        self.delta_fn = delta_fn
        self.name = name
        self._handle: Any = None
        #: Incremented every time a non-None delta is actually applied.
        self.apply_count = 0
        #: Incremented on every forward pass through the hooked module.
        self.call_count = 0

    def __call__(self, module: Any, args: Any, output: Any) -> Any:
        self.call_count += 1
        hidden, rebuild = _split_output(output)
        delta = self.delta_fn(hidden)
        if delta is None:
            # Untouched: bit-identical to running without the hook.
            return output
        self.apply_count += 1
        modified = hidden + delta.to(dtype=hidden.dtype, device=hidden.device)
        return rebuild(modified)

    def attach(self, module: Any) -> Any:
        self._handle = module.register_forward_hook(self)
        return self._handle

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "ResidualInjector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove()


class HookSet:
    """Context manager owning several hook handles at once.

    Guarantees removal even if the forward pass raises, which matters because a
    leaked injection hook would silently contaminate every subsequent
    generation in the same process -- including the "baseline" ones.
    """

    def __init__(self) -> None:
        self._hooks: list[Any] = []

    def add(self, hook: ResidualCapture | ResidualInjector, module: Any) -> Any:
        handle = hook.attach(module)
        self._hooks.append(hook)
        return handle

    def remove_all(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self) -> "HookSet":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove_all()
