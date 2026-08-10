"""Torch-dependent BrainPatch internals.

**Nothing in this subpackage may be imported from a machine without the ML
stack.** Every module here imports ``torch`` (and often ``transformers``) at
module scope, which is fine because these modules only ever execute inside a
Modal container.

The parent package :mod:`brainpatch` never imports this eagerly; see the
``__getattr__`` shim in ``brainpatch/__init__.py``.

Module map
----------
``model``
    Loading Qwen from the Volume-backed HF cache, and architecture discovery.
``hooks``
    Residual-stream capture and injection hooks.
``extraction``
    Streaming activation capture into immutable shards.
``activation_store``
    Streaming reader over those shards for SAE training.
``sae``
    The Top-K sparse autoencoder.
``training``
    SAE training loop with checkpoint/resume.
``feature_analysis``
    Per-feature statistics and top-activating contexts.
``intervention``
    Runtime feature injection and ablation.
``runtime``
    ``BrainPatchedModel``, the user-facing API.
``generation``
    Deterministic generation helpers used by the causal-validation harness.
``evaluation``
    Model-dependent measurements (log-probabilities, capability probes).
"""
