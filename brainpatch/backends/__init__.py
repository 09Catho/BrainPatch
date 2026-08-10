"""Inference-engine adapters.

Each module here implements :class:`~brainpatch.runtime.base.BrainPatchBackend`
for one engine and imports that engine's dependencies **only when
instantiated**. ``is_available()`` and ``capabilities()`` are classmethods that
work with nothing installed, which is what lets ``brainpatch doctor`` report on
a bare machine.

Nothing is imported eagerly here: importing this package must not pull in torch.

=================  ==============================================
``transformers``   reference backend; PyTorch, CUDA/CPU/MPS
``llamacpp``       GGUF via upstream llama.cpp control vectors
``vllm``           high-throughput serving with request isolation
``mlx``            Apple Silicon via MLX-LM
=================  ==============================================
"""

__all__: list[str] = []
