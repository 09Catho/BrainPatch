"""Top-K sparse autoencoder configuration.

This module holds only the *description* of an SAE. The torch implementation
lives in :mod:`brainpatch.ml.sae` and is never imported locally.

Design notes captured here because they matter for correctness of downstream
interventions:

Input normalization
    Residual-stream activations of different models/layers have wildly
    different scales. We rescale inputs so that ``E[||x||_2] == sqrt(d_in)``
    and store the resulting ``input_scale`` in the checkpoint. Any intervention
    that injects a decoder direction back into the *raw* residual stream must
    multiply by ``input_scale`` to undo the normalization -- otherwise a
    "strength of 1.0" means something different for every SAE.

Decoder normalization
    Decoder columns are constrained to unit L2 norm. Without this, the network
    can trivially shrink the decoder and inflate feature activations (or vice
    versa), which makes activation magnitudes -- and therefore any strength
    parameter defined in terms of them -- meaningless. Unit-norm columns give
    ``strength`` a stable interpretation: "add ``strength * input_scale`` units
    of length along this direction".

Dead features
    Top-K SAEs reliably produce features that stop firing. We track a rolling
    fire count and optionally apply the AuxK auxiliary loss (reconstruct the
    residual error using only currently-dead features), which is the standard
    mitigation from the OpenAI Top-K SAE work.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SAE_FORMAT_VERSION = "0.1"


@dataclass
class SAEConfig:
    """Architecture + training configuration of a Top-K SAE.

    Attributes
    ----------
    d_in:
        Residual-stream width of the base model (e.g. 1536 for Qwen2.5-1.5B).
    d_sae:
        Dictionary size (number of learned features).
    k:
        Upper bound on features kept active per token by the Top-K operator.
        ``torch.topk`` always selects ``k`` indices, but the preceding ReLU can
        make some of the selected values zero, so measured L0 is ``<= k`` rather
        than identically ``k``.
    normalize_decoder:
        Constrain decoder columns to unit L2 norm after every optimizer step.
    tied_init:
        Initialise ``W_dec = W_enc.T``; a standard and stable starting point.
    auxk_alpha:
        Weight of the AuxK dead-feature revival loss. ``0.0`` disables it.
    auxk_k:
        How many dead features AuxK reconstructs the residual with.
    dead_feature_window:
        A feature is "dead" if it has not fired in this many training tokens.
    """

    # architecture
    d_in: int
    d_sae: int
    k: int = 32
    normalize_decoder: bool = True
    tied_init: bool = True
    auxk_alpha: float = 1.0 / 32.0
    auxk_k: int = 256
    dead_feature_window: int = 200_000

    # provenance of the activations this SAE is defined over
    model: str = ""
    model_revision: str = ""
    layer: int = -1
    hook: str = ""

    # optimisation
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    batch_size: int = 512
    epochs: int = 1
    max_steps: int | None = None
    grad_clip: float = 1.0
    lr_warmup_steps: int = 50
    seed: int = 0

    # data handling
    shuffle_buffer: int = 8192
    val_fraction: float = 0.05

    #: Filled in during training: multiply raw activations by this to normalize.
    #: ``None`` until measured on a sample of the corpus.
    input_scale: float | None = None

    format_version: str = SAE_FORMAT_VERSION
    notes: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise :class:`ValueError` on an unusable configuration."""
        if self.d_in <= 0:
            raise ValueError(f"d_in must be positive, got {self.d_in}")
        if self.d_sae <= 0:
            raise ValueError(f"d_sae must be positive, got {self.d_sae}")
        if not 0 < self.k <= self.d_sae:
            raise ValueError(f"k must satisfy 0 < k <= d_sae, got k={self.k}, d_sae={self.d_sae}")
        if self.auxk_alpha < 0:
            raise ValueError(f"auxk_alpha must be non-negative, got {self.auxk_alpha}")
        if self.auxk_k <= 0 or self.auxk_k > self.d_sae:
            raise ValueError(f"auxk_k must satisfy 0 < auxk_k <= d_sae, got {self.auxk_k}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

    @property
    def expansion_factor(self) -> float:
        """Dictionary size relative to the residual width."""
        return self.d_sae / self.d_in

    @property
    def num_parameters(self) -> int:
        """Parameter count: encoder + decoder weights plus both bias vectors."""
        return 2 * self.d_in * self.d_sae + self.d_sae + self.d_in

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SAEConfig":
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        return cls(**{k: v for k, v in data.items() if k in known})  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, text: str) -> "SAEConfig":
        return cls.from_dict(json.loads(text))
