"""Top-K sparse autoencoder.

Forward pass::

    z_pre = W_enc @ (x - b_dec) + b_enc
    z     = TopK(ReLU(z_pre))          # AT MOST k non-zeros -- see below
    x_hat = W_dec @ z + b_dec

Design choices, and why each one is load-bearing for *intervention* rather than
just for reconstruction quality:

**Top-K instead of an L1 penalty.**
    L1 sparsity requires tuning a coefficient whose right value depends on the
    activation scale, and it shrinks every activation toward zero, biasing
    magnitudes. Top-K caps L0 at ``k``, so sparsity is a hyperparameter rather
    than an outcome, and activations are unbiased.

**L0 is an upper bound, not an identity.**
    ``torch.topk`` always returns exactly ``k`` indices, but the ReLU in front
    of it means some of the selected *values* can be zero -- whenever fewer
    than ``k`` encoder pre-activations are positive for a token. Those entries
    scatter zeros into the sparse tensor, so the reconstruction is correct and
    the measured L0 (counted as ``feature_acts > 0``) correctly falls below
    ``k``. What is *not* correct is treating a selected index as evidence the
    feature fired: see :meth:`TopKSAE.update_liveness`, which filters on the
    value rather than the index.

**Pre-encoder bias subtraction (``x - b_dec``).**
    Centres the input on the decoder's own bias so the dictionary models
    deviations from the mean activation rather than spending capacity on it.

**Unit-norm decoder columns.**
    Without this constraint the network can halve every decoder column and
    double every activation with no change in the loss. That is fatal here, not
    merely untidy: a BrainPatch says "add strength 1.5 along feature 1207's
    direction", and if the direction's length is arbitrary then so is the
    strength. Unit norms make ``strength`` mean a fixed distance in activation
    space.

**AuxK dead-feature revival.**
    Top-K SAEs reliably kill features. AuxK (from the OpenAI Top-K SAE work)
    asks the currently-dead features to reconstruct the residual error, which
    gives them gradient signal without perturbing the main objective. Disabled
    by setting ``auxk_alpha = 0``.

**Gradient projection on the decoder.**
    Renormalising decoder columns after an optimizer step leaves a component of
    the gradient that only changed the norm -- work the projection undoes.
    Removing the parallel component first makes the constrained optimisation
    behave.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from brainpatch.schemas.sae import SAEConfig


@dataclass
class SAEOutput:
    """Everything one forward pass produces, for loss and for metrics."""

    reconstruction: torch.Tensor
    feature_acts: torch.Tensor
    """Sparse activations, ``[batch, d_sae]``, **at most** ``k`` non-zero per row.

    Fewer than ``k`` when a token has fewer than ``k`` positive encoder
    pre-activations. This is the tensor to count L0 from.
    """
    topk_indices: torch.Tensor
    """``[batch, k]`` indices returned by ``torch.topk``.

    Always exactly ``k`` wide. An index appearing here does **not** mean the
    feature fired -- pair it with :attr:`topk_values` and require a strictly
    positive value.
    """
    topk_values: torch.Tensor
    """``[batch, k]`` selected values. Non-negative, and **may contain zeros**."""
    pre_acts: torch.Tensor
    """Dense post-ReLU, pre-TopK activations, needed by AuxK."""

    def active_mask(self) -> torch.Tensor:
        """``[batch, k]`` boolean: which Top-K selections actually fired."""
        return self.topk_values > 0

    def l0(self) -> torch.Tensor:
        """Per-row count of strictly positive feature activations."""
        return (self.feature_acts > 0).sum(dim=-1)


class TopKSAE(nn.Module):
    """A Top-K sparse autoencoder over residual-stream activations."""

    def __init__(self, config: SAEConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.d_in = config.d_in
        self.d_sae = config.d_sae
        self.k = config.k

        self.W_enc = nn.Parameter(torch.empty(config.d_sae, config.d_in))
        self.b_enc = nn.Parameter(torch.zeros(config.d_sae))
        self.W_dec = nn.Parameter(torch.empty(config.d_in, config.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(config.d_in))

        # Feature liveness tracking. Registered as a buffer so it round-trips
        # through checkpoints and resume actually resumes the dead-feature state.
        self.register_buffer("tokens_since_fired", torch.zeros(config.d_sae, dtype=torch.long))
        self.register_buffer("fire_count", torch.zeros(config.d_sae, dtype=torch.long))
        self.register_buffer("tokens_seen", torch.zeros((), dtype=torch.long))

        self._init_weights()

    # -- initialisation --------------------------------------------------------

    def _init_weights(self) -> None:
        generator = torch.Generator().manual_seed(self.config.seed)
        # Kaiming-uniform-ish scale for the decoder, then tie the encoder to it.
        bound = 1.0 / (self.d_in**0.5)
        w_dec = torch.empty(self.d_in, self.d_sae).uniform_(-bound, bound, generator=generator)
        w_dec = w_dec / w_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        with torch.no_grad():
            self.W_dec.copy_(w_dec)
            if self.config.tied_init:
                self.W_enc.copy_(w_dec.T.clone())
            else:
                self.W_enc.copy_(
                    torch.empty(self.d_sae, self.d_in).uniform_(-bound, bound, generator=generator)
                )

    @torch.no_grad()
    def set_decoder_bias_to_mean(self, sample: torch.Tensor) -> None:
        """Initialise ``b_dec`` to the corpus mean.

        Starting the decoder bias at the data mean means the dictionary begins
        by modelling deviations rather than spending its first thousand steps
        learning where the centre of the distribution is.
        """
        self.b_dec.copy_(sample.to(self.b_dec.dtype).mean(dim=0))

    # -- forward ---------------------------------------------------------------

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Dense pre-activations (before ReLU and Top-K)."""
        return F.linear(x - self.b_dec, self.W_enc, self.b_enc)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(sparse_acts, topk_indices, topk_values)``."""
        pre = F.relu(self.encode_pre(x))
        values, indices = torch.topk(pre, self.k, dim=-1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, indices, values)
        return sparse, indices, values

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Reconstruct from sparse feature activations."""
        return F.linear(feature_acts, self.W_dec, self.b_dec)

    def forward(self, x: torch.Tensor) -> SAEOutput:
        pre = F.relu(self.encode_pre(x))
        values, indices = torch.topk(pre, self.k, dim=-1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, indices, values)
        return SAEOutput(
            reconstruction=self.decode(sparse),
            feature_acts=sparse,
            topk_indices=indices,
            topk_values=values,
            pre_acts=pre,
        )

    # -- feature directions ----------------------------------------------------

    def feature_direction(self, feature_id: int, *, normalize: bool = True) -> torch.Tensor:
        """The decoder column for one feature -- the vector an intervention adds.

        With ``normalize=True`` (default) the returned vector has unit L2 norm
        even if the training-time constraint has drifted, so a caller's
        ``strength`` always means the same distance.
        """
        if not 0 <= feature_id < self.d_sae:
            raise IndexError(f"feature {feature_id} out of range for dictionary of size {self.d_sae}")
        direction = self.W_dec[:, feature_id]
        if normalize:
            direction = direction / direction.norm().clamp_min(1e-8)
        return direction

    def decoder_norms(self) -> torch.Tensor:
        """L2 norm of every decoder column."""
        return self.W_dec.norm(dim=0)

    # -- constraints -----------------------------------------------------------

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Rescale decoder columns to unit norm."""
        if not self.config.normalize_decoder:
            return
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def project_decoder_grad(self) -> None:
        """Remove the component of ``W_dec.grad`` parallel to each column.

        Called between ``backward()`` and ``step()``. Without it, the optimizer
        spends part of every update changing column norms that
        :meth:`normalize_decoder` then immediately undoes.
        """
        if not self.config.normalize_decoder or self.W_dec.grad is None:
            return
        w = self.W_dec
        g = self.W_dec.grad
        parallel = (g * w).sum(dim=0, keepdim=True) * w / w.norm(dim=0, keepdim=True).clamp_min(1e-8) ** 2
        g.sub_(parallel)

    # -- liveness --------------------------------------------------------------

    @torch.no_grad()
    def update_liveness(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        batch_tokens: int,
    ) -> None:
        """Update fire counts and the dead-feature clock.

        Only selections with a **strictly positive** value count as firings.

        ``torch.topk`` always returns ``k`` indices, but the ReLU in front of it
        means those can include zero-valued entries whenever a token has fewer
        than ``k`` positive pre-activations. Counting the raw indices would
        inflate :attr:`fire_count`, and -- worse -- would reset
        :attr:`tokens_since_fired` for a feature that did not fire, hiding it
        from :meth:`dead_mask` and therefore from AuxK revival. A permanently
        silent feature could then be reported alive forever.

        Parameters
        ----------
        indices, values:
            ``SAEOutput.topk_indices`` and ``SAEOutput.topk_values``, same shape.
        batch_tokens:
            Rows in this batch; advances the dead-feature clock.
        """
        if indices.shape != values.shape:
            raise ValueError(
                f"indices and values must have the same shape, got "
                f"{tuple(indices.shape)} and {tuple(values.shape)}"
            )

        active = indices.reshape(-1)[values.reshape(-1) > 0]

        counts = torch.bincount(active, minlength=self.d_sae)
        self.fire_count += counts.to(self.fire_count.dtype)

        self.tokens_since_fired += batch_tokens
        if active.numel() > 0:
            fired = torch.zeros(self.d_sae, dtype=torch.bool, device=indices.device)
            fired[active] = True
            self.tokens_since_fired[fired] = 0

        self.tokens_seen += batch_tokens

    def dead_mask(self) -> torch.Tensor:
        """Boolean mask of features silent for longer than the dead window.

        "Silent" means no strictly positive activation, not merely "not selected
        by Top-K" -- see :meth:`update_liveness`.
        """
        return self.tokens_since_fired > self.config.dead_feature_window

    def num_dead(self) -> int:
        return int(self.dead_mask().sum().item())

    # -- losses ----------------------------------------------------------------

    def auxk_loss(self, x: torch.Tensor, out: SAEOutput) -> torch.Tensor:
        """Reconstruct the residual error using only dead features.

        Returns a zero scalar when AuxK is disabled or nothing is dead yet, so
        the caller can add it unconditionally.
        """
        if self.config.auxk_alpha <= 0:
            return x.new_zeros(())
        dead = self.dead_mask()
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return x.new_zeros(())

        k_aux = min(self.config.auxk_k, n_dead)
        residual = (x - out.reconstruction).detach()

        masked_pre = out.pre_acts.masked_fill(~dead.unsqueeze(0), 0.0)
        values, indices = torch.topk(masked_pre, k_aux, dim=-1)
        sparse = torch.zeros_like(masked_pre)
        # Zero-valued selections are harmless here: scattering 0.0 into a zeros
        # tensor is a no-op and contributes nothing to the linear map below. The
        # zero-selection hazard is confined to liveness accounting, which counts
        # occurrences rather than summing values.
        sparse.scatter_(-1, indices, values)

        aux_reconstruction = F.linear(sparse, self.W_dec)  # no bias: modelling the residual
        return F.mse_loss(aux_reconstruction, residual)

    # -- serialization ---------------------------------------------------------

    def state_dict_with_config(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict[str, Any], *, device: str = "cpu") -> "TopKSAE":
        """Rebuild an SAE from a checkpoint dict, config included."""
        config = SAEConfig.from_dict(checkpoint["config"])
        sae = cls(config)
        sae.load_state_dict(checkpoint["state_dict"])
        sae.to(device)
        sae.eval()
        return sae


def reconstruction_metrics(x: torch.Tensor, out: SAEOutput) -> dict[str, float]:
    """Quality metrics for one batch.

    ``explained_variance`` is computed against the per-dimension variance of the
    batch, i.e. ``1 - Var(x - x_hat) / Var(x)``. It can go negative for a
    freshly initialised SAE, which is meaningful and not clipped away.

    ``l0`` counts strictly positive entries of ``feature_acts``, so it is the
    *true* mean L0 and is bounded above by ``k`` rather than equal to it. A
    reported ``l0`` below ``k`` means some tokens had fewer than ``k`` positive
    encoder pre-activations, which is legitimate and worth noticing.

    ``mean_active_value`` averages over strictly positive selections only;
    including zero-valued Top-K entries would bias it toward zero exactly when
    the SAE is sparsest.
    """
    with torch.no_grad():
        x = x.float()
        recon = out.reconstruction.float()
        residual = x - recon

        mse = residual.pow(2).mean().item()
        total_var = x.var(dim=0, unbiased=False).sum().item()
        resid_var = residual.var(dim=0, unbiased=False).sum().item()
        explained = 1.0 - (resid_var / total_var) if total_var > 0 else float("nan")

        x_norm = x.norm(dim=-1)
        cos = F.cosine_similarity(x, recon, dim=-1).mean().item()

        # True L0: strictly positive activations, bounded above by k.
        l0_per_row = out.l0().float()
        l0 = l0_per_row.mean().item()

        # Fraction of Top-K slots that selected a zero. Non-zero here means the
        # dictionary is saturating below k for some tokens; it is also the
        # condition under which naive index-based liveness accounting would be
        # wrong, so it is worth logging rather than inferring.
        active = out.active_mask()
        zero_selection_rate = 1.0 - active.float().mean().item()
        positive_values = out.topk_values[active]
        mean_active_value = (
            positive_values.mean().item() if positive_values.numel() > 0 else 0.0
        )

        return {
            "mse": mse,
            "normalized_mse": mse / x.pow(2).mean().item() if x.pow(2).mean().item() > 0 else float("nan"),
            "explained_variance": explained,
            "cosine_similarity": cos,
            "l0": l0,
            "l0_min": l0_per_row.min().item(),
            "l0_max": l0_per_row.max().item(),
            "zero_selection_rate": zero_selection_rate,
            "mean_input_norm": x_norm.mean().item(),
            "mean_recon_norm": recon.norm(dim=-1).mean().item(),
            "mean_active_value": mean_active_value,
        }
