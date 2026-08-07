"""Runtime feature injection into the residual stream.

A :class:`FeatureSteerer` owns one SAE and one hook site, and turns an
:class:`~brainpatch.steering.plan.InterventionPlan` into a concrete vector added
to the residual stream on each forward pass.

Scale
-----
The SAE was trained on activations multiplied by ``input_scale`` so that
``E[||x||] == sqrt(d_in)``. Decoder columns therefore live in normalized space.
To inject a direction back into the *raw* residual stream we divide by that
scale::

    delta_raw = (coefficient * unit_direction) / input_scale

This is the step that makes ``strength`` portable: strength 1.0 adds one unit of
normalized activation-space distance regardless of how large the raw residual
stream happens to be at that layer.

Token indexing
--------------
With a KV cache, generation runs one forward pass over the whole prompt and then
one pass per new token. The steerer counts passes so that "generated token
index" means what a user expects: index 0 is the first token the model emits,
and the prompt is not counted.

Controls
--------
:class:`RandomDirectionSteerer` and unrelated-feature interventions use the same
coefficient path as the real thing. Since both decoder columns and the random
directions are unit-norm, the injected vectors have *identical* L2 norms by
construction -- the control differs only in direction. That is the comparison
that makes an effect attributable to the feature rather than to the magnitude of
the perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from brainpatch.ml.hooks import ResidualInjector
from brainpatch.ml.sae import TopKSAE
from brainpatch.steering.plan import InterventionPlan, PlannedEdit


@dataclass
class SteeringStats:
    """Bookkeeping that lets a caller verify an intervention actually happened."""

    forward_passes: int = 0
    applied_passes: int = 0
    total_delta_norm: float = 0.0
    max_delta_norm: float = 0.0
    #: ``(generated_token_index, delta_norm)`` for every forward pass, including
    #: passes where nothing was applied (norm 0.0). Recording the skipped passes
    #: too is what makes this a usable trace of a dynamic schedule -- otherwise
    #: the list index would silently stop matching the token index.
    per_token_norms: list[tuple[int, float]] = field(default_factory=list)

    @property
    def mean_delta_norm(self) -> float:
        return self.total_delta_norm / self.applied_passes if self.applied_passes else 0.0

    def to_dict(self, *, include_per_token: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "forward_passes": self.forward_passes,
            "applied_passes": self.applied_passes,
            "mean_delta_norm": self.mean_delta_norm,
            "max_delta_norm": self.max_delta_norm,
        }
        if include_per_token:
            data["per_token_norms"] = [[i, round(n, 5)] for i, n in self.per_token_norms]
        return data


class FeatureSteerer:
    """Builds and applies residual-stream deltas for one hook site."""

    def __init__(
        self,
        sae: TopKSAE,
        plan: InterventionPlan,
        *,
        layer: int,
        input_scale: float,
        device: torch.device | str = "cuda",
        apply_to_prompt: bool = True,
    ) -> None:
        self.sae = sae
        self.plan = plan
        self.layer = layer
        self.input_scale = float(input_scale)
        if self.input_scale == 0:
            raise ValueError("input_scale must be non-zero")
        self.device = torch.device(device)
        self.apply_to_prompt = apply_to_prompt

        self._pass_index = 0
        self.stats = SteeringStats()
        #: Cache of unit decoder directions, so a long generation does not
        #: re-slice and re-normalize the same columns thousands of times.
        self._direction_cache: dict[int, torch.Tensor] = {}

    # -- lifecycle -------------------------------------------------------------

    def reset(self) -> None:
        """Reset token counting and stats. Call before every generation."""
        self._pass_index = 0
        self.stats = SteeringStats()

    @property
    def generated_index(self) -> int:
        """Generated-token index for the pass currently being processed.

        ``-1`` during the prompt pass, then 0, 1, 2, ...
        """
        return self._pass_index - 1

    # -- directions ------------------------------------------------------------

    def direction(self, feature_id: int) -> torch.Tensor:
        """Unit-norm decoder column for ``feature_id``, cached and on-device."""
        cached = self._direction_cache.get(feature_id)
        if cached is None:
            cached = self.sae.feature_direction(feature_id, normalize=True).to(self.device)
            self._direction_cache[feature_id] = cached
        return cached

    def build_delta(self, edits: list[PlannedEdit], hidden: torch.Tensor) -> torch.Tensor | None:
        """Combine planned edits into one residual-stream delta.

        Returns ``None`` when there is nothing to do, which propagates through
        :class:`~brainpatch.ml.hooks.ResidualInjector` as "leave the tensor
        completely untouched".
        """
        additive = [e for e in edits if e.mode == "add"]
        ablations = [e for e in edits if e.mode == "ablate"]
        if not additive and not ablations:
            return None

        delta = torch.zeros(self.sae.d_in, dtype=torch.float32, device=self.device)
        for edit in additive:
            delta += edit.coefficient * self.direction(edit.feature_id)
        delta = delta / self.input_scale

        if ablations:
            delta = delta + self._ablation_delta(ablations, hidden)
        return delta

    def _ablation_delta(self, edits: list[PlannedEdit], hidden: torch.Tensor) -> torch.Tensor:
        """Subtract each feature's *measured* contribution at this position.

        Unlike additive steering, ablation depends on the current activation:
        it encodes the residual stream, reads how strongly the feature is
        firing right now, and removes exactly that much of its direction.
        """
        with torch.no_grad():
            x = hidden[:, -1, :].to(torch.float32) * self.input_scale
            sparse, _, _ = self.sae.encode(x)
            delta = torch.zeros(self.sae.d_in, dtype=torch.float32, device=self.device)
            for edit in edits:
                magnitude = sparse[:, edit.feature_id].mean()
                # coefficient scales how much of the contribution is removed:
                # -1.0 removes it entirely, -0.5 halves it.
                delta += edit.coefficient * magnitude * self.direction(edit.feature_id)
            return delta / self.input_scale

    # -- hook callback ---------------------------------------------------------

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor | None:
        """Delta callback for :class:`~brainpatch.ml.hooks.ResidualInjector`."""
        self.stats.forward_passes += 1
        is_prompt_pass = self._pass_index == 0
        self._pass_index += 1
        token_index = max(0, self.generated_index)

        def skip() -> None:
            self.stats.per_token_norms.append((self.generated_index, 0.0))

        if is_prompt_pass and not self.apply_to_prompt:
            skip()
            return None

        edits = self.plan.edits_at(token_index, layer=self.layer)
        if not edits:
            skip()
            return None

        delta = self.build_delta(edits, hidden)
        if delta is None:
            skip()
            return None

        norm = float(delta.norm().item())
        self.stats.applied_passes += 1
        self.stats.total_delta_norm += norm
        self.stats.max_delta_norm = max(self.stats.max_delta_norm, norm)
        self.stats.per_token_norms.append((self.generated_index, norm))

        # Broadcast over [batch, seq, hidden]. During the prompt pass this
        # steers every prompt position; afterwards there is only one position.
        return delta.view(1, 1, -1)

    def make_injector(self) -> ResidualInjector:
        return ResidualInjector(self, name=f"steer-L{self.layer}")


class RandomDirectionSteerer(FeatureSteerer):
    """Scale-matched random-direction control.

    Replaces every decoder column with a fixed random unit vector drawn from a
    seeded generator. Because both are unit-norm and the coefficient path is
    unchanged, the injected delta has *exactly* the same L2 norm as the real
    intervention it controls for. Any behavioural difference between the two is
    therefore attributable to direction, not magnitude.
    """

    def __init__(self, *args, control_seed: int = 1234, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.control_seed = control_seed

    def direction(self, feature_id: int) -> torch.Tensor:
        cached = self._direction_cache.get(feature_id)
        if cached is None:
            generator = torch.Generator(device="cpu").manual_seed(self.control_seed + feature_id)
            vector = torch.randn(self.sae.d_in, generator=generator)
            cached = (vector / vector.norm()).to(self.device)
            self._direction_cache[feature_id] = cached
        return cached


def make_steerer(
    sae: TopKSAE,
    plan: InterventionPlan,
    *,
    layer: int,
    input_scale: float,
    device: torch.device | str = "cuda",
    control: str = "none",
    control_seed: int = 1234,
    apply_to_prompt: bool = True,
) -> FeatureSteerer:
    """Build the steerer for a condition.

    Parameters
    ----------
    control:
        ``"none"`` for the real intervention, ``"random"`` for the scale-matched
        random-direction control. The unrelated-feature control is expressed by
        building a plan over different feature IDs, not by a different class.
    """
    if control == "random":
        return RandomDirectionSteerer(
            sae,
            plan,
            layer=layer,
            input_scale=input_scale,
            device=device,
            apply_to_prompt=apply_to_prompt,
            control_seed=control_seed,
        )
    if control != "none":
        raise ValueError(f"unknown control type {control!r}; expected 'none' or 'random'")
    return FeatureSteerer(
        sae,
        plan,
        layer=layer,
        input_scale=input_scale,
        device=device,
        apply_to_prompt=apply_to_prompt,
    )
