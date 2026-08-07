"""``BrainPatchedModel`` -- the user-facing runtime.

::

    from brainpatch import BrainPatchedModel

    model = BrainPatchedModel.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model.load_sae("/vol/sae/smoke_v0/sae_latest.pt", reference="smoke_v0")
    model.install("patches/experimental-feature-1207.json")
    model.set_patch_strength("experimental-feature-1207", 1.5)
    print(model.generate("Solve this problem..."))

The base model is frozen. Every behavioural change comes from a hook that adds
a vector to one layer's residual stream, installed for the duration of a
generation and removed afterwards.

Two invariants this class exists to guarantee:

1. **No patches installed, or all strengths zero, is exactly baseline.** With
   nothing to apply, the hook returns ``None`` and the tensor is never touched.
   :meth:`assert_zero_strength_is_baseline` verifies this empirically rather
   than trusting the argument.

2. **A patch cannot be applied to the wrong model.** Every install runs
   :meth:`~brainpatch.schemas.patch.BrainPatchSpec.check_compatibility` against
   the loaded weights and SAE.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from brainpatch.ml.generation import GenerationConfig, build_chat_prompt
from brainpatch.ml.hooks import HookSet
from brainpatch.ml.intervention import FeatureSteerer, make_steerer
from brainpatch.ml.model import DEFAULT_MODEL, ModelBundle, load_model
from brainpatch.ml.sae import TopKSAE
from brainpatch.patches.io import load_patch
from brainpatch.schemas.patch import BrainPatchSpec, FeatureEdit, SAEReference
from brainpatch.steering.plan import InterventionPlan
from brainpatch.steering.schedule import StrengthSchedule


class BrainPatchedModel:
    """A frozen language model with installable activation-space patches."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self.plan = InterventionPlan()
        self.sae: TopKSAE | None = None
        self.sae_reference: str | None = None
        self.input_scale: float | None = None
        self._last_stats: dict[str, Any] = {}
        self._last_trace: list[tuple[int, float]] = []

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
    ) -> "BrainPatchedModel":
        """Load a frozen base model. Downloads go to the HF cache, never locally."""
        return cls(load_model(model_id, revision=revision, dtype=dtype, device=device))

    def load_sae(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        reference: str,
        input_scale: float | None = None,
    ) -> TopKSAE:
        """Attach a trained SAE, whose decoder columns become patch directions.

        Raises
        ------
        ValueError
            If the SAE's input width does not match the model's residual width,
            which would mean the two were never trained on the same activations.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"SAE checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        sae = TopKSAE.from_checkpoint(checkpoint, device=str(self.bundle.device))

        if sae.d_in != self.bundle.hidden_size:
            raise ValueError(
                f"SAE was trained on width {sae.d_in} but the model's residual "
                f"stream is {self.bundle.hidden_size} wide -- these are not the "
                "same activations"
            )

        scale = input_scale if input_scale is not None else sae.config.input_scale
        if scale is None:
            raise ValueError(
                "SAE checkpoint has no recorded input_scale, so a strength value "
                "would have no defined magnitude. Pass input_scale explicitly."
            )

        self.sae = sae
        self.sae_reference = reference
        self.input_scale = float(scale)
        return sae

    # -- patch management ------------------------------------------------------

    def install(
        self,
        patch: str | os.PathLike[str] | BrainPatchSpec,
        *,
        strength: float = 1.0,
        strict_revision: bool = False,
    ) -> BrainPatchSpec:
        """Install a patch from a file path or an in-memory spec.

        The compatibility check runs before anything is registered, so a
        rejected patch leaves the runtime untouched.
        """
        spec = patch if isinstance(patch, BrainPatchSpec) else load_patch(patch)
        spec.check_compatibility(
            model=self.bundle.model_id,
            hidden_size=self.bundle.hidden_size,
            num_layers=self.bundle.num_layers,
            model_revision=self.bundle.revision,
            sae_reference=self.sae_reference,
            sae_d_sae=self.sae.d_sae if self.sae is not None else None,
            strict_revision=strict_revision,
        )
        self.plan.install(spec, strength=strength)
        return spec

    def uninstall(self, name: str) -> None:
        self.plan.uninstall(name)

    def list_patches(self) -> list[str]:
        return list(self.plan.patches)

    def set_patch_strength(self, name: str, strength: float) -> None:
        """Change a patch's strength. Takes effect on the next generation."""
        self.plan.set_strength(name, strength)

    def set_patch_enabled(self, name: str, enabled: bool) -> None:
        self.plan.set_enabled(name, enabled)

    def set_patch_schedule(self, name: str, schedule: dict[int, float] | StrengthSchedule | None) -> None:
        """Install a token-indexed strength schedule for dynamic steering."""
        if isinstance(schedule, dict):
            schedule = StrengthSchedule(schedule)
        self.plan.set_schedule(name, schedule)

    def add_feature(
        self,
        *,
        layer: int,
        feature_id: int,
        strength: float,
        name: str | None = None,
        mode: str = "add",
    ) -> BrainPatchSpec:
        """Install an ad-hoc single-feature intervention.

        The convenience path for exploration. It builds a real
        :class:`BrainPatchSpec` under the hood, so an interactive experiment and
        a shipped patch go through identical machinery -- and the resulting spec
        can be saved directly.
        """
        if self.sae is None or self.sae_reference is None or self.input_scale is None:
            raise RuntimeError("load_sae() must be called before adding features")
        if layer != self.bundle_layer_for_sae():
            # Not fatal, but the SAE only means anything at the layer it was fitted on.
            raise ValueError(
                f"SAE {self.sae_reference!r} was trained at layer "
                f"{self.sae.config.layer}; refusing to inject its directions at layer {layer}"
            )
        spec = BrainPatchSpec(
            name=name or f"adhoc-feature-{feature_id}",
            base_model=self.bundle.model_id,
            model_revision=self.bundle.revision,
            sae=SAEReference(
                reference=self.sae_reference,
                layer=layer,
                hook=self.sae.config.hook or "residual_post",
                d_in=self.sae.d_in,
                d_sae=self.sae.d_sae,
                input_scale=self.input_scale,
            ),
            features=[FeatureEdit(feature_id=feature_id, strength=strength, mode=mode)],
            description="Ad-hoc exploratory intervention. No evidence of any behavioural effect.",
            evidence_level="none",
        )
        return self.install(spec)

    def bundle_layer_for_sae(self) -> int:
        """The layer the attached SAE was trained on."""
        if self.sae is None:
            raise RuntimeError("no SAE loaded")
        return int(self.sae.config.layer)

    # -- generation ------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        config: GenerationConfig | None = None,
        system: str | None = None,
        use_chat_template: bool = True,
        control: str = "none",
        control_seed: int = 1234,
        apply_to_prompt: bool = True,
    ) -> str:
        """Generate a completion with all installed patches active.

        Parameters
        ----------
        control:
            ``"random"`` swaps every feature direction for a scale-matched
            random one, keeping all else identical. This is the control
            condition, run through exactly the same code path as the real
            intervention.
        """
        cfg = config or GenerationConfig()
        text = build_chat_prompt(self.bundle.tokenizer, prompt, system) if use_chat_template else prompt
        inputs = self.bundle.tokenizer(text, return_tensors="pt").to(self.bundle.device)

        if cfg.do_sample:
            torch.manual_seed(cfg.seed)

        steerer = self._build_steerer(control=control, control_seed=control_seed,
                                      apply_to_prompt=apply_to_prompt)

        with HookSet() as hooks:
            if steerer is not None:
                steerer.reset()
                hooks.add(steerer.make_injector(), self.bundle.layer_module(steerer.layer))
            with torch.inference_mode():
                output = self.bundle.model.generate(
                    **inputs, **cfg.to_kwargs(self.bundle.tokenizer)
                )

        self._last_stats = steerer.stats.to_dict() if steerer is not None else {}
        self._last_trace = list(steerer.stats.per_token_norms) if steerer is not None else []
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.bundle.tokenizer.decode(generated, skip_special_tokens=True)

    @property
    def last_steering_stats(self) -> dict[str, Any]:
        """Delta norms and application counts from the most recent generation.

        The empirical answer to "did the intervention actually fire?" -- an
        ``applied_passes`` of 0 means the plan resolved to nothing.
        """
        return dict(self._last_stats)

    @property
    def last_steering_trace(self) -> list[tuple[int, float]]:
        """``(generated_token_index, delta_norm)`` for every pass of the last run.

        The evidence that a dynamic schedule fired where it was supposed to:
        the norm should be zero before the keyframe and non-zero after.
        """
        return list(self._last_trace)

    def _build_steerer(
        self, *, control: str, control_seed: int, apply_to_prompt: bool
    ) -> FeatureSteerer | None:
        """Construct a steerer, or ``None`` when there is nothing to apply."""
        layers = self.plan.layers()
        if not layers:
            return None
        if self.sae is None or self.input_scale is None:
            raise RuntimeError("patches are installed but no SAE is loaded")
        if len(layers) > 1:
            raise NotImplementedError(
                f"patches span layers {layers}; multi-layer steering needs one SAE "
                "per layer and is not supported in v0"
            )
        return make_steerer(
            self.sae,
            self.plan,
            layer=layers[0],
            input_scale=self.input_scale,
            device=self.bundle.device,
            control=control,
            control_seed=control_seed,
            apply_to_prompt=apply_to_prompt,
        )

    # -- verification ----------------------------------------------------------

    def assert_zero_strength_is_baseline(
        self, prompt: str, *, config: GenerationConfig | None = None
    ) -> dict[str, Any]:
        """Empirically verify that zeroed patches reproduce baseline exactly.

        Generates with every patch uninstalled, then with them installed at
        strength 0, and compares the strings. This is a correctness test of the
        hook machinery: if it ever fails, every "baseline" in every experiment
        is suspect.
        """
        cfg = config or GenerationConfig(max_new_tokens=48)
        saved = {name: p.strength for name, p in self.plan.patches.items()}

        installed = dict(self.plan.patches)
        self.plan.patches = {}
        baseline = self.generate(prompt, config=cfg)

        self.plan.patches = installed
        for name in self.plan.patches:
            self.plan.set_strength(name, 0.0)
        zeroed = self.generate(prompt, config=cfg)
        zero_stats = self.last_steering_stats

        for name, strength in saved.items():
            self.plan.set_strength(name, strength)

        return {
            "identical": baseline == zeroed,
            "baseline": baseline,
            "zero_strength": zeroed,
            "applied_passes_at_zero": zero_stats.get("applied_passes", 0),
            "num_patches": len(saved),
        }

    def describe(self) -> dict[str, Any]:
        return {
            **self.bundle.describe(),
            "sae_reference": self.sae_reference,
            "sae_d_sae": self.sae.d_sae if self.sae else None,
            "sae_k": self.sae.k if self.sae else None,
            "sae_layer": self.sae.config.layer if self.sae else None,
            "input_scale": self.input_scale,
            "installed_patches": self.plan.describe(),
        }
