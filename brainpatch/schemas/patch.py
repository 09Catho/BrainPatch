"""The BrainPatch file format.

A BrainPatch is a tiny, inspectable, shareable JSON file describing an
activation-space intervention on a *specific* model at a *specific* hook site.

The format carries enough provenance to make misapplication a loud error rather
than a silent one. Applying a patch trained on Qwen's layer-18 residual stream
to a Gemma model, or to a different layer, or against a different SAE, produces
different arithmetic on unrelated directions -- it does not "sort of work". So
:meth:`BrainPatchSpec.check_compatibility` refuses all of those cases.

Example
-------
::

    {
      "format_version": "0.1",
      "name": "experimental-feature-1207",
      "description": "Unvalidated single-feature steering direction.",
      "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
      "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
      "sae": {
        "reference": "smoke_v0",
        "layer": 18,
        "hook": "residual_post",
        "d_in": 1536,
        "d_sae": 2048
      },
      "features": [{"feature_id": 1207, "strength": 1.5}],
      "evidence_level": "interventional",
      "evaluation": {},
      "license": "Apache-2.0",
      "authors": []
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from brainpatch.schemas.feature import CONTROLLED_LEVELS, EVIDENCE_ORDER, EvidenceLevel

PATCH_FORMAT_VERSION = "0.1"

#: Format versions this build knows how to load.
SUPPORTED_FORMAT_VERSIONS = frozenset({"0.1"})

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class PatchValidationError(ValueError):
    """The patch file is malformed or self-inconsistent."""


class PatchCompatibilityError(ValueError):
    """The patch is well-formed but does not match the target model/SAE."""


@dataclass
class FeatureEdit:
    """One feature direction to add to the residual stream.

    ``strength`` is measured in units of ``input_scale`` along the unit-norm
    decoder column (see :mod:`brainpatch.schemas.sae`). Positive amplifies,
    negative suppresses. ``mode`` distinguishes plain addition from ablation,
    which needs the SAE encoder at runtime rather than just the decoder column.
    """

    feature_id: int
    strength: float
    #: ``"add"`` injects ``strength * scale * d_f``.
    #: ``"ablate"`` projects out the feature's *measured* contribution instead.
    mode: str = "add"

    def validate(self, *, d_sae: int | None = None) -> None:
        if self.feature_id < 0:
            raise PatchValidationError(f"feature_id must be non-negative, got {self.feature_id}")
        if d_sae is not None and self.feature_id >= d_sae:
            raise PatchValidationError(
                f"feature_id {self.feature_id} is out of range for a dictionary of size {d_sae}"
            )
        if self.mode not in {"add", "ablate"}:
            raise PatchValidationError(f"unknown feature edit mode {self.mode!r}")
        if not isinstance(self.strength, (int, float)):
            raise PatchValidationError(f"strength must be numeric, got {type(self.strength)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureEdit":
        if "feature_id" not in data:
            raise PatchValidationError("feature edit is missing 'feature_id'")
        if "strength" not in data:
            raise PatchValidationError("feature edit is missing 'strength'")
        return cls(
            feature_id=int(data["feature_id"]),
            strength=float(data["strength"]),
            mode=str(data.get("mode", "add")),
        )


@dataclass
class SAEReference:
    """Identifies the sparse autoencoder a patch's feature IDs refer to."""

    reference: str
    """Experiment name / repo path locating the SAE checkpoint."""
    layer: int
    hook: str
    d_in: int
    d_sae: int
    #: Multiplier that maps normalized SAE space back to raw residual scale.
    input_scale: float | None = None
    sha256: str | None = None

    def validate(self) -> None:
        if not self.reference:
            raise PatchValidationError("sae.reference must be a non-empty string")
        if self.layer < 0:
            raise PatchValidationError(f"sae.layer must be non-negative, got {self.layer}")
        if not self.hook:
            raise PatchValidationError("sae.hook must be a non-empty string")
        if self.d_in <= 0 or self.d_sae <= 0:
            raise PatchValidationError("sae.d_in and sae.d_sae must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SAEReference":
        missing = {"reference", "layer", "hook", "d_in", "d_sae"} - set(data)
        if missing:
            raise PatchValidationError(f"sae block is missing keys: {sorted(missing)}")
        return cls(
            reference=str(data["reference"]),
            layer=int(data["layer"]),
            hook=str(data["hook"]),
            d_in=int(data["d_in"]),
            d_sae=int(data["d_sae"]),
            input_scale=(None if data.get("input_scale") is None else float(data["input_scale"])),
            sha256=data.get("sha256"),
        )


@dataclass
class BrainPatchSpec:
    """A complete, portable BrainPatch definition."""

    name: str
    base_model: str
    sae: SAEReference
    features: list[FeatureEdit]
    description: str = ""
    model_revision: str | None = None
    #: Optional token-index -> strength-multiplier schedule for dynamic steering.
    schedule: dict[str, float] | None = None
    #: Measured results. Empty dict means "not evaluated", never "it works".
    evaluation: dict[str, Any] = field(default_factory=dict)
    evidence_level: EvidenceLevel = "none"
    license: str = "Apache-2.0"
    authors: list[str] = field(default_factory=list)
    format_version: str = PATCH_FORMAT_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- validation ------------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`PatchValidationError` if this patch is malformed."""
        if self.format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise PatchValidationError(
                f"unsupported format_version {self.format_version!r}; "
                f"this build supports {sorted(SUPPORTED_FORMAT_VERSIONS)}"
            )
        if not _NAME_RE.match(self.name):
            raise PatchValidationError(
                f"patch name {self.name!r} must be lowercase alphanumeric with "
                "'.', '_' or '-', 1-64 characters"
            )
        if not self.base_model:
            raise PatchValidationError("base_model must be a non-empty string")
        if self.evidence_level not in EVIDENCE_ORDER:
            raise PatchValidationError(
                f"unknown evidence_level {self.evidence_level!r}; expected one of {EVIDENCE_ORDER}"
            )
        self.sae.validate()
        if not self.features:
            raise PatchValidationError("a patch must edit at least one feature")

        seen: set[int] = set()
        for edit in self.features:
            edit.validate(d_sae=self.sae.d_sae)
            if edit.feature_id in seen:
                raise PatchValidationError(
                    f"feature_id {edit.feature_id} appears more than once; "
                    "merge the edits into a single entry"
                )
            seen.add(edit.feature_id)

        if self.schedule is not None:
            self._validate_schedule()

    def _validate_schedule(self) -> None:
        assert self.schedule is not None
        if not self.schedule:
            raise PatchValidationError("schedule, if present, must be non-empty")
        for key, value in self.schedule.items():
            try:
                step = int(key)
            except (TypeError, ValueError) as exc:
                raise PatchValidationError(
                    f"schedule keys must be integer token indices, got {key!r}"
                ) from exc
            if step < 0:
                raise PatchValidationError(f"schedule token index must be >= 0, got {step}")
            if not isinstance(value, (int, float)):
                raise PatchValidationError(
                    f"schedule value for step {step} must be numeric, got {value!r}"
                )

    # -- compatibility ---------------------------------------------------------

    def check_compatibility(
        self,
        *,
        model: str,
        hidden_size: int,
        num_layers: int | None = None,
        model_revision: str | None = None,
        sae_reference: str | None = None,
        sae_d_sae: int | None = None,
        strict_revision: bool = False,
    ) -> None:
        """Refuse to apply this patch to an incompatible target.

        Parameters
        ----------
        model:
            Hugging Face id of the model the patch is about to be applied to.
        hidden_size:
            Residual width of that model.
        num_layers:
            Layer count, used to reject an out-of-range hook layer.
        model_revision:
            Revision actually loaded. Compared against the patch's recorded
            revision when both are known.
        sae_reference, sae_d_sae:
            Identity of the SAE currently loaded in the runtime.
        strict_revision:
            When True a revision mismatch is an error rather than tolerated.
            Feature directions are properties of a specific set of weights, so
            production use should set this.

        Raises
        ------
        PatchCompatibilityError
            On any mismatch.
        """
        if model != self.base_model:
            raise PatchCompatibilityError(
                f"patch {self.name!r} targets base model {self.base_model!r} "
                f"but was applied to {model!r}. Feature directions are not "
                "transferable between models."
            )
        if hidden_size != self.sae.d_in:
            raise PatchCompatibilityError(
                f"patch {self.name!r} expects hidden size {self.sae.d_in} "
                f"but the model has {hidden_size}"
            )
        if num_layers is not None and self.sae.layer >= num_layers:
            raise PatchCompatibilityError(
                f"patch {self.name!r} hooks layer {self.sae.layer} but the model "
                f"only has {num_layers} layers"
            )
        if (
            strict_revision
            and self.model_revision is not None
            and model_revision is not None
            and model_revision != self.model_revision
        ):
            raise PatchCompatibilityError(
                f"patch {self.name!r} was derived from revision "
                f"{self.model_revision!r} but revision {model_revision!r} is loaded"
            )
        if sae_reference is not None and sae_reference != self.sae.reference:
            raise PatchCompatibilityError(
                f"patch {self.name!r} refers to SAE {self.sae.reference!r} "
                f"but SAE {sae_reference!r} is loaded; feature IDs are not "
                "comparable across SAEs"
            )
        if sae_d_sae is not None and sae_d_sae != self.sae.d_sae:
            raise PatchCompatibilityError(
                f"patch {self.name!r} expects a dictionary of size {self.sae.d_sae} "
                f"but the loaded SAE has {sae_d_sae}"
            )

    # -- convenience -----------------------------------------------------------

    @property
    def has_controlled_evidence(self) -> bool:
        """True once scale-matched controls have been run and passed."""
        return self.evidence_level in CONTROLLED_LEVELS

    @property
    def is_validated(self) -> bool:
        """True only for a controlled result that survived independent repetition.

        One passing controlled experiment is ``controlled_interventional``. It
        takes a replication to be called validated.
        """
        return self.evidence_level == "replicated"

    def summary(self) -> str:
        """One-line human summary that does not overstate evidence."""
        edits = ", ".join(f"#{e.feature_id}@{e.strength:+.2f}" for e in self.features)
        status = "validated" if self.is_validated else f"{self.evidence_level}/unvalidated"
        return f"{self.name} [{status}] L{self.sae.layer} {edits}"

    # -- serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "name": self.name,
            "description": self.description,
            "base_model": self.base_model,
            "model_revision": self.model_revision,
            "sae": self.sae.to_dict(),
            "features": [e.to_dict() for e in self.features],
            "schedule": self.schedule,
            "evaluation": self.evaluation,
            "evidence_level": self.evidence_level,
            "license": self.license,
            "authors": list(self.authors),
            "metadata": self.metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrainPatchSpec":
        """Parse and validate a patch dictionary."""
        if not isinstance(data, dict):
            raise PatchValidationError(f"patch must be a JSON object, got {type(data).__name__}")
        for key in ("name", "base_model", "sae", "features"):
            if key not in data:
                raise PatchValidationError(f"patch is missing required key {key!r}")
        if not isinstance(data["features"], list):
            raise PatchValidationError("'features' must be a list")

        spec = cls(
            name=str(data["name"]),
            base_model=str(data["base_model"]),
            sae=SAEReference.from_dict(data["sae"]),
            features=[FeatureEdit.from_dict(f) for f in data["features"]],
            description=str(data.get("description", "")),
            model_revision=data.get("model_revision"),
            schedule=data.get("schedule"),
            evaluation=dict(data.get("evaluation", {})),
            evidence_level=data.get("evidence_level", "none"),
            license=str(data.get("license", "Apache-2.0")),
            authors=list(data.get("authors", [])),
            format_version=str(data.get("format_version", PATCH_FORMAT_VERSION)),
            metadata=dict(data.get("metadata", {})),
        )
        spec.validate()
        return spec

    @classmethod
    def from_json(cls, text: str) -> "BrainPatchSpec":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PatchValidationError(f"patch is not valid JSON: {exc}") from exc
        return cls.from_dict(data)
