"""BrainPatch v1: the portable, self-contained runtime format.

What changed from v0.1, and why
-------------------------------
The v0.1 patch (``brainpatch.schemas.patch``) is a *research* record: it names
SAE feature IDs and their coefficients. To apply it you need the SAE that
produced those IDs -- a 72 MB checkpoint to use three directions. That is fine
for reproducing an experiment and wrong for shipping a product.

v1 stores the **materialised intervention vectors** instead. The runtime adds a
vector to a layer's residual stream; it neither knows nor cares whether that
vector came from an SAE decoder column, a difference of means, a PCA component,
or a learned controller. Research provenance is preserved in metadata, but
nothing at runtime depends on it.

Consequences that matter:

* a three-direction patch is tens of KB, not tens of MB
* no SAE download, no Modal, no network at apply time
* the same artifact drives every backend

Container layout
----------------
A ``.brainpatch`` file is a ZIP archive::

    manifest.json        this module's schema, the only thing that is parsed
    vectors.safetensors  the intervention vectors (inert data)
    checksums.json       sha256 of every other member
    README.md            optional human-readable description

Deliberately **not** in the format: pickles, executable code, scripts, or
anything the runtime evaluates. A patch is data. See
:mod:`brainpatch.patch.validation` for the checks that keep it that way.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from brainpatch.schemas.feature import EVIDENCE_ORDER, EvidenceLevel

FORMAT_VERSION = "1.0"
SUPPORTED_FORMAT_VERSIONS = frozenset({"1.0"})

MANIFEST_NAME = "manifest.json"
VECTORS_NAME = "vectors.safetensors"
CHECKSUMS_NAME = "checksums.json"
README_NAME = "README.md"

#: File extension for a compiled runtime artifact.
SUFFIX = ".brainpatch"

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

#: Hook sites the runtime knows how to apply. ``residual_post`` is the output of
#: decoder block *i* -- where a block's contribution is visible and where an
#: injection propagates to every later block.
SUPPORTED_HOOKS = ("residual_post",)

#: Verification states a backend entry may claim. "supported" is deliberately
#: absent: a backend is either verified by a real test or it is not.
COMPATIBILITY_STATES = ("verified", "experimental", "implemented", "unsupported")

#: Absolute ceiling on any coefficient, regardless of what a manifest asks for.
#: A patch is untrusted input; an enormous coefficient is a denial-of-service
#: on output quality at best.
ABSOLUTE_MAX_STRENGTH = 1024.0

#: How a direction was found. Recorded because it turned out to matter: on the
#: one behavioural task measured so far, PCA, a linear probe, difference-of-means
#: and an SAE feature produced very different directions from identical data, and
#: a patch that does not say which it used cannot be compared with another.
KNOWN_DISCOVERY_METHODS = (
    "caa",
    "pca",
    "probe",
    "sae_single",
    "sae_sparse",
    "handwritten",
    "other",
)

#: Where the direction was read off during discovery.
KNOWN_EXTRACTION_POSITIONS = ("last_prompt", "cont_mean", "cont_last", "all_tokens", "other")

#: Where the runtime adds it. This is not cosmetic: measured effect differed by
#: roughly 6x between steering prompt tokens and steering generated tokens.
KNOWN_INJECTION_SITES = ("prompt", "continuation", "all")

#: Provenance is documentation, not payload. The cap is what stops a patch from
#: carrying its training set: a few kilobytes of description is provenance, a
#: megabyte of it is a dataset with extra steps.
PROVENANCE_MAX_BYTES = 16_384

#: Keys that would smuggle example text into an artifact. A patch ships a
#: direction and the facts needed to audit it -- never the data it was fit on.
FORBIDDEN_PROVENANCE_KEYS = frozenset(
    {"examples", "prompts", "dataset", "training_data", "samples", "corpus", "responses"}
)


def validate_provenance(provenance: dict[str, Any]) -> None:
    """Check the optional provenance block.

    Every field is optional -- an older patch with an empty block stays valid --
    but a field that *is* present has to mean what it says. A misrecorded layer
    or discovery method is worse than an absent one, because it looks like
    an audit trail.
    """
    if not provenance:
        return
    if not isinstance(provenance, dict):
        raise PatchFormatError("provenance must be an object")

    offending = sorted(FORBIDDEN_PROVENANCE_KEYS & {str(k).lower() for k in provenance})
    if offending:
        raise PatchFormatError(
            f"provenance may not carry training data; remove {offending}. A patch "
            "records how it was made, not what it was made from."
        )

    encoded = len(json.dumps(provenance, ensure_ascii=False).encode("utf-8"))
    if encoded > PROVENANCE_MAX_BYTES:
        raise PatchFormatError(
            f"provenance is {encoded} bytes, over the {PROVENANCE_MAX_BYTES} byte cap"
        )

    method = provenance.get("discovery_method")
    if method is not None and method not in KNOWN_DISCOVERY_METHODS:
        raise PatchFormatError(
            f"unknown discovery_method {method!r}; expected one of {KNOWN_DISCOVERY_METHODS}"
        )

    position = provenance.get("extraction_position")
    if position is not None and position not in KNOWN_EXTRACTION_POSITIONS:
        raise PatchFormatError(
            f"unknown extraction_position {position!r}; "
            f"expected one of {KNOWN_EXTRACTION_POSITIONS}"
        )

    site = provenance.get("injection_site")
    if site is not None and site not in KNOWN_INJECTION_SITES:
        raise PatchFormatError(
            f"unknown injection_site {site!r}; expected one of {KNOWN_INJECTION_SITES}"
        )

    digest = provenance.get("training_dataset_hash")
    if digest is not None:
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise PatchFormatError(
                "training_dataset_hash must be a lowercase hex sha256 digest, so a "
                f"reader can check which data produced this direction; got {digest!r}"
            )

    layer = provenance.get("discovery_layer")
    if layer is not None and (not isinstance(layer, int) or isinstance(layer, bool) or layer < 0):
        raise PatchFormatError(f"discovery_layer must be a non-negative integer, got {layer!r}")

    calibration = provenance.get("strength_calibration")
    if calibration is not None and not isinstance(calibration, dict):
        raise PatchFormatError("strength_calibration must be an object")


class PatchFormatError(ValueError):
    """The artifact is malformed, unsupported, or internally inconsistent."""


@dataclass
class Intervention:
    """One vector added to one layer's residual stream.

    Attributes
    ----------
    vector:
        Key into ``vectors.safetensors``. Several interventions may reference
        the same key.
    coefficient:
        Baked-in scale. The runtime multiplies this by the user's live strength
        and by any schedule multiplier.
    """

    layer: int
    vector: str
    coefficient: float = 1.0
    hook: str = "residual_post"
    id: str | None = None

    def validate(self, *, num_layers: int | None = None) -> None:
        if self.layer < 0:
            raise PatchFormatError(f"layer must be non-negative, got {self.layer}")
        if num_layers is not None and self.layer >= num_layers:
            raise PatchFormatError(
                f"intervention targets layer {self.layer} but the patch declares "
                f"{num_layers} layers"
            )
        if self.hook not in SUPPORTED_HOOKS:
            raise PatchFormatError(
                f"unsupported hook {self.hook!r}; supported: {SUPPORTED_HOOKS}"
            )
        if not isinstance(self.coefficient, (int, float)) or isinstance(self.coefficient, bool):
            raise PatchFormatError(f"coefficient must be numeric, got {self.coefficient!r}")
        if abs(self.coefficient) > ABSOLUTE_MAX_STRENGTH:
            raise PatchFormatError(
                f"coefficient {self.coefficient} exceeds the absolute ceiling "
                f"of {ABSOLUTE_MAX_STRENGTH}"
            )
        if not self.vector:
            raise PatchFormatError("intervention is missing a vector reference")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "layer": self.layer,
            "hook": self.hook,
            "vector": self.vector,
            "coefficient": self.coefficient,
        }
        if self.id:
            data["id"] = self.id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Intervention":
        for key in ("layer", "vector"):
            if key not in data:
                raise PatchFormatError(f"intervention is missing required key {key!r}")
        return cls(
            layer=int(data["layer"]),
            vector=str(data["vector"]),
            coefficient=float(data.get("coefficient", 1.0)),
            hook=str(data.get("hook", "residual_post")),
            id=data.get("id"),
        )


@dataclass
class BaseModelSpec:
    """Which model this patch was derived from, and how strictly to enforce it.

    Same architecture does not imply compatible directions: a vector found in
    one model's layer-18 basis means nothing in another's. The runtime's default
    ``strict`` mode requires the model id to match.
    """

    model_id: str
    architecture: str = ""
    hidden_size: int = 0
    num_layers: int = 0
    revision: str | None = None
    torch_dtype: str | None = None

    def validate(self) -> None:
        if not self.model_id:
            raise PatchFormatError("base_model.model_id must be non-empty")
        if self.hidden_size <= 0:
            raise PatchFormatError(f"base_model.hidden_size must be positive, got {self.hidden_size}")
        if self.num_layers <= 0:
            raise PatchFormatError(f"base_model.num_layers must be positive, got {self.num_layers}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "architecture": self.architecture,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "torch_dtype": self.torch_dtype,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseModelSpec":
        if "model_id" not in data:
            raise PatchFormatError("base_model is missing 'model_id'")
        return cls(
            model_id=str(data["model_id"]),
            architecture=str(data.get("architecture", "")),
            hidden_size=int(data.get("hidden_size", 0)),
            num_layers=int(data.get("num_layers", 0)),
            revision=data.get("revision"),
            torch_dtype=data.get("torch_dtype"),
        )


@dataclass
class Manifest:
    """The parsed ``manifest.json`` of a v1 artifact."""

    name: str
    base_model: BaseModelSpec
    interventions: list[Intervention]
    description: str = ""
    format_version: str = FORMAT_VERSION
    evidence_level: EvidenceLevel = "none"
    #: Measured results. ``{}`` means "not evaluated", never "it works".
    evaluation: dict[str, Any] = field(default_factory=dict)
    #: Per-backend verification state; see :data:`COMPATIBILITY_STATES`.
    compatibility: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: How this patch was produced. Runtime never reads this.
    provenance: dict[str, Any] = field(default_factory=dict)
    #: Author-declared safety envelope for the live strength multiplier.
    max_abs_strength: float = 8.0
    default_strength: float = 1.0
    license: str = "Apache-2.0"
    authors: list[str] = field(default_factory=list)
    #: Optional token-indexed schedule shipped with the patch.
    schedule: dict[str, float] | None = None

    # -- validation ------------------------------------------------------------

    def validate(self) -> None:
        if self.format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise PatchFormatError(
                f"unsupported format_version {self.format_version!r}; this build "
                f"supports {sorted(SUPPORTED_FORMAT_VERSIONS)}"
            )
        if not _NAME_RE.match(self.name):
            raise PatchFormatError(
                f"patch name {self.name!r} must be lowercase alphanumeric with "
                "'.', '_' or '-', 1-64 characters"
            )
        if self.evidence_level not in EVIDENCE_ORDER:
            raise PatchFormatError(
                f"unknown evidence_level {self.evidence_level!r}; "
                f"expected one of {EVIDENCE_ORDER}"
            )
        self.base_model.validate()

        if not self.interventions:
            raise PatchFormatError("a patch must declare at least one intervention")

        seen_ids: set[str] = set()
        for intervention in self.interventions:
            intervention.validate(num_layers=self.base_model.num_layers)
            if intervention.id:
                if intervention.id in seen_ids:
                    raise PatchFormatError(
                        f"duplicate intervention id {intervention.id!r}"
                    )
                seen_ids.add(intervention.id)

        if not 0 < self.max_abs_strength <= ABSOLUTE_MAX_STRENGTH:
            raise PatchFormatError(
                f"max_abs_strength must be in (0, {ABSOLUTE_MAX_STRENGTH}], "
                f"got {self.max_abs_strength}"
            )
        if abs(self.default_strength) > self.max_abs_strength:
            raise PatchFormatError(
                f"default_strength {self.default_strength} exceeds the patch's own "
                f"max_abs_strength {self.max_abs_strength}"
            )

        validate_provenance(self.provenance)

        for backend, entry in self.compatibility.items():
            status = entry.get("status")
            if status not in COMPATIBILITY_STATES:
                raise PatchFormatError(
                    f"compatibility[{backend!r}].status must be one of "
                    f"{COMPATIBILITY_STATES}, got {status!r}"
                )

        if self.schedule is not None:
            self._validate_schedule()

    def _validate_schedule(self) -> None:
        assert self.schedule is not None
        if not self.schedule:
            raise PatchFormatError("schedule, if present, must be non-empty")
        for key, value in self.schedule.items():
            try:
                step = int(key)
            except (TypeError, ValueError) as exc:
                raise PatchFormatError(
                    f"schedule keys must be integer token indices, got {key!r}"
                ) from exc
            if step < 0:
                raise PatchFormatError(f"schedule token index must be >= 0, got {step}")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise PatchFormatError(
                    f"schedule value at {step} must be numeric, got {value!r}"
                )

    # -- queries ---------------------------------------------------------------

    @property
    def layers(self) -> list[int]:
        """Sorted distinct layers this patch touches."""
        return sorted({i.layer for i in self.interventions})

    @property
    def vector_keys(self) -> list[str]:
        return sorted({i.vector for i in self.interventions})

    def backend_status(self, backend: str) -> str:
        """Verification state for ``backend``; ``"unsupported"`` if unlisted."""
        return str(self.compatibility.get(backend, {}).get("status", "unsupported"))

    def is_verified_on(self, backend: str) -> bool:
        return self.backend_status(backend) == "verified"

    def clamp_strength(self, strength: float) -> float:
        """Clip a live strength into the patch's declared envelope."""
        limit = self.max_abs_strength
        return max(-limit, min(limit, float(strength)))

    def summary(self) -> str:
        layers = ",".join(str(layer) for layer in self.layers)
        return (
            f"{self.name} [{self.evidence_level}] "
            f"{len(self.interventions)} intervention(s) @ L{layers} "
            f"-> {self.base_model.model_id}"
        )

    # -- serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "name": self.name,
            "description": self.description,
            "base_model": self.base_model.to_dict(),
            "interventions": [i.to_dict() for i in self.interventions],
            "schedule": self.schedule,
            "evidence_level": self.evidence_level,
            "evaluation": self.evaluation,
            "compatibility": self.compatibility,
            "provenance": self.provenance,
            "max_abs_strength": self.max_abs_strength,
            "default_strength": self.default_strength,
            "license": self.license,
            "authors": list(self.authors),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        if not isinstance(data, dict):
            raise PatchFormatError(f"manifest must be an object, got {type(data).__name__}")
        for key in ("name", "base_model", "interventions"):
            if key not in data:
                raise PatchFormatError(f"manifest is missing required key {key!r}")
        if not isinstance(data["interventions"], list):
            raise PatchFormatError("'interventions' must be a list")

        manifest = cls(
            name=str(data["name"]),
            base_model=BaseModelSpec.from_dict(data["base_model"]),
            interventions=[Intervention.from_dict(i) for i in data["interventions"]],
            description=str(data.get("description", "")),
            format_version=str(data.get("format_version", FORMAT_VERSION)),
            evidence_level=data.get("evidence_level", "none"),
            evaluation=dict(data.get("evaluation", {})),
            compatibility=dict(data.get("compatibility", {})),
            provenance=dict(data.get("provenance", {})),
            max_abs_strength=float(data.get("max_abs_strength", 8.0)),
            default_strength=float(data.get("default_strength", 1.0)),
            license=str(data.get("license", "Apache-2.0")),
            authors=list(data.get("authors", [])),
            schedule=data.get("schedule"),
        )
        manifest.validate()
        return manifest

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PatchFormatError(f"manifest is not valid JSON: {exc}") from exc
        return cls.from_dict(data)
