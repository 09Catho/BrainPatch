"""BrainPatch format validation and compatibility refusal.

The compatibility tests matter more than they look. A feature direction is a
property of one specific set of weights at one specific layer under one specific
SAE. Applying a Qwen patch to Gemma is not "degraded" -- it is adding an
arbitrary vector to an unrelated coordinate system. These tests are what make
that failure loud.
"""

from __future__ import annotations

import json

import pytest

from brainpatch.schemas.patch import (
    BrainPatchSpec,
    FeatureEdit,
    PatchCompatibilityError,
    PatchValidationError,
    SAEReference,
)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_valid_patch_parses(valid_patch_dict):
    spec = BrainPatchSpec.from_dict(valid_patch_dict)
    assert spec.name == "test-patch"
    assert spec.sae.layer == 18
    assert len(spec.features) == 1
    assert spec.features[0].feature_id == 727


def test_round_trip_is_stable(valid_patch_dict):
    spec = BrainPatchSpec.from_dict(valid_patch_dict)
    again = BrainPatchSpec.from_json(spec.to_json())
    assert again.to_dict() == spec.to_dict()


@pytest.mark.parametrize("missing", ["name", "base_model", "sae", "features"])
def test_missing_required_key_rejected(valid_patch_dict, missing):
    del valid_patch_dict[missing]
    with pytest.raises(PatchValidationError, match=missing):
        BrainPatchSpec.from_dict(valid_patch_dict)


def test_unsupported_format_version_rejected(valid_patch_dict):
    valid_patch_dict["format_version"] = "99.0"
    with pytest.raises(PatchValidationError, match="format_version"):
        BrainPatchSpec.from_dict(valid_patch_dict)


@pytest.mark.parametrize("name", ["Has Spaces", "UPPER", "", "x" * 65, "-leading"])
def test_invalid_names_rejected(valid_patch_dict, name):
    valid_patch_dict["name"] = name
    with pytest.raises(PatchValidationError):
        BrainPatchSpec.from_dict(valid_patch_dict)


def test_empty_feature_list_rejected(valid_patch_dict):
    valid_patch_dict["features"] = []
    with pytest.raises(PatchValidationError, match="at least one feature"):
        BrainPatchSpec.from_dict(valid_patch_dict)


def test_duplicate_feature_ids_rejected(valid_patch_dict):
    valid_patch_dict["features"] = [
        {"feature_id": 5, "strength": 1.0},
        {"feature_id": 5, "strength": 2.0},
    ]
    with pytest.raises(PatchValidationError, match="more than once"):
        BrainPatchSpec.from_dict(valid_patch_dict)


def test_feature_id_beyond_dictionary_rejected(valid_patch_dict):
    valid_patch_dict["features"] = [{"feature_id": 999_999, "strength": 1.0}]
    with pytest.raises(PatchValidationError, match="out of range"):
        BrainPatchSpec.from_dict(valid_patch_dict)


def test_unknown_edit_mode_rejected(valid_patch_dict):
    valid_patch_dict["features"] = [{"feature_id": 1, "strength": 1.0, "mode": "teleport"}]
    with pytest.raises(PatchValidationError, match="mode"):
        BrainPatchSpec.from_dict(valid_patch_dict)


def test_unknown_evidence_level_rejected(valid_patch_dict):
    valid_patch_dict["evidence_level"] = "definitely-proven"
    with pytest.raises(PatchValidationError, match="evidence_level"):
        BrainPatchSpec.from_dict(valid_patch_dict)


def test_malformed_json_rejected():
    with pytest.raises(PatchValidationError, match="not valid JSON"):
        BrainPatchSpec.from_json("{not json")


def test_non_object_rejected():
    with pytest.raises(PatchValidationError, match="JSON object"):
        BrainPatchSpec.from_dict(json.loads("[1, 2, 3]"))


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------


def test_valid_schedule_accepted(valid_patch_dict):
    valid_patch_dict["schedule"] = {"0": 0.0, "24": 1.0, "48": 2.0}
    spec = BrainPatchSpec.from_dict(valid_patch_dict)
    assert spec.schedule == {"0": 0.0, "24": 1.0, "48": 2.0}


@pytest.mark.parametrize(
    "schedule", [{}, {"abc": 1.0}, {"-1": 1.0}, {"0": "loud"}]
)
def test_invalid_schedules_rejected(valid_patch_dict, schedule):
    valid_patch_dict["schedule"] = schedule
    with pytest.raises(PatchValidationError):
        BrainPatchSpec.from_dict(valid_patch_dict)


# ---------------------------------------------------------------------------
# compatibility -- the part that prevents silent nonsense
# ---------------------------------------------------------------------------


def _spec(**overrides) -> BrainPatchSpec:
    base = dict(
        name="p",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        model_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        sae=SAEReference(
            reference="smoke_v0", layer=18, hook="residual_post", d_in=1536, d_sae=2048
        ),
        features=[FeatureEdit(feature_id=727, strength=1.0)],
    )
    base.update(overrides)
    spec = BrainPatchSpec(**base)  # type: ignore[arg-type]
    spec.validate()
    return spec


def test_compatible_target_accepted():
    _spec().check_compatibility(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        hidden_size=1536,
        num_layers=28,
        sae_reference="smoke_v0",
        sae_d_sae=2048,
    )


def test_wrong_model_refused():
    with pytest.raises(PatchCompatibilityError, match="not.*transferable|targets base model"):
        _spec().check_compatibility(model="google/gemma-2-2b", hidden_size=1536)


def test_wrong_hidden_size_refused():
    with pytest.raises(PatchCompatibilityError, match="hidden size"):
        _spec().check_compatibility(model="Qwen/Qwen2.5-1.5B-Instruct", hidden_size=2048)


def test_layer_beyond_model_depth_refused():
    with pytest.raises(PatchCompatibilityError, match="only has"):
        _spec().check_compatibility(
            model="Qwen/Qwen2.5-1.5B-Instruct", hidden_size=1536, num_layers=12
        )


def test_wrong_sae_refused():
    with pytest.raises(PatchCompatibilityError, match="not.*comparable across SAEs"):
        _spec().check_compatibility(
            model="Qwen/Qwen2.5-1.5B-Instruct", hidden_size=1536, sae_reference="serious_v1"
        )


def test_wrong_dictionary_size_refused():
    with pytest.raises(PatchCompatibilityError, match="dictionary of size"):
        _spec().check_compatibility(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            hidden_size=1536,
            sae_reference="smoke_v0",
            sae_d_sae=16384,
        )


def test_revision_mismatch_tolerated_by_default_but_refused_when_strict():
    spec = _spec()
    # default: tolerated, since a revision is often unknown at runtime
    spec.check_compatibility(
        model="Qwen/Qwen2.5-1.5B-Instruct", hidden_size=1536, model_revision="deadbeef"
    )
    with pytest.raises(PatchCompatibilityError, match="revision"):
        spec.check_compatibility(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            hidden_size=1536,
            model_revision="deadbeef",
            strict_revision=True,
        )


# ---------------------------------------------------------------------------
# honesty of the evidence field
# ---------------------------------------------------------------------------


def test_default_patch_is_not_validated():
    assert _spec().is_validated is False


def test_summary_marks_unvalidated_patches():
    assert "unvalidated" in _spec().summary()


def test_summary_does_not_mark_validated_patches():
    assert "unvalidated" not in _spec(evidence_level="causal").summary()
