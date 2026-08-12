"""Provenance validation in the v1 manifest.

The point of these fields is that a patch can be *compared* with another one:
which method found the direction, at which layer and token position, calibrated
how, against which dataset. A patch that records that badly is worse than one
that records nothing, because it reads like an audit trail.
"""

from __future__ import annotations

import pytest

from brainpatch.patch.format import (
    KNOWN_DISCOVERY_METHODS,
    KNOWN_EXTRACTION_POSITIONS,
    KNOWN_INJECTION_SITES,
    PROVENANCE_MAX_BYTES,
    BaseModelSpec,
    Intervention,
    Manifest,
    PatchFormatError,
    validate_provenance,
)

pytestmark = pytest.mark.local

DIGEST = "a" * 64


def make_manifest(provenance: dict) -> Manifest:
    return Manifest(
        name="test-patch",
        base_model=BaseModelSpec(model_id="Qwen/Qwen2.5-1.5B-Instruct", hidden_size=1536, num_layers=28),
        interventions=[Intervention(layer=18, vector="v0", coefficient=1.0)],
        provenance=provenance,
    )


def test_empty_provenance_is_valid():
    """Patches predating these fields must keep loading."""
    make_manifest({}).validate()


def test_full_provenance_block_validates():
    make_manifest(
        {
            "discovery_method": "pca",
            "discovery_layer": 24,
            "extraction_position": "cont_mean",
            "injection_site": "prompt",
            "training_dataset_hash": DIGEST,
            "normalization": "unit_l2",
            "strength_calibration": {
                "basis": "median_residual_norm",
                "ratio": 0.1,
                "p50": 162.4,
            },
        }
    ).validate()


@pytest.mark.parametrize("method", KNOWN_DISCOVERY_METHODS)
def test_every_known_method_is_accepted(method):
    validate_provenance({"discovery_method": method})


@pytest.mark.parametrize("position", KNOWN_EXTRACTION_POSITIONS)
def test_every_known_extraction_position_is_accepted(position):
    validate_provenance({"extraction_position": position})


@pytest.mark.parametrize("site", KNOWN_INJECTION_SITES)
def test_every_known_injection_site_is_accepted(site):
    validate_provenance({"injection_site": site})


def test_unknown_discovery_method_is_rejected():
    with pytest.raises(PatchFormatError, match="discovery_method"):
        validate_provenance({"discovery_method": "vibes"})


def test_unknown_injection_site_is_rejected():
    with pytest.raises(PatchFormatError, match="injection_site"):
        validate_provenance({"injection_site": "everywhere"})


@pytest.mark.parametrize("bad", ["not-a-hash", "A" * 64, "abc", 12345, "a" * 63])
def test_bad_dataset_hash_is_rejected(bad):
    with pytest.raises(PatchFormatError, match="training_dataset_hash"):
        validate_provenance({"training_dataset_hash": bad})


@pytest.mark.parametrize("bad", [-1, "18", 1.5, True])
def test_bad_discovery_layer_is_rejected(bad):
    with pytest.raises(PatchFormatError, match="discovery_layer"):
        validate_provenance({"discovery_layer": bad})


@pytest.mark.parametrize(
    "key", ["examples", "prompts", "dataset", "training_data", "samples", "corpus", "responses"]
)
def test_training_data_cannot_be_smuggled_into_a_patch(key):
    with pytest.raises(PatchFormatError, match="training data"):
        validate_provenance({key: ["some prompt", "another prompt"]})


def test_forbidden_key_check_is_case_insensitive():
    with pytest.raises(PatchFormatError, match="training data"):
        validate_provenance({"Examples": ["a"]})


def test_oversized_provenance_is_rejected():
    """A patch is a direction plus its audit trail, not a container."""
    with pytest.raises(PatchFormatError, match="cap"):
        validate_provenance({"notes": "x" * (PROVENANCE_MAX_BYTES + 1)})


def test_provenance_just_under_the_cap_is_accepted():
    validate_provenance({"notes": "x" * (PROVENANCE_MAX_BYTES - 200)})


def test_strength_calibration_must_be_structured():
    with pytest.raises(PatchFormatError, match="strength_calibration"):
        validate_provenance({"strength_calibration": "about 10 percent"})


def test_manifest_round_trips_provenance():
    provenance = {
        "discovery_method": "caa",
        "discovery_layer": 18,
        "injection_site": "all",
        "training_dataset_hash": DIGEST,
    }
    manifest = make_manifest(provenance)
    manifest.validate()
    restored = Manifest.from_dict(manifest.to_dict())
    restored.validate()
    assert restored.provenance == provenance


def test_invalid_provenance_fails_whole_manifest_validation():
    """It must fail at the manifest level, not only when called directly."""
    with pytest.raises(PatchFormatError):
        make_manifest({"discovery_method": "made-it-up"}).validate()
