"""Sign handling between a v0.1 research spec and the compiled artifact.

This file exists because of a real, nearly-shipped defect.

SAE feature selection multiplies the decoder column by the sign of the contrast
effect: feature 204's effect is **−0.1003**, so discovery used the *negated*
column. ``compile_from_sae`` emits the **unsigned** column and carries sign in
the coefficient. The first `anti-sycophancy.brainpatch` was written with a
positive coefficient and was therefore behaviourally the *sign control* — its
correction rate was 0.150 against a 0.233 baseline, the opposite of the +0.167
that was measured.

Worse, the first artifact check compared the compiled vector against the
unsigned decoder column, so it reported ``cosine = 1.0`` and passed. Only the
behavioural comparison caught it.

The lesson these tests encode: a direction has a sign, and a check that cannot
see the sign is not a check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brainpatch.schemas.patch import BrainPatchSpec

pytestmark = pytest.mark.local

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "patches" / "anti-sycophancy.json"


@pytest.fixture(scope="module")
def spec_json() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def test_spec_exists_and_validates():
    BrainPatchSpec.from_json(SPEC.read_text(encoding="utf-8")).validate()


def test_strength_sign_matches_recorded_direction_sign(spec_json):
    """The coefficient must carry the sign discovery selected.

    If these disagree the artifact applies the opposite intervention while every
    numerical check that ignores sign still passes.
    """
    recorded = spec_json["metadata"]["direction_sign"]
    strength = spec_json["features"][0]["strength"]
    assert recorded in (-1, 1)
    assert (strength < 0) == (recorded < 0), (
        f"direction_sign={recorded} but strength={strength}: the compiled patch "
        "would apply the opposite direction"
    )


def test_anti_sycophancy_is_negatively_signed(spec_json):
    """Pinned to the measured contrast effect for feature 204 (−0.1003)."""
    assert spec_json["features"][0]["feature_id"] == 204
    assert spec_json["features"][0]["strength"] < 0
    assert spec_json["metadata"]["direction_sign"] == -1


def test_sign_reversal_is_documented(spec_json):
    """The near-miss is recorded in the artifact, not just in a commit message."""
    note = spec_json["metadata"]["sign_note"].lower()
    assert "sign" in note
    assert "0.150" in note or "sign control" in note


def test_compiled_artifact_matches_the_spec_sign():
    """The shipped artifact must not have drifted from the spec."""
    artifact = ROOT / "examples" / "patches" / "anti-sycophancy.brainpatch"
    if not artifact.is_file():
        pytest.skip("compiled artifact not present in this checkout")
    from brainpatch.patch.loader import load_patch

    manifest = load_patch(artifact).manifest
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    coefficient = manifest.interventions[0].coefficient
    assert (coefficient < 0) == (spec["features"][0]["strength"] < 0)
    assert coefficient == pytest.approx(spec["features"][0]["strength"], rel=1e-6)


def test_artifact_records_the_site_it_was_validated_for():
    artifact = ROOT / "examples" / "patches" / "anti-sycophancy.brainpatch"
    if not artifact.is_file():
        pytest.skip("compiled artifact not present in this checkout")
    from brainpatch.patch.loader import load_patch

    manifest = load_patch(artifact).manifest
    assert manifest.interventions[0].site == "prompt"
    assert manifest.provenance.get("injection_site") == "prompt"


def test_backends_that_cannot_express_the_site_are_not_claimed():
    artifact = ROOT / "examples" / "patches" / "anti-sycophancy.brainpatch"
    if not artifact.is_file():
        pytest.skip("compiled artifact not present in this checkout")
    from brainpatch.patch.loader import load_patch

    compatibility = load_patch(artifact).manifest.compatibility
    assert compatibility["llamacpp"]["status"] == "unsupported"
    assert compatibility["vllm"]["status"] == "unsupported"
