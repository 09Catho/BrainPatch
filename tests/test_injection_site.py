"""Injection-site support in the v1 format and runtime.

Added because `anti_sycophancy_v3`'s validated configuration could not be
expressed otherwise: the effect was measured with **prompt-token-only**
injection, and a schedule cannot encode that (the prompt pass and generated
token 0 share an index). Shipping the patch without this field would have meant
shipping a configuration that was never tested.
"""

from __future__ import annotations

import pytest

from brainpatch.patch.format import (
    KNOWN_INJECTION_SITES,
    BaseModelSpec,
    Intervention,
    Manifest,
    PatchFormatError,
)
from brainpatch.runtime.base import BrainPatchBackend

pytestmark = pytest.mark.local


def make_manifest(site: str = "all") -> Manifest:
    return Manifest(
        name="site-test",
        base_model=BaseModelSpec(
            model_id="Qwen/Qwen2.5-1.5B-Instruct", hidden_size=1536, num_layers=28
        ),
        interventions=[Intervention(layer=18, vector="v0", coefficient=1.0, site=site)],
    )


# --- schema ----------------------------------------------------------------


def test_default_site_is_all():
    assert Intervention(layer=1, vector="v").site == "all"


@pytest.mark.parametrize("site", KNOWN_INJECTION_SITES)
def test_every_known_site_validates(site):
    make_manifest(site).validate()


def test_unknown_site_is_rejected():
    with pytest.raises(PatchFormatError, match="injection site"):
        make_manifest("everywhere").validate()


def test_default_site_is_omitted_from_serialisation():
    """Patches written before this field keep a byte-identical manifest."""
    data = Intervention(layer=1, vector="v").to_dict()
    assert "site" not in data


def test_non_default_site_round_trips():
    manifest = make_manifest("prompt")
    restored = Manifest.from_dict(manifest.to_dict())
    restored.validate()
    assert restored.interventions[0].site == "prompt"


def test_missing_site_reads_as_all():
    """Backward compatibility with every previously written patch."""
    restored = Intervention.from_dict({"layer": 3, "vector": "v0", "coefficient": 1.0})
    assert restored.site == "all"


# --- runtime resolution ----------------------------------------------------


class _Backend(BrainPatchBackend):
    """Minimal concrete backend; only edit resolution is under test."""

    name = "stub"

    def load_model(self, *a, **k):  # pragma: no cover - not exercised
        raise NotImplementedError

    def describe_model(self):  # pragma: no cover
        raise NotImplementedError

    def generate(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def capabilities(self):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def is_available(cls) -> bool:  # pragma: no cover
        return True


def install(site: str) -> _Backend:
    from brainpatch.patch.loader import LoadedPatch
    from brainpatch.runtime.base import ActivePatch

    backend = _Backend()
    loaded = LoadedPatch(manifest=make_manifest(site), vectors={})
    backend._patches["site-test"] = ActivePatch(patch=loaded, strength=1.0)
    return backend


@pytest.mark.parametrize(
    "site,prompt_pass,expected",
    [
        ("all", True, 1),
        ("all", False, 1),
        ("prompt", True, 1),
        ("prompt", False, 0),
        ("continuation", True, 0),
        ("continuation", False, 1),
    ],
)
def test_site_gates_edit_resolution(site, prompt_pass, expected):
    backend = install(site)
    edits = backend.resolve_edits(0, layer=18, is_prompt_pass=prompt_pass)
    assert len(edits) == expected


def test_unknown_pass_kind_applies_everything():
    """Backends that cannot tell prompt from continuation must still work.

    llama.cpp binds a control vector for a whole run and vLLM shares one forward
    pass across a batch, so neither can honour a site restriction. They pass
    None and get the unrestricted behaviour, which the compatibility block in
    the patch records as a capability gap rather than silently pretending.
    """
    for site in KNOWN_INJECTION_SITES:
        backend = install(site)
        assert len(backend.resolve_edits(0, layer=18, is_prompt_pass=None)) == 1


def test_site_restriction_still_respects_zero_strength():
    backend = install("prompt")
    backend._patches["site-test"].strength = 0.0
    assert backend.resolve_edits(0, layer=18, is_prompt_pass=True) == []
