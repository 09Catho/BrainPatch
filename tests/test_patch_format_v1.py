"""The portable ``.brainpatch`` v1 format: round trip, safety, validation.

A patch is untrusted input downloaded from the internet, so most of this file is
about what the loader *refuses*. The format's entire security argument is that a
patch is inert data -- these tests are what keep that true.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from brainpatch.patch import tensors as ts
from brainpatch.patch.format import (
    ABSOLUTE_MAX_STRENGTH,
    CHECKSUMS_NAME,
    MANIFEST_NAME,
    VECTORS_NAME,
    BaseModelSpec,
    Intervention,
    Manifest,
    PatchFormatError,
)
from brainpatch.patch.loader import PatchLoadError, load_patch, patch_size_report, save_patch
from brainpatch.patch.validation import ModelDescriptor, check_compatibility

HIDDEN = 8


def make_manifest(**overrides) -> Manifest:
    base = dict(
        name="test-patch",
        base_model=BaseModelSpec(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            architecture="Qwen2ForCausalLM",
            hidden_size=HIDDEN,
            num_layers=28,
            revision="989aa798",
        ),
        interventions=[Intervention(layer=18, vector="v0", coefficient=1.5)],
    )
    base.update(overrides)
    return Manifest(**base)  # type: ignore[arg-type]


def make_vectors(n: int = 1) -> dict[str, ts.Tensor]:
    return {
        f"v{i}": ts.vector([float(i + j) / 8 for j in range(HIDDEN)], dtype="F16")
        for i in range(n)
    }


# ---------------------------------------------------------------------------
# pure-python safetensors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["F32", "F64", "F16", "BF16"])
def test_tensor_round_trip(dtype):
    original = ts.vector([1.0, -0.5, 0.25, 0.0], dtype=dtype)
    restored, _ = ts.load(ts.dump({"x": original}))
    assert restored["x"].data == original.data
    assert restored["x"].dtype == dtype


def test_dump_is_deterministic():
    """Byte-determinism is what makes a published patch hash verifiable."""
    vectors = make_vectors(3)
    assert ts.dump(vectors) == ts.dump(dict(reversed(list(vectors.items()))))


def test_metadata_round_trip():
    _, meta = ts.load(ts.dump({"x": ts.vector([1.0])}, metadata={"a": "b"}))
    assert meta == {"a": "b"}


@pytest.mark.parametrize(
    "blob,match",
    [
        (b"", "too short"),
        (b"\x00" * 8 + b"", "not valid UTF-8 JSON"),
        (b"\xff" * 8, "refusing to read"),
    ],
)
def test_malformed_tensor_blobs_rejected(blob, match):
    with pytest.raises(ts.SafetensorsError, match=match):
        ts.load(blob)


def test_offsets_outside_payload_rejected():
    header = json.dumps({"x": {"dtype": "F32", "shape": [4], "data_offsets": [0, 9999]}})
    blob = len(header).to_bytes(8, "little") + header.encode() + b"\x00" * 16
    with pytest.raises(ts.SafetensorsError, match="outside"):
        ts.load(blob)


def test_unsupported_dtype_rejected():
    header = json.dumps({"x": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}})
    blob = len(header).to_bytes(8, "little") + header.encode() + b"\x00" * 8
    with pytest.raises(ts.SafetensorsError, match="unsupported dtype"):
        ts.load(blob)


# ---------------------------------------------------------------------------
# manifest validation
# ---------------------------------------------------------------------------


def test_valid_manifest_round_trips():
    manifest = make_manifest()
    manifest.validate()
    assert Manifest.from_json(manifest.to_json()).to_dict() == manifest.to_dict()


def test_layers_and_vector_keys():
    manifest = make_manifest(
        interventions=[
            Intervention(layer=18, vector="v0"),
            Intervention(layer=4, vector="v1"),
            Intervention(layer=18, vector="v0"),
        ]
    )
    assert manifest.layers == [4, 18]
    assert manifest.vector_keys == ["v0", "v1"]


@pytest.mark.parametrize("name", ["Has Space", "UPPER", "", "x" * 65, "-lead"])
def test_invalid_names_rejected(name):
    with pytest.raises(PatchFormatError):
        make_manifest(name=name).validate()


def test_empty_interventions_rejected():
    with pytest.raises(PatchFormatError, match="at least one intervention"):
        make_manifest(interventions=[]).validate()


def test_layer_beyond_declared_depth_rejected():
    with pytest.raises(PatchFormatError, match="declares"):
        make_manifest(interventions=[Intervention(layer=99, vector="v0")]).validate()


def test_unknown_hook_rejected():
    with pytest.raises(PatchFormatError, match="unsupported hook"):
        make_manifest(interventions=[Intervention(layer=1, vector="v0", hook="attn")]).validate()


def test_duplicate_intervention_ids_rejected():
    with pytest.raises(PatchFormatError, match="duplicate intervention id"):
        make_manifest(
            interventions=[
                Intervention(layer=1, vector="v0", id="a"),
                Intervention(layer=2, vector="v0", id="a"),
            ]
        ).validate()


def test_unsupported_format_version_rejected():
    with pytest.raises(PatchFormatError, match="format_version"):
        make_manifest(format_version="99.0").validate()


def test_absurd_coefficient_rejected():
    with pytest.raises(PatchFormatError, match="absolute ceiling"):
        make_manifest(
            interventions=[Intervention(layer=1, vector="v0", coefficient=1e9)]
        ).validate()


def test_max_abs_strength_bounds():
    with pytest.raises(PatchFormatError, match="max_abs_strength"):
        make_manifest(max_abs_strength=0).validate()
    with pytest.raises(PatchFormatError, match="max_abs_strength"):
        make_manifest(max_abs_strength=ABSOLUTE_MAX_STRENGTH * 2).validate()


def test_default_strength_within_envelope():
    with pytest.raises(PatchFormatError, match="default_strength"):
        make_manifest(max_abs_strength=1.0, default_strength=5.0).validate()


def test_clamp_strength():
    manifest = make_manifest(max_abs_strength=2.0)
    assert manifest.clamp_strength(10.0) == 2.0
    assert manifest.clamp_strength(-10.0) == -2.0
    assert manifest.clamp_strength(1.5) == 1.5


def test_unknown_compatibility_status_rejected():
    with pytest.raises(PatchFormatError, match="status"):
        make_manifest(compatibility={"vllm": {"status": "supported"}}).validate()


def test_backend_status_defaults_to_unsupported():
    manifest = make_manifest(compatibility={"transformers": {"status": "verified"}})
    manifest.validate()
    assert manifest.is_verified_on("transformers")
    assert manifest.backend_status("vllm") == "unsupported"
    assert not manifest.is_verified_on("vllm")


@pytest.mark.parametrize("schedule", [{}, {"a": 1.0}, {"-1": 1.0}, {"0": "x"}])
def test_invalid_schedules_rejected(schedule):
    with pytest.raises(PatchFormatError):
        make_manifest(schedule=schedule).validate()


def test_unknown_evidence_level_rejected():
    with pytest.raises(PatchFormatError, match="evidence_level"):
        make_manifest(evidence_level="proven").validate()


def test_causal_is_not_an_evidence_level():
    """The old over-strong rung must stay rejected."""
    with pytest.raises(PatchFormatError, match="evidence_level"):
        make_manifest(evidence_level="causal").validate()


# ---------------------------------------------------------------------------
# archive round trip
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    manifest = make_manifest()
    vectors = make_vectors(1)
    path = save_patch(manifest, vectors, tmp_path / "p.brainpatch")

    loaded = load_patch(path)
    assert loaded.manifest.to_dict() == manifest.to_dict()
    assert loaded.vectors["v0"].data == vectors["v0"].data
    assert loaded.archive_bytes > 0


def test_archive_is_deterministic(tmp_path):
    manifest, vectors = make_manifest(), make_vectors(2)
    a = save_patch(manifest, vectors, tmp_path / "a.brainpatch").read_bytes()
    b = save_patch(manifest, vectors, tmp_path / "b.brainpatch").read_bytes()
    assert a == b


def test_suffix_is_applied(tmp_path):
    path = save_patch(make_manifest(), make_vectors(), tmp_path / "p")
    assert path.suffix == ".brainpatch"


def test_save_refuses_to_clobber(tmp_path):
    save_patch(make_manifest(), make_vectors(), tmp_path / "p.brainpatch")
    with pytest.raises(FileExistsError):
        save_patch(make_manifest(), make_vectors(), tmp_path / "p.brainpatch")
    save_patch(make_manifest(), make_vectors(), tmp_path / "p.brainpatch", overwrite=True)


def test_missing_referenced_vector_rejected(tmp_path):
    with pytest.raises(PatchFormatError, match="references vectors not provided"):
        save_patch(make_manifest(), {}, tmp_path / "p.brainpatch")


def test_wrong_vector_length_rejected(tmp_path):
    bad = {"v0": ts.vector([1.0, 2.0], dtype="F16")}
    with pytest.raises(PatchFormatError, match="must be 1-D of length"):
        save_patch(make_manifest(), bad, tmp_path / "p.brainpatch")


def test_size_report(tmp_path):
    path = save_patch(make_manifest(), make_vectors(1), tmp_path / "p.brainpatch")
    report = patch_size_report(load_patch(path))
    assert report["num_vectors"] == 1
    assert report["hidden_size"] == HIDDEN
    assert report["dtype"] == "F16"
    assert report["vector_payload_bytes"] == HIDDEN * 2


# ---------------------------------------------------------------------------
# loader safety -- a patch is untrusted input
# ---------------------------------------------------------------------------


def test_missing_file(tmp_path):
    with pytest.raises(PatchLoadError, match="not found"):
        load_patch(tmp_path / "nope.brainpatch")


def test_not_a_zip(tmp_path):
    path = tmp_path / "p.brainpatch"
    path.write_bytes(b"definitely not a zip")
    with pytest.raises(PatchLoadError, match="not a valid archive"):
        load_patch(path)


def test_missing_manifest_rejected(tmp_path):
    path = tmp_path / "p.brainpatch"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(VECTORS_NAME, ts.dump(make_vectors()))
        zf.writestr(CHECKSUMS_NAME, "{}")
    with pytest.raises(PatchLoadError, match="missing required member"):
        load_patch(path)


def test_unexpected_member_rejected(tmp_path):
    """An archive may contain only the four known members."""
    good = save_patch(make_manifest(), make_vectors(), tmp_path / "good.brainpatch")
    sneaky = tmp_path / "sneaky.brainpatch"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(sneaky, "w") as dst:
        for name in src.namelist():
            dst.writestr(name, src.read(name))
        dst.writestr("evil.py", "import os; os.system('echo pwned')")
    with pytest.raises(PatchLoadError, match="unexpected members"):
        load_patch(sneaky)


@pytest.mark.parametrize(
    "evil_name",
    ["../escape.json", "/absolute.json", "a/../../escape.json"],
)
def test_path_traversal_rejected(tmp_path, evil_name):
    path = tmp_path / "p.brainpatch"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(evil_name, "{}")
    with pytest.raises(PatchLoadError, match="absolute path|escapes the archive"):
        load_patch(path)


def test_checksum_mismatch_rejected(tmp_path):
    """Tampering with a member after signing must be caught."""
    good = save_patch(make_manifest(), make_vectors(), tmp_path / "good.brainpatch")
    tampered = tmp_path / "tampered.brainpatch"
    with zipfile.ZipFile(good) as src:
        members = {n: src.read(n) for n in src.namelist()}
    manifest = json.loads(members[MANIFEST_NAME])
    manifest["max_abs_strength"] = 999.0  # widen the safety envelope
    members[MANIFEST_NAME] = json.dumps(manifest).encode()
    with zipfile.ZipFile(tampered, "w") as dst:
        for name, blob in members.items():
            dst.writestr(name, blob)
    with pytest.raises(PatchLoadError, match="checksum mismatch"):
        load_patch(tampered)


def test_missing_checksums_rejected(tmp_path):
    good = save_patch(make_manifest(), make_vectors(), tmp_path / "good.brainpatch")
    stripped = tmp_path / "stripped.brainpatch"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(stripped, "w") as dst:
        for name in src.namelist():
            if name != CHECKSUMS_NAME:
                dst.writestr(name, src.read(name))
    with pytest.raises(PatchLoadError, match=CHECKSUMS_NAME):
        load_patch(stripped)


def test_vector_length_mismatch_detected_on_load(tmp_path):
    """A manifest declaring a different hidden size than its vectors is caught."""
    manifest = make_manifest()
    vectors = make_vectors()
    path = save_patch(manifest, vectors, tmp_path / "p.brainpatch")

    lying = tmp_path / "lying.brainpatch"
    with zipfile.ZipFile(path) as src:
        members = {n: src.read(n) for n in src.namelist()}
    data = json.loads(members[MANIFEST_NAME])
    data["base_model"]["hidden_size"] = HIDDEN + 1
    members[MANIFEST_NAME] = json.dumps(data).encode()
    members[CHECKSUMS_NAME] = json.dumps(
        {n: hashlib.sha256(b).hexdigest() for n, b in members.items() if n != CHECKSUMS_NAME}
    ).encode()
    with zipfile.ZipFile(lying, "w") as dst:
        for name, blob in members.items():
            dst.writestr(name, blob)
    with pytest.raises(PatchLoadError, match="hidden_size|length"):
        load_patch(lying)


# ---------------------------------------------------------------------------
# compatibility modes
# ---------------------------------------------------------------------------


def descriptor(**overrides) -> ModelDescriptor:
    base = dict(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        hidden_size=HIDDEN,
        num_layers=28,
        architecture="Qwen2ForCausalLM",
        revision="989aa798",
    )
    base.update(overrides)
    return ModelDescriptor(**base)  # type: ignore[arg-type]


def test_strict_accepts_exact_match():
    assert check_compatibility(make_manifest(), descriptor(), mode="strict").ok


def test_strict_rejects_different_model():
    report = check_compatibility(make_manifest(), descriptor(model_id="google/gemma-2-2b"))
    assert not report.ok
    assert any("model mismatch" in e for e in report.errors)


def test_strict_rejects_revision_mismatch():
    report = check_compatibility(make_manifest(), descriptor(revision="deadbeef"))
    assert not report.ok
    assert any("revision" in e for e in report.errors)


def test_hidden_size_mismatch_rejected_in_every_mode():
    for mode in ("strict", "architecture", "unsafe"):
        report = check_compatibility(make_manifest(), descriptor(hidden_size=999), mode=mode)
        assert not report.ok, mode
        assert any("hidden size" in e for e in report.errors)


def test_layer_beyond_model_depth_rejected_in_every_mode():
    for mode in ("strict", "architecture", "unsafe"):
        report = check_compatibility(make_manifest(), descriptor(num_layers=4), mode=mode)
        assert not report.ok, mode


def test_architecture_mode_allows_different_model_id_with_warning():
    report = check_compatibility(
        make_manifest(), descriptor(model_id="someone/qwen-finetune"), mode="architecture"
    )
    assert report.ok
    assert report.warnings


def test_architecture_mode_still_rejects_different_architecture():
    report = check_compatibility(
        make_manifest(),
        descriptor(model_id="x/y", architecture="LlamaForCausalLM"),
        mode="architecture",
    )
    assert not report.ok


def test_unsafe_mode_allows_wrong_model_but_warns_loudly():
    report = check_compatibility(
        make_manifest(),
        descriptor(model_id="google/gemma-2-2b", architecture="GemmaForCausalLM"),
        mode="unsafe",
    )
    assert report.ok
    assert any("UNSAFE MODE" in w for w in report.warnings)


def test_report_raises_with_actionable_message():
    from brainpatch.patch.validation import PatchCompatibilityError

    report = check_compatibility(make_manifest(), descriptor(model_id="other/model"))
    with pytest.raises(PatchCompatibilityError, match="compatibility_mode"):
        report.raise_if_failed()


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown compatibility mode"):
        check_compatibility(make_manifest(), descriptor(), mode="yolo")  # type: ignore[arg-type]
