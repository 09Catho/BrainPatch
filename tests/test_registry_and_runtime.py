"""Local registry, backend contract, and edit resolution.

The backend tests use a stub engine rather than a real model: the shared
bookkeeping in :class:`~brainpatch.runtime.base.BrainPatchBackend` is where the
subtle logic lives (strength clamping, schedule resolution, the guarantee that
zero means *no edit at all*), and it is identical for every engine. Testing it
here means it is covered on a laptop with no ML stack, instead of only inside a
paid GPU job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brainpatch.patch import tensors as ts
from brainpatch.patch.format import BaseModelSpec, Intervention, Manifest
from brainpatch.patch.loader import load_patch, save_patch
from brainpatch.patch.registry import PatchRegistry, RegistryError
from brainpatch.patch.validation import ModelDescriptor
from brainpatch.runtime.base import BrainPatchBackend, GenerationConfig
from brainpatch.runtime.capabilities import CAPABILITY_FLAGS, Capabilities
from brainpatch.runtime.scheduling import StrengthSchedule

HIDDEN = 8


def build_patch(tmp_path: Path, name: str = "test-patch", **manifest_kwargs) -> Path:
    manifest = Manifest(
        name=name,
        base_model=BaseModelSpec(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            architecture="Qwen2ForCausalLM",
            hidden_size=HIDDEN,
            num_layers=28,
        ),
        interventions=manifest_kwargs.pop(
            "interventions", [Intervention(layer=18, vector="v0", coefficient=2.0)]
        ),
        **manifest_kwargs,
    )
    vectors = {"v0": ts.vector([1.0] * HIDDEN, dtype="F16")}
    return save_patch(manifest, vectors, tmp_path / f"{name}.brainpatch", overwrite=True)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path) -> PatchRegistry:
    return PatchRegistry(tmp_path / "home")


def test_install_and_list(registry, tmp_path):
    path = build_patch(tmp_path)
    installed = registry.install_file(path)
    assert installed.name == "test-patch"
    assert [p.name for p in registry.list_patches()] == ["test-patch"]
    assert registry.is_installed("test-patch")


def test_install_records_provenance(registry, tmp_path):
    installed = registry.install_file(build_patch(tmp_path))
    assert installed.source["kind"] == "file"
    assert installed.source["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert "installed_at" in installed.source


def test_install_refuses_to_overwrite(registry, tmp_path):
    path = build_patch(tmp_path)
    registry.install_file(path)
    with pytest.raises(RegistryError, match="already installed"):
        registry.install_file(path)
    registry.install_file(path, overwrite=True)


def test_corrupt_patch_never_lands_in_registry(registry, tmp_path):
    """Validation happens before anything is written."""
    bad = tmp_path / "bad.brainpatch"
    bad.write_bytes(b"not a zip")
    with pytest.raises(Exception):
        registry.install_file(bad)
    assert registry.list_patches() == []


def test_uninstall(registry, tmp_path):
    registry.install_file(build_patch(tmp_path))
    registry.uninstall("test-patch")
    assert registry.list_patches() == []
    with pytest.raises(RegistryError, match="not installed"):
        registry.uninstall("test-patch")


def test_get_unknown_lists_available(registry, tmp_path):
    registry.install_file(build_patch(tmp_path, name="alpha"))
    with pytest.raises(RegistryError, match="alpha"):
        registry.get("beta")


def test_resolve_prefers_installed_name_over_cwd_file(registry, tmp_path, monkeypatch):
    """A bare name must not accidentally read a same-named file from the CWD."""
    registry.install_file(build_patch(tmp_path, name="test-patch"))
    monkeypatch.chdir(tmp_path)
    Path("test-patch").write_text("decoy")
    assert registry.resolve("test-patch") == registry.path_for("test-patch")


def test_resolve_falls_back_to_path(registry, tmp_path):
    path = build_patch(tmp_path)
    assert registry.resolve(str(path)) == path


def test_resolve_unknown_raises(registry):
    with pytest.raises(RegistryError, match="neither an installed patch"):
        registry.resolve("nope")


def test_offline_install_from_hub_refused(registry):
    with pytest.raises(RegistryError, match="offline"):
        registry.install_from_hub("owner/repo", offline=True)


@pytest.mark.parametrize("ref", ["not a ref", "owner", "/abs/path"])
def test_invalid_hub_refs_rejected(registry, ref):
    with pytest.raises(RegistryError):
        registry.install(ref)


def test_registry_home_env_override(tmp_path, monkeypatch):
    from brainpatch.patch.registry import ENV_HOME, registry_home

    monkeypatch.setenv(ENV_HOME, str(tmp_path / "custom"))
    assert registry_home() == tmp_path / "custom"


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


def test_capabilities_default_to_false():
    """A backend must opt in to every claim."""
    caps = Capabilities(name="stub")
    for flag in CAPABILITY_FLAGS:
        assert caps.supports(flag) is False, flag


def test_require_raises_with_note():
    caps = Capabilities(name="stub", notes={"streaming": "not implemented here."})
    with pytest.raises(NotImplementedError, match="not implemented here"):
        caps.require("streaming")


def test_unknown_capability_rejected():
    with pytest.raises(ValueError, match="unknown capability"):
        Capabilities(name="stub").supports("telepathy")


# ---------------------------------------------------------------------------
# backend contract
# ---------------------------------------------------------------------------


class StubBackend(BrainPatchBackend):
    """A backend with no engine, to exercise the shared bookkeeping."""

    name = "stub"

    def __init__(self, *, multiple: bool = True, schedules: bool = True) -> None:
        super().__init__()
        self._multiple = multiple
        self._schedules = schedules
        self.generated: list[str] = []

    @classmethod
    def is_available(cls):
        return True, "stub"

    @classmethod
    def capabilities(cls):
        return Capabilities(
            name=cls.name, static_intervention=True, dynamic_schedule=True, multiple_patches=True
        )

    def load_model(self, model: str, **kwargs):
        self._model = model

    def describe_model(self):
        return ModelDescriptor(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            hidden_size=HIDDEN,
            num_layers=28,
            architecture="Qwen2ForCausalLM",
        )

    def generate(self, prompt, config=None, **kwargs):
        self.generated.append(prompt)
        return f"<{len(self.resolve_edits(0))} edits>"


@pytest.fixture
def backend(tmp_path):
    b = StubBackend()
    b.load_model("Qwen/Qwen2.5-1.5B-Instruct")
    return b


def test_install_and_resolve_edits(backend, tmp_path):
    backend.install_patch(load_patch(build_patch(tmp_path)))
    edits = backend.resolve_edits(0)
    assert len(edits) == 1
    # default_strength 1.0 * intervention coefficient 2.0
    assert edits[0].coefficient == pytest.approx(2.0)
    assert edits[0].layer == 18


def test_strength_multiplies_coefficient(backend, tmp_path):
    backend.install_patch(load_patch(build_patch(tmp_path)))
    backend.set_strength("test-patch", 1.5)
    assert backend.resolve_edits(0)[0].coefficient == pytest.approx(3.0)


def test_zero_strength_yields_no_edits(backend, tmp_path):
    """The property that makes strength 0 identical to baseline."""
    backend.install_patch(load_patch(build_patch(tmp_path)))
    backend.set_strength("test-patch", 0.0)
    assert backend.resolve_edits(0) == []


def test_disabled_patch_yields_no_edits(backend, tmp_path):
    backend.install_patch(load_patch(build_patch(tmp_path)))
    backend.set_enabled("test-patch", False)
    assert backend.resolve_edits(0) == []
    backend.set_enabled("test-patch", True)
    assert backend.resolve_edits(0) != []


def test_strength_is_clamped_to_manifest_envelope(backend, tmp_path):
    path = build_patch(tmp_path, max_abs_strength=2.0)
    backend.install_patch(load_patch(path))
    with pytest.warns(UserWarning, match="clamped"):
        actual = backend.set_strength("test-patch", 100.0)
    assert actual == 2.0


def test_layer_filter(backend, tmp_path):
    path = build_patch(
        tmp_path,
        interventions=[
            Intervention(layer=4, vector="v0"),
            Intervention(layer=18, vector="v0"),
        ],
    )
    backend.install_patch(load_patch(path))
    assert len(backend.resolve_edits(0, layer=4)) == 1
    assert len(backend.resolve_edits(0)) == 2
    assert backend.active_layers() == [4, 18]


def test_schedule_gates_by_token_index(backend, tmp_path):
    backend.install_patch(load_patch(build_patch(tmp_path)))
    backend.set_schedule("test-patch", StrengthSchedule({0: 0.0, 24: 1.0}))
    assert backend.resolve_edits(0) == []
    assert backend.resolve_edits(23) == []
    assert backend.resolve_edits(24) != []


def test_schedule_from_manifest_is_picked_up(backend, tmp_path):
    path = build_patch(tmp_path, schedule={"0": 0.0, "10": 1.0})
    backend.install_patch(load_patch(path))
    assert backend.resolve_edits(5) == []
    assert backend.resolve_edits(10) != []


def test_incompatible_patch_is_refused(backend, tmp_path):
    from brainpatch.patch.validation import PatchCompatibilityError

    manifest = Manifest(
        name="wrong-model",
        base_model=BaseModelSpec(
            model_id="google/gemma-2-2b", hidden_size=HIDDEN, num_layers=28
        ),
        interventions=[Intervention(layer=1, vector="v0")],
    )
    path = save_patch(
        manifest, {"v0": ts.vector([1.0] * HIDDEN)}, tmp_path / "wrong.brainpatch"
    )
    with pytest.raises(PatchCompatibilityError, match="not.*transferable|model mismatch"):
        backend.install_patch(load_patch(path))
    assert backend.list_patches() == []


def test_unverified_backend_warns_on_install(backend, tmp_path):
    path = build_patch(tmp_path, compatibility={"stub": {"status": "implemented"}})
    with pytest.warns(UserWarning, match="not been verified"):
        backend.install_patch(load_patch(path))


def test_remove_patch(backend, tmp_path):
    backend.install_patch(load_patch(build_patch(tmp_path)))
    backend.remove_patch("test-patch")
    assert backend.list_patches() == []
    with pytest.raises(KeyError):
        backend.remove_patch("test-patch")


def test_operations_on_unknown_patch_raise(backend):
    with pytest.raises(KeyError, match="no patch named"):
        backend.set_strength("ghost", 1.0)


def test_multiple_patches_accumulate(backend, tmp_path):
    backend.install_patch(load_patch(build_patch(tmp_path, name="a")))
    backend.install_patch(load_patch(build_patch(tmp_path, name="b")))
    assert len(backend.resolve_edits(0)) == 2


def test_vector_values_round_trip(backend, tmp_path):
    backend.install_patch(load_patch(build_patch(tmp_path)))
    assert backend.vector_values("test-patch", "v0") == [1.0] * HIDDEN


# ---------------------------------------------------------------------------
# backend discovery
# ---------------------------------------------------------------------------


def test_normalize_backend_aliases():
    from brainpatch.runtime.auto import normalize_backend_name

    assert normalize_backend_name("llama.cpp") == "llamacpp"
    assert normalize_backend_name("HF") == "transformers"
    assert normalize_backend_name("mlx-lm") == "mlx"


def test_unknown_backend_raises():
    from brainpatch.runtime.auto import BackendNotAvailable, backend_class

    with pytest.raises(BackendNotAvailable, match="unknown backend"):
        backend_class("tensorflow")


def test_probe_never_raises():
    """doctor must work on a machine with nothing installed."""
    from brainpatch.runtime.auto import BACKEND_MODULES, probe_backend

    for name in BACKEND_MODULES:
        status = probe_backend(name)
        assert isinstance(status.available, bool)
        assert status.detail


def test_environment_report_is_json_serialisable():
    from brainpatch.runtime.auto import environment_report

    json.dumps(environment_report())
