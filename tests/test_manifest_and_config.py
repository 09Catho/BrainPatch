"""Manifests, paths, SAE config, and YAML configuration handling."""

from __future__ import annotations

import json

import pytest

from brainpatch.config import (
    ConfigError,
    apply_overrides,
    deep_merge,
    get,
    load_config,
    parse_override,
    require,
)
from brainpatch.paths import VolumePaths, parse_shard_index, shard_filename
from brainpatch.schemas.manifest import ActivationManifest, ShardRecord
from brainpatch.schemas.sae import SAEConfig


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def test_shard_filenames_sort_lexically_in_numeric_order():
    names = [shard_filename(i) for i in (0, 1, 9, 10, 99, 100)]
    assert names == sorted(names)


def test_shard_filename_round_trip():
    for i in (0, 7, 12345):
        assert parse_shard_index(shard_filename(i)) == i


def test_negative_shard_index_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        shard_filename(-1)


def test_parse_shard_index_rejects_other_files():
    with pytest.raises(ValueError, match="not a shard"):
        parse_shard_index("manifest.json")


def test_volume_paths_are_posix_under_root():
    paths = VolumePaths("/vol")
    assert str(paths.activation_manifest("smoke_v0")) == "/vol/activations/smoke_v0/manifest.json"
    assert str(paths.sae_checkpoint("smoke_v0")) == "/vol/sae/smoke_v0/sae_latest.pt"
    assert str(paths.features_jsonl("smoke_v0")) == "/vol/feature-db/smoke_v0/features.jsonl"


def test_all_top_level_covers_documented_layout():
    names = {p.name for p in VolumePaths().all_top_level()}
    assert names == {
        "hf-cache",
        "datasets",
        "activations",
        "sae",
        "feature-db",
        "patches",
        "experiments",
        "reports",
    }


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def make_manifest(**overrides) -> ActivationManifest:
    base = dict(
        experiment="smoke_v0",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        model_revision="989aa798",
        layer=18,
        hook="residual_post",
        hidden_size=1536,
        dtype="bfloat16",
        dataset="Salesforce/wikitext:wikitext-2-raw-v1",
        dataset_split="train",
        sequence_length=256,
        requested_tokens=20_000,
    )
    base.update(overrides)
    return ActivationManifest(**base)  # type: ignore[arg-type]


def test_empty_manifest_is_valid_and_incomplete():
    manifest = make_manifest()
    manifest.validate()
    assert not manifest.is_complete
    assert manifest.next_shard_index == 0


def test_completed_tokens_must_match_shards():
    manifest = make_manifest(completed_tokens=100)
    with pytest.raises(ValueError, match="inconsistent"):
        manifest.validate()


def test_shard_indices_must_be_contiguous():
    manifest = make_manifest()
    manifest.shards = [
        ShardRecord(index=0, filename=shard_filename(0), num_tokens=10, first_example=0, last_example=1),
        ShardRecord(index=2, filename=shard_filename(2), num_tokens=10, first_example=2, last_example=3),
    ]
    manifest.completed_tokens = 20
    with pytest.raises(ValueError, match="contiguous"):
        manifest.validate()


def test_bytes_per_token_is_measured_not_assumed():
    manifest = make_manifest()
    manifest.shards = [
        ShardRecord(
            index=0, filename=shard_filename(0), num_tokens=1000,
            first_example=0, last_example=9, bytes=3_084_000,
        )
    ]
    manifest.completed_tokens = 1000
    manifest.validate()
    assert manifest.bytes_per_token == pytest.approx(3084.0)


def test_bytes_per_token_is_none_without_sizes():
    manifest = make_manifest()
    manifest.shards = [
        ShardRecord(index=0, filename=shard_filename(0), num_tokens=10, first_example=0, last_example=1)
    ]
    manifest.completed_tokens = 10
    assert manifest.bytes_per_token is None


def test_manifest_round_trip():
    manifest = make_manifest(completed_tokens=20, num_examples=3)
    manifest.shards = [
        ShardRecord(
            index=0, filename=shard_filename(0), num_tokens=20,
            first_example=0, last_example=2, bytes=61_680,
        )
    ]
    manifest.validate()
    restored = ActivationManifest.from_json(manifest.to_json())
    assert restored.to_dict() == manifest.to_dict()


def test_is_complete_at_target():
    manifest = make_manifest(requested_tokens=100)
    manifest.shards = [
        ShardRecord(index=0, filename=shard_filename(0), num_tokens=100, first_example=0, last_example=1)
    ]
    manifest.completed_tokens = 100
    assert manifest.is_complete


@pytest.mark.parametrize(
    "field,value", [("hidden_size", 0), ("layer", -1), ("sequence_length", 0), ("shard_size", 0)]
)
def test_invalid_manifest_fields_rejected(field, value):
    manifest = make_manifest(**{field: value})
    with pytest.raises(ValueError):
        manifest.validate()


# ---------------------------------------------------------------------------
# SAE config
# ---------------------------------------------------------------------------


def test_sae_config_derived_properties():
    config = SAEConfig(d_in=1536, d_sae=2048, k=32)
    config.validate()
    assert config.expansion_factor == pytest.approx(2048 / 1536)
    # encoder + decoder + both biases
    assert config.num_parameters == 2 * 1536 * 2048 + 2048 + 1536


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_in": 0, "d_sae": 8},
        {"d_in": 8, "d_sae": 0},
        {"d_in": 8, "d_sae": 8, "k": 0},
        {"d_in": 8, "d_sae": 8, "k": 9},
        {"d_in": 8, "d_sae": 8, "auxk_alpha": -1.0},
        {"d_in": 8, "d_sae": 8, "auxk_k": 0},
        {"d_in": 8, "d_sae": 8, "val_fraction": 1.0},
        {"d_in": 8, "d_sae": 8, "batch_size": 0},
    ],
)
def test_invalid_sae_configs_rejected(kwargs):
    with pytest.raises(ValueError):
        SAEConfig(**kwargs).validate()  # type: ignore[arg-type]


def test_sae_config_round_trip():
    config = SAEConfig(d_in=1536, d_sae=2048, k=32, input_scale=0.5610531069008018)
    assert SAEConfig.from_json(config.to_json()).to_dict() == config.to_dict()


# ---------------------------------------------------------------------------
# configuration loading
# ---------------------------------------------------------------------------


def test_deep_merge_merges_nested_mappings():
    assert deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}}


def test_deep_merge_replaces_non_mappings():
    assert deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}


@pytest.mark.parametrize(
    "text,expected_path,expected_value",
    [
        ("sae.k=32", ["sae", "k"], 32),
        ("run.force=true", ["run", "force"], True),
        ("sae.lr=1e-4", ["sae", "lr"], 1e-4),
        ("name=smoke_v0", ["name"], "smoke_v0"),
        ("a.b.c=null", ["a", "b", "c"], None),
    ],
)
def test_parse_override_types(text, expected_path, expected_value):
    path, value = parse_override(text)
    assert path == expected_path
    assert value == expected_value


@pytest.mark.parametrize("bad", ["nokey", "=value"])
def test_malformed_override_rejected(bad):
    with pytest.raises(ConfigError):
        parse_override(bad)


def test_overrides_create_missing_nesting():
    assert apply_overrides({}, ["a.b.c=1"]) == {"a": {"b": {"c": 1}}}


def test_overrides_do_not_mutate_input():
    original = {"a": {"b": 1}}
    apply_overrides(original, ["a.b=2"])
    assert original == {"a": {"b": 1}}


def test_require_and_get():
    config = {"sae": {"k": 32}}
    assert require(config, "sae.k") == 32
    assert get(config, "sae.missing", "fallback") == "fallback"
    with pytest.raises(ConfigError, match="missing required config key"):
        require(config, "sae.missing")


def test_load_config_layers_defaults_file_and_overrides(tmp_path):
    config_file = tmp_path / "c.yaml"
    config_file.write_text("sae:\n  k: 16\n  d_sae: 2048\n", encoding="utf-8")
    resolved = load_config(config_file, ["sae.k=32"], defaults={"sae": {"lr": 3e-4}})
    assert resolved == {"sae": {"lr": 3e-4, "k": 32, "d_sae": 2048}}


def test_missing_config_file_rejected():
    with pytest.raises(ConfigError, match="not found"):
        load_config("does-not-exist.yaml")


def test_non_mapping_config_rejected(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_shipped_configs_parse(repo_root):
    """Every YAML config in the repo must load and contain the expected blocks."""
    configs = sorted((repo_root / "configs").rglob("*.yaml"))
    assert configs, "no configs found"
    for path in configs:
        config = load_config(path)
        assert isinstance(config, dict) and config, f"{path} is empty"
        json.dumps(config)  # must be JSON-serialisable for provenance recording
