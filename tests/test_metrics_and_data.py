"""Model-free metrics, contrast fixtures, and patch file I/O."""

from __future__ import annotations

import pytest

from brainpatch.datasets import CONTRAST_SET_NAMES, list_contrast_sets, load_contrast_set
from brainpatch.evaluation.metrics import (
    compare_generations,
    distinct_n,
    jaccard_similarity,
    longest_repeated_ngram,
    most_common_ngram_fraction,
    score_generation,
    tokenize_words,
)
from brainpatch.schemas.patch_io import discover_patches, load_patch, load_patch_dir, save_patch
from brainpatch.schemas.contrast import ContrastExample, ContrastSet

# Real generations captured from the smoke_v0 strength sweep on Modal. Using
# actual observed output rather than invented strings keeps the degeneration
# thresholds anchored to behaviour the model really produced.
CLEAN_GENERATION = (
    "The sky appears blue because the Earth's atmosphere scatters sunlight in all "
    "directions, and blue light is scattered more than other colors due to its "
    "shorter wavelength. This phenomenon is known as Rayleigh scattering."
)
LOOPING_GENERATION = (
    "As we look at an object from an observer, as it is exposed to light and light "
    "waves, as they pass through their particles, as they encounter each other, as "
    "they interact with each other, as they collide, as they merge, as they combine, "
    "as they fuse, as they interpenetate, as"
)
COLLAPSED_GENERATION = "As H H H H H H H H H H H F F F F F F F F F F F F F F F F F F F F F"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_tokenizer_splits_words_and_punctuation():
    assert tokenize_words("Hello, world!") == ["hello", ",", "world", "!"]


def test_distinct_n_is_one_for_unique_text():
    assert distinct_n(tokenize_words("a b c d e"), 2) == 1.0


def test_distinct_n_falls_with_repetition():
    repeated = tokenize_words("a b a b a b a b")
    assert distinct_n(repeated, 2) < 0.5


def test_distinct_n_on_short_text_is_zero():
    assert distinct_n(["a"], 2) == 0.0


def test_distinct_n_rejects_non_positive_n():
    with pytest.raises(ValueError):
        distinct_n(["a", "b"], 0)


def test_longest_repeated_ngram_finds_the_loop():
    tokens = tokenize_words("the cat sat on the mat the cat sat on the mat")
    assert longest_repeated_ngram(tokens) >= 6


def test_longest_repeated_ngram_is_low_for_clean_text():
    assert longest_repeated_ngram(tokenize_words(CLEAN_GENERATION)) < 5


def test_most_common_ngram_fraction_detects_clause_loops():
    """The metric added after the heuristic missed a real degenerate output."""
    clean = most_common_ngram_fraction(tokenize_words(CLEAN_GENERATION), 2)
    looping = most_common_ngram_fraction(tokenize_words(LOOPING_GENERATION), 2)
    assert clean < 0.05
    assert looping > 0.10


def test_degeneration_flag_on_real_generations():
    assert score_generation(CLEAN_GENERATION).degeneration_flag is False
    assert score_generation(LOOPING_GENERATION).degeneration_flag is True
    assert score_generation(COLLAPSED_GENERATION).degeneration_flag is True


def test_empty_generation_is_flagged():
    metrics = score_generation("   ")
    assert metrics.is_empty
    assert metrics.degeneration_flag


def test_jaccard_of_identical_text_is_one():
    assert jaccard_similarity(CLEAN_GENERATION, CLEAN_GENERATION) == 1.0


def test_jaccard_of_unrelated_text_is_low():
    assert jaccard_similarity(CLEAN_GENERATION, COLLAPSED_GENERATION) < 0.1


def test_compare_generations_detects_identity():
    result = compare_generations(CLEAN_GENERATION, CLEAN_GENERATION)
    assert result["identical"] is True
    assert result["jaccard_3"] == 1.0
    assert result["degeneration_introduced"] is False


def test_compare_generations_detects_introduced_degeneration():
    result = compare_generations(CLEAN_GENERATION, LOOPING_GENERATION)
    assert result["identical"] is False
    assert result["degeneration_introduced"] is True


def test_metrics_dict_includes_flag():
    assert "degeneration_flag" in score_generation(CLEAN_GENERATION).to_dict()


# ---------------------------------------------------------------------------
# contrast fixtures
# ---------------------------------------------------------------------------


def test_all_named_contrast_sets_exist():
    assert set(CONTRAST_SET_NAMES).issubset(set(list_contrast_sets()))


@pytest.mark.parametrize("name", CONTRAST_SET_NAMES)
def test_contrast_sets_load_and_validate(name):
    contrast_set = load_contrast_set(name)
    contrast_set.validate()
    assert len(contrast_set) >= 8


@pytest.mark.parametrize("name", CONTRAST_SET_NAMES)
def test_contrast_sets_are_marked_synthetic(name):
    """A fixture must never be mistakable for a validated benchmark."""
    contrast_set = load_contrast_set(name)
    assert contrast_set.synthetic is True
    assert "SYNTHETIC" in contrast_set.description.upper()


def test_unknown_contrast_set_lists_alternatives():
    with pytest.raises(FileNotFoundError, match="Available"):
        load_contrast_set("does-not-exist")


def test_contrast_example_rejects_identical_responses():
    example = ContrastExample(prompt="p", positive_response="same", negative_response="same")
    with pytest.raises(ValueError, match="identical"):
        example.validate()


def test_contrast_example_rejects_empty_fields():
    with pytest.raises(ValueError, match="empty prompt"):
        ContrastExample(prompt="  ", positive_response="a", negative_response="b").validate()


def test_contrast_split_is_deterministic_and_disjoint():
    contrast_set = load_contrast_set("sycophancy")
    train_a, hold_a = contrast_set.split(0.25, seed=7)
    train_b, hold_b = contrast_set.split(0.25, seed=7)
    assert [e.prompt for e in train_a] == [e.prompt for e in train_b]
    assert [e.prompt for e in hold_a] == [e.prompt for e in hold_b]
    assert set(e.prompt for e in train_a).isdisjoint(e.prompt for e in hold_a)
    assert len(train_a) + len(hold_a) == len(contrast_set)


def test_contrast_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        load_contrast_set("sycophancy").split(1.5)


def test_contrast_filter_by_category():
    contrast_set = load_contrast_set("sycophancy")
    category = contrast_set.categories()[0]
    filtered = contrast_set.filter(category)
    assert all(e.category == category for e in filtered)


def test_contrast_round_trip():
    contrast_set = load_contrast_set("verification")
    restored = ContrastSet.from_json(contrast_set.to_json())
    assert len(restored) == len(contrast_set)


# ---------------------------------------------------------------------------
# patch file I/O
# ---------------------------------------------------------------------------


def test_shipped_patches_are_valid(repo_root):
    specs, failures = load_patch_dir(repo_root / "patches", strict=False)
    assert not failures, f"invalid shipped patches: {failures}"
    assert specs, "no patches shipped"


def test_shipped_patches_do_not_overclaim(repo_root):
    """No shipped patch may assert a causal label the experiments do not support."""
    specs, _ = load_patch_dir(repo_root / "patches", strict=False)
    for spec in specs:
        if spec.is_validated:
            assert spec.evaluation, (
                f"{spec.name} claims causal evidence but records no evaluation"
            )


def test_save_and_load_round_trip(tmp_path, valid_patch_dict):
    from brainpatch.schemas.patch import BrainPatchSpec

    spec = BrainPatchSpec.from_dict(valid_patch_dict)
    path = save_patch(spec, tmp_path / "p.json")
    assert load_patch(path).to_dict() == spec.to_dict()


def test_save_refuses_to_clobber(tmp_path, valid_patch_dict):
    from brainpatch.schemas.patch import BrainPatchSpec

    spec = BrainPatchSpec.from_dict(valid_patch_dict)
    save_patch(spec, tmp_path / "p.json")
    with pytest.raises(FileExistsError):
        save_patch(spec, tmp_path / "p.json")
    save_patch(spec, tmp_path / "p.json", overwrite=True)


def test_load_missing_patch_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_patch(tmp_path / "nope.json")


def test_discover_patches_on_missing_dir_returns_empty(tmp_path):
    assert discover_patches(tmp_path / "nope") == []


def test_load_patch_dir_non_strict_collects_failures(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    specs, failures = load_patch_dir(tmp_path, strict=False)
    assert specs == []
    assert len(failures) == 1
