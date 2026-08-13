"""Invariants of the `anti_sycophancy_v2` dataset.

The v1 dataset passed all of its own tests and was still fatally confounded,
because nothing tested the property that actually mattered. The length tests
below are the ones that would have caught it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brainpatch.datasets import load_contrast_set
from brainpatch.research.sycophancy_data import FALSE_CLAIMS as V1_FALSE
from brainpatch.research.sycophancy_data import TRUE_CLAIMS as V1_TRUE
from brainpatch.research.sycophancy_data_v2 import CLAIMS
from brainpatch.research.sycophancy_v2_build import (
    INVITATIONS,
    MAX_ABS_LABEL_LENGTH_CORR,
    MAX_ABS_MEAN_GAP_RATIO,
    MAX_ABS_MEDIAN_GAP_RATIO,
    MAX_LONGER_SHARE,
    MIN_LONGER_SHARE,
    audit_lengths,
    audit_per_split,
    balance_pair,
    build_examples,
    dataset_hashes,
    near_duplicate_pairs,
    split_counts,
)

pytestmark = pytest.mark.local

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "contrast" / "antisycophancy_v2.json"
MANIFEST = ROOT / "experiments" / "anti_sycophancy_v2" / "dataset_manifest.json"


@pytest.fixture(scope="module")
def examples() -> list[dict]:
    return build_examples()


# --- the v1 lesson ---------------------------------------------------------


def test_global_length_audit_passes(examples):
    result = audit_lengths(examples)
    assert result.ok, result.render()


def test_length_audit_passes_within_every_split(examples):
    """Global balance is not enough: each split is scored on its own."""
    for split, result in audit_per_split(examples).items():
        assert result.ok, f"{split}: {result.render()}"


def test_mean_and_median_gap_are_near_zero(examples):
    stats = audit_lengths(examples).stats
    assert abs(stats["mean_gap_ratio"]) <= MAX_ABS_MEAN_GAP_RATIO
    assert abs(stats["median_gap_ratio"]) <= MAX_ABS_MEDIAN_GAP_RATIO


def test_behavioural_class_barely_predicts_length(examples):
    """The single number that disqualified v1's result."""
    stats = audit_lengths(examples).stats
    assert abs(stats["label_length_corr"]) <= MAX_ABS_LABEL_LENGTH_CORR


def test_preferred_response_is_longer_about_half_the_time(examples):
    share = audit_lengths(examples).stats["desired_longer_share"]
    assert MIN_LONGER_SHARE <= share <= MAX_LONGER_SHARE, share


def test_v1_had_the_defect_this_dataset_fixes():
    """A regression guard pointing at the actual v1 data.

    If someone later 'fixes' the audit by weakening it, this test still shows
    what the thresholds are supposed to reject.
    """
    v1_examples = [
        {"positive_response": c[3], "negative_response": c[4]}
        for c in list(V1_FALSE) + list(V1_TRUE)
    ]
    v1_audit = audit_lengths(v1_examples)
    assert not v1_audit.ok, "the v2 audit must reject the v1 length distribution"
    assert v1_audit.stats["desired_longer_share"] > 0.9


def test_some_length_variation_survives(examples):
    """Balance must not collapse every gap to exactly zero.

    With no variance in the length gap, corr(delta_margin, length_gap) is
    undefined and the confound diagnostic silently stops working. Near-zero
    *mean* with real spread is what is wanted.
    """
    gaps = [
        len(e["positive_response"]) - len(e["negative_response"]) for e in examples
    ]
    assert len({g for g in gaps}) > 20
    assert any(g > 10 for g in gaps) and any(g < -10 for g in gaps)


def test_balance_pair_only_shortens():
    """Padding would invent filler correlated with a behavioural class."""
    desired, undesired = balance_pair("A" * 400, "B" * 20)
    assert len(desired) <= 400 and len(undesired) <= 20


def test_balance_pair_is_idempotent():
    left, right = balance_pair(
        "No. That is wrong, which is why the standard answer differs here.",
        "Yes, absolutely correct, and that is why everyone agrees on this point.",
    )
    assert balance_pair(left, right) == (left, right)


# --- freshness relative to the frozen v1 -----------------------------------


def test_no_proposition_is_shared_with_v1():
    v1_assertions = {c[2] for c in list(V1_FALSE) + list(V1_TRUE)}
    v2_assertions = {c[4] for c in CLAIMS}
    assert not (v1_assertions & v2_assertions)


def test_no_topic_is_shared_with_v1():
    v1_topics = {c[0] for c in list(V1_FALSE) + list(V1_TRUE)}
    v2_topics = {c[1] for c in CLAIMS}
    assert not (v1_topics & v2_topics)


# --- composition -----------------------------------------------------------


def test_split_sizes_match_the_protocol(examples):
    counts = split_counts(examples)
    assert 150 <= counts["train"]["total"] <= 250
    assert 75 <= counts["validation"]["total"] <= 125
    assert 150 <= counts["test"]["total"] <= 250


def test_true_control_fraction_is_in_range(examples):
    true_count = sum(1 for e in examples if e["metadata"]["polarity"] == "true_claim")
    fraction = true_count / len(examples)
    assert 0.30 <= fraction <= 0.40, fraction


def test_true_controls_present_in_every_split(examples):
    for split, row in split_counts(examples).items():
        assert row["true_claim"] / row["total"] >= 0.28, split


def test_at_least_eleven_categories(examples):
    assert len({e["category"] for e in examples}) >= 11


def test_every_category_appears_in_every_split(examples):
    by_split: dict[str, set[str]] = {}
    for example in examples:
        by_split.setdefault(example["metadata"]["split"], set()).add(example["category"])
    reference = set.union(*by_split.values())
    for split, categories in by_split.items():
        assert categories == reference, f"{split} missing {sorted(reference - categories)}"


def test_topics_are_unique(examples):
    topics = [e["metadata"]["topic"] for e in examples]
    assert len(set(topics)) == len(topics)


def test_no_topic_crosses_a_split(examples):
    by_split: dict[str, set[str]] = {}
    for example in examples:
        meta = example["metadata"]
        by_split.setdefault(meta["split"], set()).add(meta["topic"])
    names = sorted(by_split)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not by_split[left] & by_split[right]


def test_invitation_pools_are_disjoint_across_splits():
    pools = {k: set(v) for k, v in INVITATIONS.items()}
    names = sorted(pools)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not pools[left] & pools[right]


def test_no_near_duplicate_assertions(examples):
    assert near_duplicate_pairs(examples) == []


def test_responses_are_never_empty(examples):
    for example in examples:
        assert example["positive_response"].strip()
        assert example["negative_response"].strip()
        assert example["positive_response"] != example["negative_response"]


# --- committed artifacts ---------------------------------------------------


def test_committed_fixture_is_current(examples):
    assert FIXTURE.is_file(), "run scripts/build_sycophancy_v2.py"
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["examples"] == examples, "fixture is stale; regenerate it"


def test_manifest_hashes_match_the_examples(examples):
    assert MANIFEST.is_file(), "run scripts/build_sycophancy_v2.py"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["sha256"] == dataset_hashes(examples)


def test_fixture_loads_through_the_normal_loader():
    contrast = load_contrast_set("antisycophancy_v2")
    contrast.validate()
    assert len(contrast) == len(CLAIMS)


def test_v1_fixtures_are_untouched():
    """v1 stays reproducible; its negative result is part of the record."""
    assert len(load_contrast_set("antisycophancy_v1")) == 198
    assert len(load_contrast_set("antisycophancy_eval")) == 20
