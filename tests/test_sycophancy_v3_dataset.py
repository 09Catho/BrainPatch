"""Invariants of the `anti_sycophancy_v3` dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brainpatch.datasets import load_contrast_set
from brainpatch.research.sycophancy_data import FALSE_CLAIMS as V1_FALSE
from brainpatch.research.sycophancy_data import TRUE_CLAIMS as V1_TRUE
from brainpatch.research.sycophancy_data_v2 import CLAIMS as V2_CLAIMS
from brainpatch.research.sycophancy_data_v3 import CLAIMS
from brainpatch.research.sycophancy_v3_build import (
    INVITATIONS,
    SPLIT_SIZES,
    audit_lengths,
    audit_per_split,
    build_examples,
    dataset_hashes,
    near_duplicate_pairs,
    split_counts,
)

pytestmark = pytest.mark.local

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "contrast" / "antisycophancy_v3.json"
MANIFEST = ROOT / "experiments" / "anti_sycophancy_v3" / "dataset_manifest.json"


@pytest.fixture(scope="module")
def examples() -> list[dict]:
    return build_examples()


# --- freshness: the reason v3 needed a new pool ----------------------------


def test_no_topic_shared_with_v1_or_v2():
    previous = {c[0] for c in list(V1_FALSE) + list(V1_TRUE)} | {c[1] for c in V2_CLAIMS}
    assert not (previous & {c[1] for c in CLAIMS})


def test_no_assertion_shared_with_v1_or_v2():
    previous = {c[2] for c in list(V1_FALSE) + list(V1_TRUE)} | {c[4] for c in V2_CLAIMS}
    assert not (previous & {c[4] for c in CLAIMS})


def test_previous_experiments_remain_loadable():
    """v1 and v2 are frozen, not deleted."""
    assert len(load_contrast_set("antisycophancy_v1")) == 198
    assert len(load_contrast_set("antisycophancy_v2")) == 387
    assert len(load_contrast_set("antisycophancy_eval")) == 20


# --- split shape, which the generation metric depends on -------------------


def test_split_sizes_match_the_protocol(examples):
    counts = split_counts(examples)
    assert 200 <= counts["train"]["total"] <= 300
    assert 150 <= counts["validation"]["total"] <= 200
    assert 200 <= counts["test"]["total"] <= 300


def test_split_sizes_are_exactly_as_declared(examples):
    counts = split_counts(examples)
    for name, size in SPLIT_SIZES:
        assert counts[name]["total"] == size


def test_true_control_fraction_in_range(examples):
    true_count = sum(1 for e in examples if e["metadata"]["polarity"] == "true_claim")
    assert 0.35 <= true_count / len(examples) <= 0.40


def test_polarity_balance_is_consistent_across_splits(examples):
    """A split with a different true-claim share would make its SIS
    incomparable with the others."""
    shares = {
        name: row["true_claim"] / row["total"]
        for name, row in split_counts(examples).items()
    }
    assert max(shares.values()) - min(shares.values()) < 0.03, shares


def test_every_category_appears_in_every_split(examples):
    by_split: dict[str, set[str]] = {}
    for example in examples:
        by_split.setdefault(example["metadata"]["split"], set()).add(example["category"])
    reference = set.union(*by_split.values())
    for split, categories in by_split.items():
        assert categories == reference, f"{split} missing {sorted(reference - categories)}"


def test_categories_are_evenly_represented(examples):
    for split in ("train", "validation", "test"):
        counts: dict[str, int] = {}
        for example in examples:
            if example["metadata"]["split"] == split:
                counts[example["category"]] = counts.get(example["category"], 0) + 1
        assert max(counts.values()) - min(counts.values()) <= 3, (split, counts)


def test_length_polarity_balanced_within_each_split(examples):
    for split in ("train", "validation", "test"):
        subset = [e for e in examples if e["metadata"]["split"] == split]
        longer = sum(1 for e in subset if e["metadata"]["intended_longer"] == "desired")
        assert abs(longer / len(subset) - 0.5) < 0.05, split


def test_at_least_eleven_categories(examples):
    assert len({e["category"] for e in examples}) >= 11


def test_topics_unique_and_split_disjoint(examples):
    topics = [e["metadata"]["topic"] for e in examples]
    assert len(set(topics)) == len(topics)
    by_split: dict[str, set[str]] = {}
    for example in examples:
        meta = example["metadata"]
        by_split.setdefault(meta["split"], set()).add(meta["topic"])
    names = sorted(by_split)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not by_split[left] & by_split[right]


def test_invitation_pools_disjoint_across_splits():
    pools = {k: set(v) for k, v in INVITATIONS.items()}
    names = sorted(pools)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not pools[left] & pools[right]


# --- the length fix carried over from v2 -----------------------------------


def test_length_audit_passes_globally_and_per_split(examples):
    assert audit_lengths(examples).ok, audit_lengths(examples).render()
    for split, result in audit_per_split(examples).items():
        assert result.ok, f"{split}: {result.render()}"


def test_behavioural_class_barely_predicts_length(examples):
    assert abs(audit_lengths(examples).stats["label_length_corr"]) <= 0.15


def test_length_variation_survives_for_the_diagnostic(examples):
    """Log-probability is still reported, so the length diagnostic must remain
    computable: a gap distribution with no variance makes it undefined."""
    gaps = [len(e["positive_response"]) - len(e["negative_response"]) for e in examples]
    assert len(set(gaps)) > 20
    assert any(g > 10 for g in gaps) and any(g < -10 for g in gaps)


def test_no_near_duplicates(examples):
    assert near_duplicate_pairs(examples) == []


def test_assertion_is_carried_for_the_evaluator(examples):
    """The deterministic layer needs the claim, not just the prompt wrapper."""
    for example in examples:
        assert example["metadata"]["assertion"]
        assert example["metadata"]["assertion"] in example["prompt"]


def test_responses_are_distinct_and_non_empty(examples):
    for example in examples:
        assert example["positive_response"].strip()
        assert example["negative_response"].strip()
        assert example["positive_response"] != example["negative_response"]


# --- committed artifacts ---------------------------------------------------


def test_committed_fixture_is_current(examples):
    assert FIXTURE.is_file(), "run scripts/build_sycophancy_v3.py"
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["examples"] == examples, "fixture is stale; regenerate it"


def test_manifest_hashes_match(examples):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["sha256"] == dataset_hashes(examples)


def test_fixture_loads_through_the_normal_loader():
    contrast = load_contrast_set("antisycophancy_v3")
    contrast.validate()
    assert len(contrast) == len(CLAIMS)
