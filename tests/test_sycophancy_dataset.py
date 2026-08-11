"""Invariants of the anti-sycophancy dataset.

These guard the properties that make the experiment interpretable. A dataset
bug here does not crash anything -- it silently produces a result that looks
like evidence and is not. That is exactly the failure mode worth a test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brainpatch.datasets import load_contrast_set
from brainpatch.research.sycophancy_data import (
    FALSE_CLAIMS,
    INVITATIONS,
    TRUE_CLAIMS,
    build_examples,
    split_counts,
)

pytestmark = pytest.mark.local

FIXTURE = Path(__file__).resolve().parent.parent / "examples" / "contrast" / "antisycophancy_v1.json"


@pytest.fixture(scope="module")
def examples() -> list[dict]:
    return build_examples()


def test_every_claim_becomes_exactly_one_example(examples):
    assert len(examples) == len(FALSE_CLAIMS) + len(TRUE_CLAIMS)


def test_topics_are_unique(examples):
    topics = [e["metadata"]["topic"] for e in examples]
    assert len(set(topics)) == len(topics)


def test_no_assertion_is_reused(examples):
    """No proposition is restated in a second wording anywhere in the pool."""
    assertions = [c[2] for c in FALSE_CLAIMS + TRUE_CLAIMS]
    assert len(set(assertions)) == len(assertions)


def test_no_topic_crosses_a_split(examples):
    by_split: dict[str, set[str]] = {}
    for example in examples:
        meta = example["metadata"]
        by_split.setdefault(meta["split"], set()).add(meta["topic"])
    names = sorted(by_split)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not by_split[left] & by_split[right], f"topic leaked {left}->{right}"


def test_invitation_pools_are_disjoint_across_splits():
    """The wrapper phrasing must not be learnable as the signal."""
    pools = {k: set(v) for k, v in INVITATIONS.items()}
    names = sorted(pools)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert not pools[left] & pools[right]


def test_both_polarities_present_in_every_split(examples):
    counts = split_counts(examples)
    assert set(counts) == {"train", "validation", "test"}
    for split, row in counts.items():
        assert row["false_claim"] > 0, split
        assert row["true_claim"] > 0, split
        # The control has to be a substantial minority, not a token gesture:
        # too few true items and a contrarian direction passes unnoticed.
        assert row["true_claim"] / row["total"] > 0.2, split


def test_every_category_appears_in_every_split(examples):
    by_split: dict[str, set[str]] = {}
    for example in examples:
        by_split.setdefault(example["metadata"]["split"], set()).add(example["category"])
    reference = set.union(*by_split.values())
    for split, categories in by_split.items():
        assert categories == reference, f"{split} is missing {sorted(reference - categories)}"


def test_at_least_eight_categories(examples):
    assert len({e["category"] for e in examples}) >= 8


def test_true_and_false_items_are_labelled_consistently(examples):
    false_topics = {c[0] for c in FALSE_CLAIMS}
    for example in examples:
        meta = example["metadata"]
        expected = "false_claim" if meta["topic"] in false_topics else "true_claim"
        assert meta["polarity"] == expected


def test_no_pair_is_wildly_unbalanced(examples):
    """No single pair may be more than 2x apart in length.

    An extreme pair lets the log-probability margin be driven by length rather
    than by stance. This bounds the per-item damage; the systematic skew is a
    separate problem, documented in the test below.
    """
    for example in examples:
        positive = len(example["positive_response"])
        negative = len(example["negative_response"])
        ratio = max(positive, negative) / max(1, min(positive, negative))
        assert ratio < 2.0, example["metadata"]["topic"]


def test_the_known_length_skew_is_still_what_we_documented(examples):
    """The positive response is systematically the longer one. This is a real
    confound and it is recorded here rather than hidden.

    Correcting a false claim, or confirming a true one with a reason, takes more
    words than agreeing does. Every mitigation depends on the *direction* of the
    skew being uniform, so if that ever stops being true this test should fail
    and the analysis needs revisiting:

    - the primary metric is the **per-token** margin, not the total;
    - the pre-registered endpoint subtracts the true-claim margin from the
      false-claim margin, and since the longer side is the positive one on
      *both* polarities, a direction that merely prefers longer text cancels;
    - scale-matched random directions are scored on the same items, so a length
      preference would have to beat them to count.
    """
    longer = sum(
        1
        for e in examples
        if len(e["positive_response"]) > len(e["negative_response"])
    )
    fraction = longer / len(examples)
    assert fraction > 0.85, (
        f"the positive response is longer in only {fraction:.0%} of pairs; the skew "
        "is no longer one-directional, so the cancellation argument above needs redoing"
    )


def test_committed_fixture_matches_the_claim_pool(examples):
    """The committed JSON must be what the current pool generates."""
    assert FIXTURE.is_file(), "run scripts/build_sycophancy_dataset.py"
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["examples"] == examples, "fixture is stale; regenerate it"
    assert payload["synthetic"] is True


def test_fixture_loads_through_the_normal_loader():
    contrast = load_contrast_set("antisycophancy_v1")
    contrast.validate()
    assert len(contrast) == len(FALSE_CLAIMS) + len(TRUE_CLAIMS)


def test_original_fixture_is_untouched():
    """The v0 set stays available so the earlier negative result reproduces."""
    contrast = load_contrast_set("antisycophancy_eval")
    assert len(contrast) == 20
