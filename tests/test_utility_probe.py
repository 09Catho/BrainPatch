"""The utility battery, pinned so its scoring cannot drift with results."""

from __future__ import annotations

import pytest

from brainpatch.research.utility_probe import (
    UTILITY_ITEMS,
    _check,
    score_utility,
    utility_prompts,
)

pytestmark = pytest.mark.local


def test_every_item_is_well_formed():
    for category, prompt, kind, expected in UTILITY_ITEMS:
        assert category and prompt and kind and expected
        assert prompt.strip() == prompt


def test_prompts_match_items():
    assert len(utility_prompts()) == len(UTILITY_ITEMS)


def test_battery_covers_the_required_categories():
    categories = {c for c, _, _, _ in UTILITY_ITEMS}
    assert {"arithmetic", "factual_qa", "instruction", "reasoning", "helpfulness"} <= categories


def test_no_utility_prompt_resembles_the_target_task():
    """Utility must be measured on something the patch is not aimed at.

    If these prompts invited agreement, a patch that changes agreement
    behaviour would score as a utility regression for the wrong reason.
    """
    for _, prompt, _, _ in UTILITY_ITEMS:
        lowered = prompt.lower()
        assert "right?" not in lowered
        assert "do you agree" not in lowered
        assert "isn't it" not in lowered
        assert "am i" not in lowered


@pytest.mark.parametrize(
    "kind,expected,response,result",
    [
        ("contains", "115", "The answer is 115.", True),
        ("contains", "115", "The answer is 116.", False),
        ("exact_word", "banana", "banana", True),
        ("exact_word", "banana", "The answer is banana.", False),
        ("max_words", "10", "The Danube.", True),
        ("max_words", "10", " ".join(["word"] * 11), False),
        ("min_words", "5", "Keep it in a sealed container please", True),
        ("min_words", "5", "Yes.", False),
        ("comma_count", "2", "red, green, blue", True),
        ("comma_count", "2", "red and green", False),
        ("starts_with", "answer:", "Answer: 42", True),
        ("starts_with", "answer:", "42", False),
        ("short_answer", "10", "yes", True),
        ("short_answer", "10", " ".join(["word"] * 20), False),
    ],
)
def test_checkers(kind, expected, response, result):
    assert _check(kind, expected, response) is result


def test_unknown_checker_raises():
    with pytest.raises(ValueError):
        _check("mystery", "x", "y")


def test_perfect_run_scores_one():
    responses = []
    for _, _, kind, expected in UTILITY_ITEMS:
        if kind == "contains":
            responses.append(f"The answer is {expected}.")
        elif kind == "exact_word":
            responses.append(expected)
        elif kind == "comma_count":
            responses.append("red, green, blue")
        elif kind == "starts_with":
            responses.append("Answer: 42")
        else:
            responses.append("one two three four five six")
    result = score_utility(responses)
    assert result["accuracy"] == 1.0
    assert result["n"] == len(UTILITY_ITEMS)


def test_refusals_are_counted():
    responses = ["I can't help with that."] * len(UTILITY_ITEMS)
    result = score_utility(responses)
    assert result["refusal_rate"] == 1.0
    assert result["accuracy"] < 0.2


def test_empty_responses_are_visible():
    result = score_utility([""] * len(UTILITY_ITEMS))
    assert result["empty_rate"] == 1.0


def test_repetition_is_reported():
    looping = "the same thing over and over " * 12
    result = score_utility([looping] * len(UTILITY_ITEMS))
    assert result["max_ngram_repetition"] > 0.15


def test_result_always_carries_sample_size():
    """No accuracy figure should be readable without its n."""
    result = score_utility(["x"] * len(UTILITY_ITEMS))
    assert result["n"] == len(UTILITY_ITEMS)
    for row in result["by_category"].values():
        assert "n" in row


def test_wrong_response_count_raises():
    with pytest.raises(ValueError):
        score_utility(["only one"])
