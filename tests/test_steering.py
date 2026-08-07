"""Strength schedules and intervention planning.

The zero-strength tests here are the local half of the project's most important
correctness property: with everything zeroed, the plan must resolve to *no
edits at all*, so the hook can leave the residual stream untouched. Approximate
equality would not be good enough -- a baseline that is "nearly" baseline makes
every measured effect suspect.
"""

from __future__ import annotations

import pytest

from brainpatch.schemas.patch import BrainPatchSpec, FeatureEdit, SAEReference
from brainpatch.steering.plan import InterventionPlan
from brainpatch.steering.schedule import StrengthSchedule


def make_spec(name: str = "p", feature_id: int = 727, strength: float = 2.0, layer: int = 18):
    spec = BrainPatchSpec(
        name=name,
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        sae=SAEReference(
            reference="smoke_v0", layer=layer, hook="residual_post", d_in=1536, d_sae=2048
        ),
        features=[FeatureEdit(feature_id=feature_id, strength=strength)],
    )
    spec.validate()
    return spec


# ---------------------------------------------------------------------------
# StrengthSchedule
# ---------------------------------------------------------------------------


def test_step_hold_semantics():
    schedule = StrengthSchedule({0: 0.0, 20: 1.0, 40: 2.0})
    assert schedule.strength_at(0) == 0.0
    assert schedule.strength_at(19) == 0.0
    assert schedule.strength_at(20) == 1.0
    assert schedule.strength_at(39) == 1.0
    assert schedule.strength_at(40) == 2.0
    assert schedule.strength_at(10_000) == 2.0


def test_default_applies_before_first_keyframe():
    schedule = StrengthSchedule({10: 5.0}, default=0.5)
    assert schedule.strength_at(0) == 0.5
    assert schedule.strength_at(9) == 0.5
    assert schedule.strength_at(10) == 5.0


def test_interpolation_is_linear_between_keyframes():
    schedule = StrengthSchedule({0: 0.0, 10: 1.0}, interpolate=True)
    assert schedule.strength_at(0) == 0.0
    assert schedule.strength_at(5) == pytest.approx(0.5)
    assert schedule.strength_at(10) == 1.0
    # past the final keyframe the last value holds
    assert schedule.strength_at(50) == 1.0


def test_keys_are_normalized_and_sorted():
    schedule = StrengthSchedule({"40": 2.0, "0": 0.0, "20": 1.0})  # type: ignore[arg-type]
    assert schedule.steps == [0, 20, 40]


def test_empty_schedule_rejected():
    with pytest.raises(ValueError, match="at least one keyframe"):
        StrengthSchedule({})


def test_negative_index_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        StrengthSchedule({-1: 1.0})


def test_non_numeric_value_rejected():
    with pytest.raises(ValueError, match="numeric"):
        StrengthSchedule({0: "loud"})  # type: ignore[dict-item]


def test_negative_query_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        StrengthSchedule({0: 1.0}).strength_at(-1)


def test_from_dict_accepts_bare_shorthand():
    schedule = StrengthSchedule.from_dict({"0": 0.0, "24": 1.5})
    assert schedule.strength_at(24) == 1.5


def test_from_dict_accepts_full_form():
    schedule = StrengthSchedule.from_dict(
        {"keyframes": {"0": 1.0, "10": 2.0}, "interpolate": True, "default": 0.0}
    )
    assert schedule.interpolate is True
    assert schedule.strength_at(5) == pytest.approx(1.5)


def test_constant_schedule():
    schedule = StrengthSchedule.constant(3.0)
    assert schedule.strength_at(0) == 3.0
    assert schedule.strength_at(500) == 3.0
    assert schedule.is_constant()


def test_round_trip():
    original = StrengthSchedule({0: 0.0, 24: 1.0}, interpolate=True, default=0.25)
    restored = StrengthSchedule.from_dict(original.to_dict())
    assert restored.keyframes == original.keyframes
    assert restored.interpolate == original.interpolate
    assert restored.default == original.default


# ---------------------------------------------------------------------------
# InterventionPlan
# ---------------------------------------------------------------------------


def test_install_and_resolve():
    plan = InterventionPlan()
    plan.install(make_spec(strength=2.0))
    edits = plan.edits_at(0, layer=18)
    assert len(edits) == 1
    assert edits[0].feature_id == 727
    assert edits[0].coefficient == pytest.approx(2.0)


def test_patch_strength_multiplies_feature_strength():
    plan = InterventionPlan()
    plan.install(make_spec(strength=2.0), strength=1.5)
    assert plan.edits_at(0, layer=18)[0].coefficient == pytest.approx(3.0)


def test_zero_patch_strength_produces_no_edits():
    """The property that makes strength=0 exactly baseline."""
    plan = InterventionPlan()
    plan.install(make_spec(strength=2.0), strength=0.0)
    assert plan.edits_at(0, layer=18) == []
    assert plan.is_active(0) is False


def test_zero_feature_strength_produces_no_edits():
    plan = InterventionPlan()
    plan.install(make_spec(strength=0.0))
    assert plan.edits_at(0, layer=18) == []


def test_disabled_patch_produces_no_edits():
    plan = InterventionPlan()
    plan.install(make_spec())
    plan.set_enabled("p", False)
    assert plan.edits_at(0, layer=18) == []
    plan.set_enabled("p", True)
    assert plan.edits_at(0, layer=18) != []


def test_no_patches_means_no_edits():
    assert InterventionPlan().edits_at(0) == []


def test_schedule_gates_edits_by_token_index():
    plan = InterventionPlan()
    plan.install(make_spec(strength=2.0))
    plan.set_schedule("p", StrengthSchedule({0: 0.0, 24: 1.0}))
    assert plan.edits_at(0, layer=18) == []
    assert plan.edits_at(23, layer=18) == []
    assert plan.edits_at(24, layer=18)[0].coefficient == pytest.approx(2.0)


def test_schedule_from_spec_is_picked_up_on_install():
    spec = make_spec()
    spec.schedule = {"0": 0.0, "10": 1.0}
    plan = InterventionPlan()
    plan.install(spec)
    assert plan.edits_at(5, layer=18) == []
    assert plan.edits_at(10, layer=18) != []


def test_layer_filter():
    plan = InterventionPlan()
    plan.install(make_spec(name="a", layer=18))
    plan.install(make_spec(name="b", feature_id=1, layer=12))
    assert plan.layers() == [12, 18]
    assert len(plan.edits_at(0, layer=18)) == 1
    assert len(plan.edits_at(0, layer=12)) == 1
    assert len(plan.edits_at(0)) == 2


def test_token_range_restricts_application():
    plan = InterventionPlan(token_range=(10, 20))
    plan.install(make_spec())
    assert plan.edits_at(9) == []
    assert plan.edits_at(10) != []
    assert plan.edits_at(19) != []
    assert plan.edits_at(20) == []


def test_multiple_patches_accumulate():
    plan = InterventionPlan()
    plan.install(make_spec(name="a", feature_id=1, strength=1.0))
    plan.install(make_spec(name="b", feature_id=2, strength=-3.0))
    edits = plan.edits_at(0, layer=18)
    assert {e.feature_id: e.coefficient for e in edits} == pytest.approx({1: 1.0, 2: -3.0})


def test_reinstalling_same_name_replaces():
    plan = InterventionPlan()
    plan.install(make_spec(strength=1.0))
    plan.install(make_spec(strength=5.0))
    assert len(plan.patches) == 1
    assert plan.edits_at(0, layer=18)[0].coefficient == pytest.approx(5.0)


def test_uninstall():
    plan = InterventionPlan()
    plan.install(make_spec())
    plan.uninstall("p")
    assert plan.patches == {}
    with pytest.raises(KeyError):
        plan.uninstall("p")


@pytest.mark.parametrize("method", ["set_strength", "set_enabled", "set_schedule"])
def test_operations_on_unknown_patch_raise(method):
    plan = InterventionPlan()
    with pytest.raises(KeyError, match="no patch named"):
        getattr(plan, method)("nope", None if method == "set_schedule" else 1)


def test_negative_strength_survives_as_suppression():
    plan = InterventionPlan()
    plan.install(make_spec(strength=-4.0))
    assert plan.edits_at(0, layer=18)[0].coefficient == pytest.approx(-4.0)
