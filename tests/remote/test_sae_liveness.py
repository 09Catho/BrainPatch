"""Regression tests for Top-K liveness accounting.

**These run inside Modal, not locally.** They import torch, which the local
test suite deliberately blocks, so ``tests/conftest.py`` excludes this directory
from local collection. Run them with::

    modal run modal_app/app.py::sae_unit_tests

The bug they pin down
---------------------
``torch.topk`` always returns exactly ``k`` indices. The ReLU in front of it
means some of those selected *values* can be zero, whenever a token has fewer
than ``k`` positive encoder pre-activations.

The original ``update_liveness`` counted every selected index as a firing. That
inflated ``fire_count`` and, more seriously, reset ``tokens_since_fired`` for
features that never actually activated -- so a permanently silent feature would
be reported alive forever, never appear in ``dead_mask``, and never be revived
by AuxK.

Every test below drives the SAE into the fewer-than-k-positive regime on
purpose, which requires overriding the encoder weights: at realistic scale
(d_sae 2048) roughly half the pre-activations are positive and the condition
essentially never arises.
"""

from __future__ import annotations

import pytest
import torch

from brainpatch.ml.sae import TopKSAE, reconstruction_metrics
from brainpatch.schemas.sae import SAEConfig


def make_sae(*, d_in: int = 4, d_sae: int = 8, k: int = 4, **kwargs) -> TopKSAE:
    """A tiny SAE with a deterministic, fully-controlled encoder.

    ``auxk_k`` must be defaulted down: ``SAEConfig`` ships 256, and its own
    validation requires ``auxk_k <= d_sae``, so the production default is
    invalid at this toy scale.
    """
    kwargs.setdefault("auxk_alpha", 0.0)
    kwargs.setdefault("auxk_k", min(2, d_sae))
    config = SAEConfig(d_in=d_in, d_sae=d_sae, k=k, **kwargs)
    return TopKSAE(config)


def force_n_positive(sae: TopKSAE, n_positive: int) -> torch.Tensor:
    """Rig the SAE so exactly ``n_positive`` features have positive pre-acts.

    The encoder is zeroed and the bias set so that features ``0..n_positive-1``
    are positive and the rest are strongly negative. With a zero input the
    pre-activation is exactly the bias, so post-ReLU exactly ``n_positive``
    entries are non-zero regardless of the input vector.

    Returns an input tensor of shape ``[1, d_in]``.
    """
    with torch.no_grad():
        sae.W_enc.zero_()
        sae.b_dec.zero_()
        sae.b_enc.fill_(-1.0)
        sae.b_enc[:n_positive] = torch.arange(1.0, n_positive + 1.0)
    return torch.zeros(1, sae.d_in)


# ---------------------------------------------------------------------------
# the invariant itself
# ---------------------------------------------------------------------------


def test_topk_returns_k_indices_even_when_fewer_are_positive():
    """The premise: torch.topk pads the selection with zero-valued entries."""
    sae = make_sae(d_sae=8, k=4)
    x = force_n_positive(sae, 2)
    out = sae(x)

    assert out.topk_indices.shape == (1, 4), "topk always returns k indices"
    assert out.topk_values.shape == (1, 4)
    # Two of the four selected values are zero.
    assert int((out.topk_values > 0).sum().item()) == 2
    assert int((out.topk_values == 0).sum().item()) == 2


def test_l0_is_at_most_k_not_exactly_k():
    """Measured L0 counts positive activations, so it falls below k here."""
    sae = make_sae(d_sae=8, k=4)
    x = force_n_positive(sae, 2)
    out = sae(x)

    assert int(out.l0().item()) == 2
    assert int((out.feature_acts > 0).sum().item()) == 2
    assert reconstruction_metrics(x, out)["l0"] == pytest.approx(2.0)


def test_active_mask_matches_positive_values():
    sae = make_sae(d_sae=8, k=4)
    out = sae(force_n_positive(sae, 3))
    assert out.active_mask().sum().item() == 3


# ---------------------------------------------------------------------------
# fire_count must not be inflated by zero-valued selections
# ---------------------------------------------------------------------------


def test_fire_count_ignores_zero_valued_selections():
    """The core regression: only strictly positive selections are firings."""
    sae = make_sae(d_sae=8, k=4)
    x = force_n_positive(sae, 2)
    out = sae(x)

    sae.update_liveness(out.topk_indices, out.topk_values, x.shape[0])

    # Features 0 and 1 are the positive ones; everything else must stay at zero
    # even though two of them were *selected* by topk.
    assert sae.fire_count[0].item() == 1
    assert sae.fire_count[1].item() == 1
    assert sae.fire_count[2:].sum().item() == 0, (
        "zero-valued Top-K selections incremented fire_count"
    )
    assert sae.fire_count.sum().item() == 2, "total firings must equal positive selections"


def test_fire_count_total_is_below_k_times_tokens_when_saturating():
    """Under the bug this sum would be exactly k * tokens."""
    sae = make_sae(d_sae=8, k=4)
    x = force_n_positive(sae, 2).repeat(5, 1)
    out = sae(x)

    sae.update_liveness(out.topk_indices, out.topk_values, x.shape[0])

    naive = sae.k * x.shape[0]  # what index-counting would have produced
    assert sae.fire_count.sum().item() == 2 * x.shape[0]
    assert sae.fire_count.sum().item() < naive


def test_fire_count_matches_k_when_all_selections_are_positive():
    """No regression in the ordinary case: k positive selections, k firings."""
    sae = make_sae(d_sae=8, k=4)
    x = force_n_positive(sae, 8)  # all features positive
    out = sae(x)

    sae.update_liveness(out.topk_indices, out.topk_values, x.shape[0])
    assert sae.fire_count.sum().item() == sae.k * x.shape[0]


# ---------------------------------------------------------------------------
# the dead-feature timer must not be reset by zero-valued selections
# ---------------------------------------------------------------------------


def test_zero_valued_selection_does_not_reset_dead_timer():
    """The consequential half of the bug: a silent feature staying 'alive'."""
    sae = make_sae(d_sae=8, k=4, dead_feature_window=10)
    x = force_n_positive(sae, 2)

    # Age every feature well past the dead window.
    with torch.no_grad():
        sae.tokens_since_fired.fill_(1000)

    out = sae(x)
    sae.update_liveness(out.topk_indices, out.topk_values, x.shape[0])

    # Features 0 and 1 fired, so their clocks reset.
    assert sae.tokens_since_fired[0].item() == 0
    assert sae.tokens_since_fired[1].item() == 0

    # Every other feature must still be counted as long-silent, including the
    # two that topk selected with a zero value.
    assert (sae.tokens_since_fired[2:] > 10).all(), (
        "zero-valued Top-K selections reset the dead-feature timer"
    )


def test_dead_mask_still_reports_silent_features():
    sae = make_sae(d_sae=8, k=4, dead_feature_window=10)
    x = force_n_positive(sae, 2)
    with torch.no_grad():
        sae.tokens_since_fired.fill_(1000)

    out = sae(x)
    sae.update_liveness(out.topk_indices, out.topk_values, x.shape[0])

    dead = sae.dead_mask()
    assert not bool(dead[0])
    assert not bool(dead[1])
    assert sae.num_dead() == 6, "the six never-firing features must remain dead"


def test_repeated_zero_selection_lets_timer_keep_advancing():
    """A feature selected-but-never-positive eventually crosses the window."""
    sae = make_sae(d_sae=8, k=4, dead_feature_window=10)
    x = force_n_positive(sae, 2)

    for _ in range(6):
        out = sae(x)
        sae.update_liveness(out.topk_indices, out.topk_values, x.shape[0])

    assert sae.tokens_since_fired[0].item() == 0
    assert sae.tokens_since_fired[2:].min().item() == 6
    # window is 10, so 6 tokens is not yet dead -- but the clock is advancing,
    # which is the property the bug destroyed.
    assert sae.num_dead() == 0

    for _ in range(6):
        out = sae(x)
        sae.update_liveness(out.topk_indices, out.topk_values, x.shape[0])

    assert sae.num_dead() == 6, "silent features must eventually be flagged dead"


# ---------------------------------------------------------------------------
# AuxK interaction
# ---------------------------------------------------------------------------


def test_auxk_sees_features_the_fixed_accounting_marks_dead():
    """AuxK depends on dead_mask, so the fix restores its input."""
    sae = make_sae(d_sae=8, k=4, dead_feature_window=10, auxk_alpha=1.0, auxk_k=2)
    x = force_n_positive(sae, 2)
    with torch.no_grad():
        sae.tokens_since_fired.fill_(1000)

    out = sae(x)
    sae.update_liveness(out.topk_indices, out.topk_values, x.shape[0])

    assert sae.num_dead() == 6
    loss = sae.auxk_loss(x, out)
    assert torch.isfinite(loss), "AuxK must produce a finite loss over dead features"


def test_auxk_returns_zero_when_nothing_is_dead():
    sae = make_sae(d_sae=8, k=4, auxk_alpha=1.0)
    x = force_n_positive(sae, 8)
    out = sae(x)
    assert sae.auxk_loss(x, out).item() == 0.0


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_update_liveness_rejects_mismatched_shapes():
    """Guards against a caller passing the old single-argument form."""
    sae = make_sae()
    out = sae(force_n_positive(sae, 4))
    with pytest.raises(ValueError, match="same shape"):
        sae.update_liveness(out.topk_indices, out.topk_values[:, :2], 1)


def test_update_liveness_handles_no_positive_selections():
    """All-zero pre-activations must not crash bincount or reset any clock."""
    sae = make_sae(d_sae=8, k=4, dead_feature_window=10)
    with torch.no_grad():
        sae.W_enc.zero_()
        sae.b_enc.fill_(-1.0)
        sae.b_dec.zero_()
        sae.tokens_since_fired.fill_(1000)

    x = torch.zeros(3, sae.d_in)
    out = sae(x)
    assert int((out.topk_values > 0).sum().item()) == 0

    sae.update_liveness(out.topk_indices, out.topk_values, x.shape[0])

    assert sae.fire_count.sum().item() == 0
    assert sae.num_dead() == 8
    assert sae.tokens_seen.item() == 3


def test_reconstruction_metrics_reports_zero_selection_rate():
    sae = make_sae(d_sae=8, k=4)
    out = sae(force_n_positive(sae, 2))
    metrics = reconstruction_metrics(torch.zeros(1, sae.d_in), out)
    assert metrics["zero_selection_rate"] == pytest.approx(0.5)
    assert metrics["l0_min"] == pytest.approx(2.0)


def test_reconstruction_metrics_zero_selection_rate_is_zero_normally():
    sae = make_sae(d_sae=8, k=4)
    out = sae(force_n_positive(sae, 8))
    metrics = reconstruction_metrics(torch.zeros(1, sae.d_in), out)
    assert metrics["zero_selection_rate"] == pytest.approx(0.0)
    assert metrics["l0"] == pytest.approx(4.0)
