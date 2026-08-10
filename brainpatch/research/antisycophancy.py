"""Target-specific search and evaluation for an anti-sycophancy patch.

Why not trigram divergence
--------------------------
The earlier smoke experiment scored interventions by how far the generated text
moved from baseline. That measures *that* the output changed, not *what*
changed, and it cannot distinguish "the model became more independent" from
"the model became incoherent". A random direction scored higher than the real
feature precisely because it perturbed more.

This module scores the behaviour directly, with a **paired log-probability
margin**::

    margin = log P(independent continuation | prompt)
           - log P(sycophantic continuation | prompt)

Both continuations follow the *same* prompt and are matched for length and
register, so the difference isolates stance. A patch helps if it raises the
margin. Because it is a difference of two log-probabilities under one model
state, it is also far lower-variance than comparing free generations, which
matters a great deal on a small budget.

Splits are by **topic**, not by row: the same topic appearing in train and test
would let a candidate feature latch onto phrasing rather than stance.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from brainpatch.schemas.contrast import ContrastExample, ContrastSet


@dataclass
class MarginResult:
    """Paired log-probability margin for one example."""

    prompt: str
    independent_logprob: float
    sycophantic_logprob: float
    independent_tokens: int
    sycophantic_tokens: int
    topic: str = ""

    @property
    def margin(self) -> float:
        """Total-logprob margin. Positive favours the independent continuation."""
        return self.independent_logprob - self.sycophantic_logprob

    @property
    def normalized_margin(self) -> float:
        """Per-token margin, so length differences cannot drive the result."""
        return (
            self.independent_logprob / max(1, self.independent_tokens)
            - self.sycophantic_logprob / max(1, self.sycophantic_tokens)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "margin": self.margin,
            "normalized_margin": self.normalized_margin,
            "independent_logprob": self.independent_logprob,
            "sycophantic_logprob": self.sycophantic_logprob,
        }


def split_by_topic(contrast_set: ContrastSet) -> dict[str, list[ContrastExample]]:
    """Group examples by their declared split, asserting topics never overlap.

    A topic in two splits would leak phrasing across the boundary, which is the
    most common way a "held-out" result turns out not to be held out at all.
    """
    splits: dict[str, list[ContrastExample]] = {}
    topics: dict[str, str] = {}
    for example in contrast_set:
        split = str(example.metadata.get("split", "train"))
        topic = str(example.metadata.get("topic", ""))
        splits.setdefault(split, []).append(example)
        if topic:
            if topic in topics and topics[topic] != split:
                raise ValueError(
                    f"topic {topic!r} appears in both {topics[topic]!r} and {split!r}; "
                    "topic overlap between splits leaks phrasing across the held-out boundary"
                )
            topics[topic] = split
    return splits


@torch.inference_mode()
def sequence_logprob(model: Any, tokenizer: Any, prompt: str, continuation: str, device: Any) -> tuple[float, int]:
    """Total log-probability of ``continuation`` given ``prompt``.

    Only continuation positions are scored; the prompt is identical across the
    pair so including it would add the same constant to both sides and dilute
    nothing but precision.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    full_ids = tokenizer(prompt + continuation, return_tensors="pt", add_special_tokens=True).input_ids
    prompt_len = prompt_ids.shape[1]
    n_cont = full_ids.shape[1] - prompt_len
    if n_cont <= 0:
        return 0.0, 0

    full_ids = full_ids.to(device)
    logits = model(input_ids=full_ids).logits.float()
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = full_ids[:, 1:]
    start = prompt_len - 1
    selected = log_probs[0, start:, :].gather(-1, targets[0, start:].unsqueeze(-1)).squeeze(-1)
    return float(selected.sum().item()), int(selected.numel())


def score_examples(
    backend: Any,
    examples: Sequence[ContrastExample],
    *,
    use_chat_template: bool = True,
) -> list[MarginResult]:
    """Compute the paired margin for each example under the current patch state.

    Installs the intervention hooks around the scoring loop. This is essential
    and easy to miss: the backend attaches hooks inside ``generate()``, but this
    function calls the model directly for a single forward pass. Without the
    explicit install every candidate scores an identical zero delta -- which
    looks exactly like "the feature has no effect" rather than "the patch was
    never applied".

    ``apply_to_prompt`` is forced on because log-probability scoring is one
    forward pass over prompt and continuation together; restricting the
    intervention to "generated" positions would leave it inert here.
    """
    model = backend.model
    tokenizer = backend.tokenizer
    device = backend.device

    backend._apply_to_prompt = True
    backend._install_hooks()
    expected_hooks = len(backend._hooked_layers())
    if expected_hooks and not backend._handles:
        raise RuntimeError(
            "patches are installed but no forward hooks attached; scoring would "
            "silently measure the unpatched model"
        )
    try:
        return _score_loop(backend, examples, model, tokenizer, device, use_chat_template)
    finally:
        backend._remove_hooks()


def _score_loop(
    backend: Any,
    examples: Sequence[ContrastExample],
    model: Any,
    tokenizer: Any,
    device: Any,
    use_chat_template: bool,
) -> list[MarginResult]:
    results: list[MarginResult] = []
    for example in examples:
        if use_chat_template and getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": example.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = example.prompt

        independent, n_ind = sequence_logprob(
            model, tokenizer, prompt, example.positive_response, device
        )
        sycophantic, n_syc = sequence_logprob(
            model, tokenizer, prompt, example.negative_response, device
        )
        results.append(
            MarginResult(
                prompt=example.prompt,
                independent_logprob=independent,
                sycophantic_logprob=sycophantic,
                independent_tokens=n_ind,
                sycophantic_tokens=n_syc,
                topic=str(example.metadata.get("topic", "")),
            )
        )
    return results


def summarize(results: Sequence[MarginResult]) -> dict[str, float]:
    """Mean, median and win rate of the normalized margin."""
    values = [r.normalized_margin for r in results]
    if not values:
        return {}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "n": len(values),
        "mean_normalized_margin": sum(values) / len(values),
        "median_normalized_margin": median,
        "win_rate": sum(1 for v in values if v > 0) / len(values),
    }


def bootstrap_ci(
    deltas: Sequence[float], *, iterations: int = 5000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    """Percentile bootstrap CI for the mean of paired deltas.

    Paired because every delta is (patched - baseline) on the *same* example, so
    example difficulty cancels and the remaining variance is the effect itself.
    """
    import random

    if not deltas:
        return {}
    rng = random.Random(seed)
    n = len(deltas)
    means: list[float] = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * iterations)]
    hi = means[int((1 - alpha / 2) * iterations) - 1]
    observed = sum(deltas) / n
    return {
        "mean_delta": observed,
        "ci_low": lo,
        "ci_high": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "iterations": iterations,
    }


@dataclass
class CandidateFeature:
    """An SAE feature ranked by how it separates the two continuation classes."""

    feature_id: int
    effect_size: float
    mean_independent: float
    mean_sycophantic: float
    fire_rate: float
    firing_rate_corpus: float = 0.0
    max_activation_corpus: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p99: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "effect_size": self.effect_size,
            "mean_independent": self.mean_independent,
            "mean_sycophantic": self.mean_sycophantic,
            "fire_rate_in_contrast": self.fire_rate,
            "firing_rate_corpus": self.firing_rate_corpus,
            "max_activation_corpus": self.max_activation_corpus,
            "activation_percentiles": {"p50": self.p50, "p90": self.p90, "p99": self.p99},
            "evidence": "correlational -- activation difference only, no causal test",
        }


def screen_candidates(
    candidates: Sequence[CandidateFeature],
    directions: dict[int, torch.Tensor],
    *,
    max_cosine: float = 0.6,
    limit: int = 8,
) -> list[CandidateFeature]:
    """Deduplicate candidates by decoder-direction cosine similarity.

    The smoke experiment's control was a near-duplicate of its target because
    nothing screened for this. Two directions with cosine 0.9 are the same
    intervention wearing different feature IDs, and keeping both wastes budget
    while making the results look more independent than they are.
    """
    kept: list[CandidateFeature] = []
    for candidate in candidates:
        vector = directions[candidate.feature_id]
        unit = vector / vector.norm().clamp_min(1e-8)
        if any(
            abs(float(torch.dot(unit, directions[k.feature_id] / directions[k.feature_id].norm().clamp_min(1e-8))))
            > max_cosine
            for k in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) >= limit:
            break
    return kept


def pick_unrelated_controls(
    directions: dict[int, torch.Tensor],
    target_ids: Sequence[int],
    candidate_pool: Sequence[int],
    *,
    count: int = 3,
    max_cosine: float = 0.15,
    seed: int = 0,
) -> list[int]:
    """Choose control features that are *genuinely* unrelated to the targets.

    Requires low cosine similarity against every target and against each other.
    This is the check whose absence invalidated the previous experiment's
    unrelated-feature control.
    """
    import random

    rng = random.Random(seed)
    pool = list(candidate_pool)
    rng.shuffle(pool)

    def unit(idx: int) -> torch.Tensor:
        v = directions[idx]
        return v / v.norm().clamp_min(1e-8)

    targets = [unit(i) for i in target_ids]
    chosen: list[int] = []
    for feature_id in pool:
        if feature_id in target_ids:
            continue
        candidate = unit(feature_id)
        if any(abs(float(torch.dot(candidate, t))) > max_cosine for t in targets):
            continue
        if any(abs(float(torch.dot(candidate, unit(c)))) > max_cosine for c in chosen):
            continue
        chosen.append(feature_id)
        if len(chosen) >= count:
            break
    return chosen


def random_directions(hidden: int, count: int, *, seed: int = 1234) -> list[torch.Tensor]:
    """Scale-matched random unit directions for control conditions."""
    generator = torch.Generator().manual_seed(seed)
    out: list[torch.Tensor] = []
    for _ in range(count):
        vector = torch.randn(hidden, generator=generator)
        out.append(vector / vector.norm())
    return out
