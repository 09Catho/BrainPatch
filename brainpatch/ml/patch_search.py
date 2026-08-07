"""Candidate-feature search from behavioural contrast data.

Pipeline::

    contrast pairs -> activations for positive/negative responses
                   -> SAE encode
                   -> per-feature mean activation difference
                   -> ranked candidates              (correlational)
                   -> causal test with controls      (interventional)
                   -> strength sweep
                   -> greedy sparse combination
                   -> held-out evaluation

Only the first three steps are cheap. Everything after that costs GPU time per
candidate, which is why :func:`rank_candidate_features` returns a *small*
shortlist and the causal stages take an explicit budget. A search that fans out
over hundreds of candidates is how a $10 budget disappears.

The ranking step produces correlational evidence only. A feature that activates
more on "independent criticism" continuations than on "agreeable" ones is a
feature that *tracks* that distinction in this fixture -- it is not an
anti-sycophancy control knob until an intervention says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from brainpatch.ml.generation import build_chat_prompt
from brainpatch.ml.hooks import ResidualCapture
from brainpatch.ml.model import ModelBundle
from brainpatch.ml.sae import TopKSAE
from brainpatch.schemas.contrast import ContrastSet


@dataclass
class CandidateFeature:
    """A feature ranked by its activation difference across a contrast set."""

    feature_id: int
    mean_diff: float
    """Mean(positive activation) - Mean(negative activation), normalized space."""
    mean_positive: float
    mean_negative: float
    positive_fire_rate: float
    negative_fire_rate: float
    effect_size: float
    """Standardised difference (Cohen's d style), using pooled per-token std."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "mean_diff": self.mean_diff,
            "mean_positive": self.mean_positive,
            "mean_negative": self.mean_negative,
            "positive_fire_rate": self.positive_fire_rate,
            "negative_fire_rate": self.negative_fire_rate,
            "effect_size": self.effect_size,
            "evidence": "correlational -- activation difference only, no causal test",
        }


@torch.inference_mode()
def collect_response_activations(
    bundle: ModelBundle,
    texts: Sequence[str],
    *,
    layer: int,
    prompts: Sequence[str] | None = None,
    max_length: int = 384,
) -> torch.Tensor:
    """Capture residual activations over the *response* tokens only.

    When ``prompts`` is supplied, activations from the prompt portion are
    excluded. This matters: the prompt is identical across a contrast pair, so
    including its activations would dilute the signal with tokens that carry no
    information about the contrast.
    """
    capture = ResidualCapture(to_cpu=True, dtype=torch.float32)
    handle = capture.attach(bundle.layer_module(layer))
    collected: list[torch.Tensor] = []
    try:
        for i, text in enumerate(texts):
            prompt = prompts[i] if prompts is not None else None
            full = (prompt + text) if prompt else text
            encoded = bundle.tokenizer(
                full, return_tensors="pt", truncation=True, max_length=max_length
            ).to(bundle.device)
            capture.activations = None
            bundle.model(**encoded, use_cache=False)
            hidden = capture.activations
            if hidden is None:
                raise RuntimeError("capture hook did not fire during contrast collection")

            start = 0
            if prompt:
                prompt_len = bundle.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=max_length
                ).input_ids.shape[1]
                start = min(prompt_len, hidden.shape[1] - 1)
            collected.append(hidden[0, start:, :])
    finally:
        handle.remove()
    return torch.cat(collected, dim=0)


def rank_candidate_features(
    bundle: ModelBundle,
    sae: TopKSAE,
    contrast_set: ContrastSet,
    *,
    layer: int,
    input_scale: float,
    top_n: int = 20,
    use_chat_template: bool = True,
    min_fire_rate: float = 0.01,
    max_fire_rate: float = 0.9,
) -> list[CandidateFeature]:
    """Rank features by how differently they fire on positive vs negative responses.

    Parameters
    ----------
    min_fire_rate, max_fire_rate:
        Discard features that essentially never fire (no signal) or fire almost
        everywhere (modelling something global rather than the contrast).

    Returns
    -------
    list[CandidateFeature]
        Sorted by absolute effect size, longest-first, truncated to ``top_n``.
        **Correlational evidence only.**
    """
    prompts = [
        build_chat_prompt(bundle.tokenizer, e.prompt) if use_chat_template else e.prompt
        for e in contrast_set
    ]
    positives = [e.positive_response for e in contrast_set]
    negatives = [e.negative_response for e in contrast_set]

    pos_acts = collect_response_activations(bundle, positives, layer=layer, prompts=prompts)
    neg_acts = collect_response_activations(bundle, negatives, layer=layer, prompts=prompts)

    sae.eval()
    device = next(sae.parameters()).device
    pos_features = _encode_in_batches(sae, pos_acts, input_scale, device)
    neg_features = _encode_in_batches(sae, neg_acts, input_scale, device)

    pos_mean = pos_features.mean(dim=0)
    neg_mean = neg_features.mean(dim=0)
    pos_rate = (pos_features > 0).float().mean(dim=0)
    neg_rate = (neg_features > 0).float().mean(dim=0)

    pooled_std = torch.sqrt(
        (pos_features.var(dim=0, unbiased=False) + neg_features.var(dim=0, unbiased=False)) / 2
    ).clamp_min(1e-6)
    diff = pos_mean - neg_mean
    effect = diff / pooled_std

    overall_rate = (pos_rate + neg_rate) / 2
    eligible = (overall_rate >= min_fire_rate) & (overall_rate <= max_fire_rate)
    effect = torch.where(eligible, effect, torch.zeros_like(effect))

    order = torch.argsort(effect.abs(), descending=True)[:top_n]
    return [
        CandidateFeature(
            feature_id=int(fid),
            mean_diff=float(diff[fid].item()),
            mean_positive=float(pos_mean[fid].item()),
            mean_negative=float(neg_mean[fid].item()),
            positive_fire_rate=float(pos_rate[fid].item()),
            negative_fire_rate=float(neg_rate[fid].item()),
            effect_size=float(effect[fid].item()),
        )
        for fid in order.tolist()
        if effect[fid] != 0
    ]


def _encode_in_batches(
    sae: TopKSAE, acts: torch.Tensor, input_scale: float, device: torch.device, batch: int = 4096
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, acts.shape[0], batch):
        x = acts[start : start + batch].to(device=device, dtype=torch.float32) * input_scale
        sparse, _, _ = sae.encode(x)
        chunks.append(sparse.cpu())
    return torch.cat(chunks, dim=0)


@dataclass
class SweepPoint:
    """One point of a strength sweep."""

    strength: float
    mean_divergence: float
    mean_words: float
    degeneration_rate: float
    generations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strength": self.strength,
            "mean_divergence": self.mean_divergence,
            "mean_words": self.mean_words,
            "degeneration_rate": self.degeneration_rate,
        }


def strength_sweep(
    model: Any,
    prompts: Sequence[str],
    *,
    feature_id: int,
    strengths: Sequence[float],
    generation: Any = None,
) -> list[SweepPoint]:
    """Sweep one feature's strength and measure divergence and degeneration.

    The purpose is to find the window where an intervention changes behaviour
    *before* it destroys fluency. Every sweep point costs
    ``len(prompts)`` generations, so keep both lists short.
    """
    from brainpatch.evaluation.metrics import jaccard_similarity, score_generation
    from brainpatch.ml.causal import _spec_for
    from brainpatch.ml.generation import GenerationConfig

    cfg = generation or GenerationConfig(max_new_tokens=96)
    saved = dict(model.plan.patches)

    model.plan.patches = {}
    baselines = [model.generate(p, config=cfg) for p in prompts]

    points: list[SweepPoint] = []
    for strength in strengths:
        divergences: list[float] = []
        words: list[int] = []
        degenerate = 0
        texts: list[str] = []
        for prompt, baseline in zip(prompts, baselines):
            model.plan.patches = {}
            model.install(_spec_for(model, feature_id, strength, f"sweep-{feature_id}"))
            text = model.generate(prompt, config=cfg)
            metrics = score_generation(text)
            divergences.append(1.0 - jaccard_similarity(baseline, text, n=3))
            words.append(metrics.num_words)
            degenerate += int(metrics.degeneration_flag)
            texts.append(text)
        points.append(
            SweepPoint(
                strength=float(strength),
                mean_divergence=sum(divergences) / len(divergences),
                mean_words=sum(words) / len(words),
                degeneration_rate=degenerate / len(prompts),
                generations=texts,
            )
        )

    model.plan.patches = saved
    return points


def greedy_feature_selection(
    model: Any,
    prompts: Sequence[str],
    candidates: Sequence[CandidateFeature],
    *,
    strength: float,
    max_features: int = 3,
    generation: Any = None,
    degeneration_ceiling: float = 0.25,
) -> list[tuple[int, float]]:
    """Greedily build a sparse multi-feature patch.

    At each round, every remaining candidate is added tentatively and the one
    producing the largest divergence-from-baseline *without* pushing the
    degeneration rate above ``degeneration_ceiling`` is kept. Selection stops
    when no candidate improves the objective.

    Cost is ``O(max_features * len(candidates) * len(prompts))`` generations.
    Keep the candidate list to single digits.
    """
    from brainpatch.evaluation.metrics import jaccard_similarity, score_generation
    from brainpatch.ml.causal import _spec_for
    from brainpatch.ml.generation import GenerationConfig
    from brainpatch.schemas.patch import BrainPatchSpec, FeatureEdit

    cfg = generation or GenerationConfig(max_new_tokens=96)
    saved = dict(model.plan.patches)

    model.plan.patches = {}
    baselines = [model.generate(p, config=cfg) for p in prompts]

    selected: list[tuple[int, float]] = []
    remaining = [c.feature_id for c in candidates]
    best_score = 0.0

    def evaluate(edits: list[tuple[int, float]]) -> tuple[float, float]:
        base_spec = _spec_for(model, edits[0][0], edits[0][1], "greedy")
        spec = BrainPatchSpec(
            name="greedy",
            base_model=base_spec.base_model,
            model_revision=base_spec.model_revision,
            sae=base_spec.sae,
            features=[FeatureEdit(feature_id=f, strength=s) for f, s in edits],
            description="Transient greedy-search candidate.",
        )
        divergences: list[float] = []
        degenerate = 0
        for prompt, baseline in zip(prompts, baselines):
            model.plan.patches = {}
            model.install(spec)
            text = model.generate(prompt, config=cfg)
            divergences.append(1.0 - jaccard_similarity(baseline, text, n=3))
            degenerate += int(score_generation(text).degeneration_flag)
        return sum(divergences) / len(divergences), degenerate / len(prompts)

    for _ in range(max_features):
        best_feature: int | None = None
        best_round = best_score
        for feature_id in list(remaining):
            score, degeneration = evaluate([*selected, (feature_id, strength)])
            if degeneration > degeneration_ceiling:
                continue
            if score > best_round:
                best_round = score
                best_feature = feature_id
        if best_feature is None:
            break
        selected.append((best_feature, strength))
        remaining.remove(best_feature)
        best_score = best_round

    model.plan.patches = saved
    return selected
