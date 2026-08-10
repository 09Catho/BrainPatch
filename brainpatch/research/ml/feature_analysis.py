"""Building the feature database from a trained SAE.

For every dictionary feature this computes firing statistics over the
activation corpus and recovers the token contexts that drive it hardest.

What this module deliberately does **not** do is assign a semantic label. Top
activating contexts are correlational evidence. A feature whose top examples are
all hedging language is a feature that *correlates with* hedging language in
this corpus -- it is not "the uncertainty feature" until steering it changes
behaviour and scale-matched controls do not. Every record leaves
``hypothesis=None`` and ``evidence_level="none"``; the causal-validation
pipeline is the only thing that writes anything stronger.

Runs on CPU. A 2048-feature dictionary over 20k activations is a couple of
matrix multiplies, and CPU Modal Functions cost a fraction of GPU ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from brainpatch.research.ml.activation_store import ActivationSubset, read_manifest
from brainpatch.research.ml.sae import TopKSAE
from brainpatch.paths import VolumePaths
from brainpatch.schemas.feature import FeatureContext, FeatureRecord, FeatureStats


def load_examples(paths: VolumePaths, experiment: str) -> dict[int, dict[str, Any]]:
    """Load ``examples.jsonl`` into an index -> row mapping."""
    path = Path(paths.activation_examples(experiment))
    if not path.is_file():
        raise FileNotFoundError(f"examples file not found: {path}")
    examples: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples[int(row["index"])] = row
    return examples


@torch.no_grad()
def compute_feature_activations(
    sae: TopKSAE,
    subset: ActivationSubset,
    *,
    input_scale: float,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Encode the whole corpus into a ``[tokens, d_sae]`` sparse activation matrix.

    Kept dense in float32 for simplicity: at smoke scale that is
    ``20k x 2048 x 4B = 164 MB``. For a serious run this should stream and
    accumulate statistics incrementally instead; the ``max_bytes`` guard on
    :class:`ActivationSubset` is what stops that limit being crossed silently.
    """
    sae.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(subset), batch_size):
        batch = subset.activations[start : start + batch_size].to(torch.float32) * input_scale
        sparse, _, _ = sae.encode(batch)
        chunks.append(sparse.cpu())
    return torch.cat(chunks, dim=0)


def build_feature_database(
    sae: TopKSAE,
    subset: ActivationSubset,
    paths: VolumePaths,
    experiment: str,
    *,
    input_scale: float,
    top_k_contexts: int = 8,
    context_window: int = 12,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Compute per-feature statistics and top contexts, and persist them.

    Returns a summary dict; the per-feature records go to
    ``/vol/feature-db/<experiment>/features.jsonl``.
    """
    manifest = read_manifest(paths, experiment)
    examples = load_examples(paths, experiment)
    acts = compute_feature_activations(sae, subset, input_scale=input_scale)
    n_tokens, d_sae = acts.shape

    fire_mask = acts > 0
    fire_count = fire_mask.sum(dim=0)
    act_sum = acts.sum(dim=0)
    max_act = acts.max(dim=0).values
    decoder_norms = sae.decoder_norms().cpu()

    # Mean/std over *firing* tokens only: averaging in the structural zeros of a
    # Top-K SAE would just report k/d_sae times the true magnitude.
    mean_act = torch.where(fire_count > 0, act_sum / fire_count.clamp_min(1), torch.zeros_like(act_sum))
    sq_sum = (acts.pow(2)).sum(dim=0)
    var = torch.where(
        fire_count > 0,
        (sq_sum / fire_count.clamp_min(1)) - mean_act.pow(2),
        torch.zeros_like(act_sum),
    ).clamp_min(0.0)
    std_act = var.sqrt()

    out_dir = Path(paths.feature_db(experiment))
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = Path(paths.features_jsonl(experiment))

    alive = 0
    with features_path.open("w", encoding="utf-8") as handle:
        for feature_id in range(d_sae):
            count = int(fire_count[feature_id].item())
            stats = FeatureStats(
                fire_count=count,
                total_tokens=n_tokens,
                mean_activation=float(mean_act[feature_id].item()),
                max_activation=float(max_act[feature_id].item()),
                std_activation=float(std_act[feature_id].item()),
                decoder_norm=float(decoder_norms[feature_id].item()),
            )
            contexts: list[FeatureContext] = []
            if count > 0:
                alive += 1
                contexts = _top_contexts(
                    acts[:, feature_id],
                    subset.meta,
                    examples,
                    manifest.sequence_length,
                    top_k=top_k_contexts,
                    window=context_window,
                    tokenizer=tokenizer,
                )
            # hypothesis stays None and evidence_level stays "none" by design.
            record = FeatureRecord(feature_id=feature_id, stats=stats, top_contexts=contexts)
            handle.write(record.to_json() + "\n")

    firing_rates = (fire_count.float() / n_tokens).tolist()
    alive_rates = [r for r in firing_rates if r > 0]
    summary = {
        "experiment": experiment,
        "num_features": d_sae,
        "num_tokens_analysed": n_tokens,
        "alive_features": alive,
        "dead_features": d_sae - alive,
        "dead_fraction": (d_sae - alive) / d_sae,
        "mean_firing_rate_alive": (sum(alive_rates) / len(alive_rates)) if alive_rates else 0.0,
        "median_firing_rate_alive": _median(alive_rates),
        "max_firing_rate": max(firing_rates) if firing_rates else 0.0,
        "mean_l0": float(fire_mask.sum(dim=1).float().mean().item()),
        "decoder_norm_mean": float(decoder_norms.mean().item()),
        "decoder_norm_min": float(decoder_norms.min().item()),
        "decoder_norm_max": float(decoder_norms.max().item()),
        "input_scale": input_scale,
        "features_path": str(features_path),
        "note": (
            "Statistics only. No feature carries a semantic label: top-activating "
            "contexts are correlational evidence and are not sufficient to name a feature."
        ),
    }
    Path(paths.feature_summary(experiment)).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _top_contexts(
    feature_column: torch.Tensor,
    meta: torch.Tensor,
    examples: dict[int, dict[str, Any]],
    sequence_length: int,
    *,
    top_k: int,
    window: int,
    tokenizer: Any = None,
) -> list[FeatureContext]:
    """Recover the highest-activating token occurrences with surrounding text."""
    nonzero = (feature_column > 0).nonzero(as_tuple=True)[0]
    if nonzero.numel() == 0:
        return []
    k = min(top_k, nonzero.numel())
    values = feature_column[nonzero]
    order = torch.topk(values, k).indices
    rows = nonzero[order]

    contexts: list[FeatureContext] = []
    for row in rows.tolist():
        example_index = int(meta[row, 0].item())
        position = int(meta[row, 1].item())
        token_id = int(meta[row, 2].item())
        example = examples.get(example_index)

        token_text = ""
        before = after = ""
        if tokenizer is not None:
            token_text = tokenizer.decode([token_id])
            if example is not None:
                before, after = _decode_window(
                    tokenizer, example.get("text", ""), position, window
                )
        elif example is not None:
            before, after = "", example.get("text", "")[:200]

        contexts.append(
            FeatureContext(
                example_index=example_index,
                token_position=position,
                token_id=token_id,
                token_text=token_text,
                activation=float(feature_column[row].item()),
                context_before=before,
                context_after=after,
            )
        )
    return contexts


def _decode_window(tokenizer: Any, text: str, position: int, window: int) -> tuple[str, str]:
    """Re-tokenize the stored text and slice a window around ``position``.

    Re-tokenizing is cheap and avoids storing per-token strings for the whole
    corpus, which would dominate the on-disk size.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if position >= len(ids):
        return "", ""
    lo = max(0, position - window)
    hi = min(len(ids), position + window + 1)
    return tokenizer.decode(ids[lo:position]), tokenizer.decode(ids[position + 1 : hi])


def rank_features(
    paths: VolumePaths,
    experiment: str,
    *,
    by: str = "max_activation",
    limit: int = 50,
    min_fire_count: int = 1,
    max_firing_rate: float = 1.0,
    min_firing_rate: float = 0.0,
) -> list[FeatureRecord]:
    """Rank features from the persisted database.

    Parameters
    ----------
    by:
        ``max_activation``, ``mean_activation``, ``fire_count`` or ``firing_rate``.
    max_firing_rate:
        Drop features that fire on more than this fraction of tokens. Very
        high-frequency features are usually modelling something positional or
        distributional rather than anything specific.
    min_firing_rate:
        Drop features that fire on *fewer* than this fraction of tokens. See the
        warning below -- this is the guard that matters in practice.

    Warning
    -------
    **Ranking by ``max_activation`` alone selects outliers, and did so
    destructively in ``smoke_v0``.** Measured on that feature database: the top
    32 features by ``max_activation`` all fired on 3-6 tokens out of 20,000, all
    with the same top token (``" Bd"``, chess notation from a handful of
    wikitext articles), at activations 100x+ the dictionary median of 9.06. An
    undertrained SAE shatters rare high-norm tokens across many near-duplicate
    features, and this ranking finds precisely those.

    The consequence in ``smoke_v0`` was worse than a poor choice of target: the
    "unrelated feature" control was drawn from the same ranking and landed on
    feature 1270, a near-duplicate of the target firing on the same token. That
    control was therefore not unrelated and its comparison is uninformative.

    For intervention candidates, pass ``min_firing_rate`` at or near the
    dictionary median firing rate, or rank by ``mean_activation`` /
    ``fire_count`` instead.
    """
    path = Path(paths.features_jsonl(experiment))
    if not path.is_file():
        raise FileNotFoundError(f"feature database not found: {path}")

    records: list[FeatureRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = FeatureRecord.from_dict(json.loads(line))
            if record.stats.fire_count < min_fire_count:
                continue
            if record.stats.firing_rate > max_firing_rate:
                continue
            if record.stats.firing_rate < min_firing_rate:
                continue
            records.append(record)

    keys = {
        "max_activation": lambda r: r.stats.max_activation,
        "mean_activation": lambda r: r.stats.mean_activation,
        "fire_count": lambda r: r.stats.fire_count,
        "firing_rate": lambda r: r.stats.firing_rate,
    }
    if by not in keys:
        raise ValueError(f"unknown ranking key {by!r}; expected one of {sorted(keys)}")
    records.sort(key=keys[by], reverse=True)
    return records[:limit]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
