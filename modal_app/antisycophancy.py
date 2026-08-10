"""Staged anti-sycophancy patch search on the existing SAE.

Stage A of the plan: reuse the ``smoke_v0`` SAE and search it with a
behaviour-specific objective rather than max-activation ranking. Costs a few
minutes of L4 because log-probability scoring is one forward pass per
continuation and the continuations are short.

Method, and what each part is defending against
-----------------------------------------------
**Objective is the paired log-probability margin**, not trigram divergence. The
previous experiment's metric rewarded *any* perturbation, which is how a random
direction beat the real feature.

**Candidates are screened** on corpus firing rate (excluding the rare-token
cluster that poisoned the last run), on cosine similarity to each other, and on
contrast effect size.

**Strength is calibrated to the feature's own activation distribution** — the
p90 of what the feature naturally reaches — rather than an arbitrary large
number, so any effect is on-manifold.

**Splits are by topic.** Candidate discovery sees train only; strength selection
sees validation only; the test split is scored once, at the end.

**Controls are plural and screened**: three scale-matched random directions and
three SAE features verified to have low cosine similarity to the target. The
previous "unrelated" control was a near-duplicate; this one checks.
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.image import ML_IMAGE
from modal_app.resources import VOL_MOUNT, app, gpu_kwargs, volume

#: The contrast fixtures are JSON data files, not Python sources, so
#: add_local_python_source does not carry them. Mount them explicitly.
RESEARCH_IMAGE = ML_IMAGE.add_local_dir("examples/contrast", remote_path="/root/examples/contrast")

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
EXPERIMENT = "smoke_v0"

#: Screening thresholds. The corpus median firing rate is ~0.0136; anything an
#: order of magnitude below that is in the rare-token regime that produced the
#: previous failure.
MIN_CORPUS_FIRING_RATE = 0.002
MAX_CORPUS_FIRING_RATE = 0.30


@app.function(**gpu_kwargs(timeout=60 * 40, image=RESEARCH_IMAGE))
def antisycophancy_stage_a(
    top_n: int = 6,
    max_cosine: float = 0.6,
    seed: int = 0,
) -> dict[str, Any]:
    """Discover, tune and test anti-sycophancy candidates on the existing SAE."""
    import json as _json
    from pathlib import Path

    import torch

    from brainpatch.datasets import load_contrast_set
    from brainpatch.research.antisycophancy import (
        CandidateFeature,
        bootstrap_ci,
        pick_unrelated_controls,
        random_directions,
        score_examples,
        screen_candidates,
        split_by_topic,
        summarize,
    )
    from brainpatch.research.ml.sae import TopKSAE
    from brainpatch.backends.transformers_backend import TransformersBackend
    from brainpatch.paths import VolumePaths

    paths = VolumePaths(VOL_MOUNT)

    # ---- load model and SAE ---------------------------------------------------
    backend = TransformersBackend()
    backend.load_model(MODEL, revision=REVISION, device="cuda")
    descriptor = backend.describe_model()
    hidden = descriptor.hidden_size

    checkpoint = torch.load(
        str(paths.sae_checkpoint(EXPERIMENT)), map_location="cuda", weights_only=False
    )
    sae = TopKSAE.from_checkpoint(checkpoint, device="cuda")
    input_scale = float(sae.config.input_scale)
    layer = int(sae.config.layer)

    # ---- corpus statistics, for screening -------------------------------------
    corpus_stats: dict[int, dict[str, float]] = {}
    with open(paths.features_jsonl(EXPERIMENT), encoding="utf-8") as handle:
        for line in handle:
            row = _json.loads(line)
            corpus_stats[int(row["feature_id"])] = {
                "firing_rate": float(row["stats"]["firing_rate"]),
                "max_activation": float(row["stats"]["max_activation"]),
            }

    eligible = {
        fid
        for fid, s in corpus_stats.items()
        if MIN_CORPUS_FIRING_RATE <= s["firing_rate"] <= MAX_CORPUS_FIRING_RATE
    }
    print(f"[stage-a] {len(eligible)} of {len(corpus_stats)} features pass the firing-rate screen")

    contrast = load_contrast_set("antisycophancy_eval")
    splits = split_by_topic(contrast)
    print(
        "[stage-a] splits: "
        + ", ".join(f"{k}={len(v)}" for k, v in sorted(splits.items()))
    )

    tokenizer = backend.tokenizer

    def chat(prompt: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    # ---- collect activations over TRAIN continuations only ---------------------
    from brainpatch.research.ml.hooks import ResidualCapture

    def encode_continuations(examples: list, which: str) -> torch.Tensor:
        capture = ResidualCapture(to_cpu=False, dtype=torch.float32)
        handle = capture.attach(backend.model.model.layers[layer])
        chunks: list[torch.Tensor] = []
        try:
            for example in examples:
                prompt = chat(example.prompt)
                text = prompt + (
                    example.positive_response if which == "independent" else example.negative_response
                )
                prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
                with torch.inference_mode():
                    backend.model(**inputs, use_cache=False)
                acts = capture.activations
                # Continuation positions only: the prompt is identical across the
                # pair, so its activations carry no contrast information.
                chunks.append(acts[0, prompt_len:, :])
        finally:
            handle.remove()
        return torch.cat(chunks, dim=0)

    train = splits["train"]
    ind_acts = encode_continuations(train, "independent")
    syc_acts = encode_continuations(train, "sycophantic")
    print(f"[stage-a] train activations: independent {tuple(ind_acts.shape)}, sycophantic {tuple(syc_acts.shape)}")

    with torch.inference_mode():
        ind_feat, _, _ = sae.encode(ind_acts * input_scale)
        syc_feat, _, _ = sae.encode(syc_acts * input_scale)

    ind_mean = ind_feat.mean(dim=0)
    syc_mean = syc_feat.mean(dim=0)
    pooled = torch.sqrt(
        (ind_feat.var(dim=0, unbiased=False) + syc_feat.var(dim=0, unbiased=False)) / 2
    ).clamp_min(1e-6)
    effect = (ind_mean - syc_mean) / pooled

    mask = torch.zeros_like(effect, dtype=torch.bool)
    for fid in eligible:
        mask[fid] = True
    effect = torch.where(mask, effect, torch.zeros_like(effect))

    order = torch.argsort(effect.abs(), descending=True)[: top_n * 6]
    directions = {int(i): sae.feature_direction(int(i), normalize=True).cpu() for i in order.tolist()}

    all_feat = torch.cat([ind_feat, syc_feat], dim=0)
    raw: list[CandidateFeature] = []
    for i in order.tolist():
        fid = int(i)
        if effect[fid] == 0:
            continue
        column = all_feat[:, fid]
        positive = column[column > 0]
        raw.append(
            CandidateFeature(
                feature_id=fid,
                effect_size=float(effect[fid].item()),
                mean_independent=float(ind_mean[fid].item()),
                mean_sycophantic=float(syc_mean[fid].item()),
                fire_rate=float((column > 0).float().mean().item()),
                firing_rate_corpus=corpus_stats[fid]["firing_rate"],
                max_activation_corpus=corpus_stats[fid]["max_activation"],
                p50=float(positive.median().item()) if positive.numel() else 0.0,
                p90=float(torch.quantile(positive, 0.90).item()) if positive.numel() else 0.0,
                p99=float(torch.quantile(positive, 0.99).item()) if positive.numel() else 0.0,
            )
        )

    candidates = screen_candidates(raw, directions, max_cosine=max_cosine, limit=top_n)
    print(f"[stage-a] {len(candidates)} candidates after cosine screening")
    for c in candidates:
        print(
            f"    f{c.feature_id:<5} effect={c.effect_size:+.3f} "
            f"corpus_rate={c.firing_rate_corpus:.4f} p90={c.p90:.2f}"
        )

    if not candidates:
        return {"ok": True, "stage": "A", "candidates": [], "verdict": "no candidate passed screening"}

    # ---- baseline margins ------------------------------------------------------
    def install_direction(vector: torch.Tensor, coefficient: float) -> None:
        """Install a raw direction as a live patch, bypassing the file format."""
        backend._patches = {}
        backend._vector_cache = {}
        raw_vec = (vector / float(input_scale)).tolist()
        _install_raw(backend, raw_vec, layer, coefficient, hidden)

    def margins(examples: list) -> list:
        return score_examples(backend, examples)

    backend._patches = {}
    base_val = margins(splits["validation"])
    base_test = margins(splits["test"])
    print(f"[stage-a] baseline validation margin {summarize(base_val)['mean_normalized_margin']:+.4f}")

    # ---- validation sweep: candidate x strength -------------------------------
    sweep: list[dict[str, Any]] = []
    for candidate in candidates:
        direction = directions[candidate.feature_id]
        # Calibrate to the feature's own scale: p90 of what it naturally reaches.
        for factor in (0.5, 1.0, 2.0):
            coefficient = candidate.p90 * factor
            if coefficient <= 0:
                continue
            for sign in (1.0, -1.0):
                install_direction(direction, coefficient * sign)
                results = margins(splits["validation"])
                deltas = [
                    r.normalized_margin - b.normalized_margin for r, b in zip(results, base_val)
                ]
                sweep.append(
                    {
                        "feature_id": candidate.feature_id,
                        "coefficient": coefficient * sign,
                        "scale_factor": factor * sign,
                        "mean_delta": sum(deltas) / len(deltas),
                        "win_rate": sum(1 for d in deltas if d > 0) / len(deltas),
                        **summarize(results),
                    }
                )

    sweep.sort(key=lambda r: r["mean_delta"], reverse=True)
    print("[stage-a] top validation configurations:")
    for row in sweep[:5]:
        print(
            f"    f{row['feature_id']:<5} coef={row['coefficient']:+.2f} "
            f"delta={row['mean_delta']:+.4f} win={row['win_rate']:.2f}"
        )

    best = sweep[0] if sweep else None
    if best is None or best["mean_delta"] <= 0:
        return {
            "ok": True,
            "stage": "A",
            "candidates": [c.to_dict() for c in candidates],
            "validation_sweep": sweep[:12],
            "verdict": (
                "NEGATIVE: no candidate improved the validation margin. "
                "The existing wikitext SAE provides no usable anti-sycophancy direction."
            ),
            "held_out_test_run": False,
        }

    # ---- controls, chosen against the winning direction ------------------------
    target_id = int(best["feature_id"])
    target_dir = directions[target_id]
    pool = [f for f in eligible if f in corpus_stats]
    all_dirs = {**directions}
    import random as _random

    rng = _random.Random(seed)
    sample_pool = rng.sample(sorted(pool), min(400, len(pool)))
    for fid in sample_pool:
        if fid not in all_dirs:
            all_dirs[fid] = sae.feature_direction(fid, normalize=True).cpu()

    unrelated = pick_unrelated_controls(
        all_dirs, [target_id], sample_pool, count=3, max_cosine=0.15, seed=seed
    )
    randoms = random_directions(hidden, 3, seed=1234)
    print(f"[stage-a] controls: unrelated features {unrelated}, 3 random directions")

    # ---- HELD-OUT TEST, scored once -------------------------------------------
    coefficient = float(best["coefficient"])
    conditions: dict[str, Any] = {}

    install_direction(target_dir, coefficient)
    target_results = margins(splits["test"])
    target_deltas = [r.normalized_margin - b.normalized_margin for r, b in zip(target_results, base_test)]
    conditions["target"] = {
        "feature_id": target_id,
        "coefficient": coefficient,
        **summarize(target_results),
        **bootstrap_ci(target_deltas),
    }

    for i, fid in enumerate(unrelated):
        install_direction(all_dirs[fid], coefficient)
        res = margins(splits["test"])
        deltas = [r.normalized_margin - b.normalized_margin for r, b in zip(res, base_test)]
        conditions[f"unrelated_feature_{fid}"] = {
            "feature_id": fid,
            **summarize(res),
            "mean_delta": sum(deltas) / len(deltas),
        }

    for i, vector in enumerate(randoms):
        install_direction(vector, coefficient)
        res = margins(splits["test"])
        deltas = [r.normalized_margin - b.normalized_margin for r, b in zip(res, base_test)]
        conditions[f"random_{i}"] = {
            **summarize(res),
            "mean_delta": sum(deltas) / len(deltas),
        }

    backend._patches = {}
    conditions["baseline"] = summarize(base_test)

    control_deltas = [
        v["mean_delta"] for k, v in conditions.items() if k.startswith(("unrelated", "random"))
    ]
    target_delta = conditions["target"]["mean_delta"]
    beats_all_controls = all(target_delta > d for d in control_deltas)

    result = {
        "ok": True,
        "stage": "A",
        "model": MODEL,
        "sae": EXPERIMENT,
        "layer": layer,
        "splits": {k: len(v) for k, v in splits.items()},
        "features_screened": {"eligible": len(eligible), "total": len(corpus_stats)},
        "candidates": [c.to_dict() for c in candidates],
        "validation_sweep": sweep[:12],
        "selected": {"feature_id": target_id, "coefficient": coefficient},
        "held_out_test": conditions,
        "control_deltas": control_deltas,
        "target_delta": target_delta,
        "beats_all_controls": beats_all_controls,
        "ci_excludes_zero": conditions["target"].get("excludes_zero"),
        "held_out_test_run": True,
    }
    print(json.dumps(result, indent=2, default=str)[:8000])
    return result


def _install_raw(backend: Any, values: list, layer: int, coefficient: float, hidden: int) -> None:
    """Install a bare direction on the backend without going through a file.

    Stage A evaluates hundreds of candidate directions; writing each one to a
    ``.brainpatch`` archive first would be pure overhead. The runtime path is
    identical -- the same hook, the same resolve_edits arithmetic -- so this
    shortcut does not change what is being measured.
    """
    from brainpatch.patch.format import BaseModelSpec, Intervention, Manifest
    from brainpatch.patch.loader import LoadedPatch
    from brainpatch.patch import tensors as ts
    from brainpatch.runtime.base import ActivePatch

    manifest = Manifest(
        name="candidate",
        base_model=BaseModelSpec(
            model_id=MODEL,
            architecture="Qwen2ForCausalLM",
            hidden_size=hidden,
            num_layers=28,
            revision=REVISION,
        ),
        interventions=[Intervention(layer=layer, vector="v", coefficient=coefficient)],
        max_abs_strength=1024.0,
    )
    manifest.validate()
    loaded = LoadedPatch(manifest=manifest, vectors={"v": ts.vector(values, dtype="F32")})
    backend._patches = {"candidate": ActivePatch(patch=loaded, strength=1.0)}
    backend._on_patches_changed()
