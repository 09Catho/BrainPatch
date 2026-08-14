"""`anti_sycophancy_v3`: select by generated behaviour, not by log-probability.

v2 measured ``corr(log-prob steering effect, generation correction gain) =
-0.298``. Ranking directions by continuation log-probability *anti-selects* for
the behaviour we want, which is why v1 shipped a control-beating log-prob effect
whose free-generation correction rate fell. v3 therefore optimises the thing we
actually care about and demotes log-probability to a diagnostic.

Stages, deliberately separate so the test split cannot be reached early:

``stage_a_baseline``
    Length audit, baseline free generation and baseline log-prob margins on
    train and validation. Test is never loaded. Its output calibrates the
    minimum meaningful effect size before the criteria are written.

``stage_b_discovery``
    A funnel. Cheap representation and log-prob metrics *filter* candidates but
    never rank the finalists; free generation on a validation subset ranks them;
    the top few are re-scored on the full validation split.

``stage_c_test``
    Reads the frozen configuration, scores the test split once with every
    pre-registered control and the utility battery.

Decoding is deterministic everywhere: greedy, fixed token budget, fixed chat
template, identical prompts between baseline and patched runs. Every generated
response is stored.
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.image import ML_IMAGE
from modal_app.resources import VOL_MOUNT, app, gpu_kwargs, volume

RESEARCH_IMAGE = ML_IMAGE.add_local_dir("examples/contrast", remote_path="/root/examples/contrast")

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATASET = "antisycophancy_v3"
SAE_EXPERIMENT = "smoke_v0"
RESULTS_DIR = "experiments/anti_sycophancy_v3"

#: Narrow scan. v1 and v2 both found the useful region in the upper-middle of
#: the stack; 18 is retained because the SAE lives there and a like-for-like
#: comparison needs it. Still scanned rather than hard-coded to v2's answer.
CANDIDATE_LAYERS: tuple[int, ...] = (16, 18, 20, 22, 24)

#: v1 and v2 both measured prompt-token injection as far stronger than
#: generated-token steering (v2: +0.355 against -0.0002). Generated-only is not
#: scanned; the two prompt-side variants are compared.
PRIMARY_SITE = "prompt"
COMPARISON_SITES: tuple[str, ...] = ("prompt", "all")

EXTRACTION_POSITIONS: tuple[str, ...] = ("last_prompt", "cont_mean")

STRENGTH_RATIOS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35)

#: Hard cap on candidates that reach free-generation evaluation, declared before
#: running. Generation trials are the expensive and overfittable resource, so the
#: number of them is bounded up front rather than discovered.
MAX_GENERATION_CANDIDATES = 30
#: Finalists re-scored on the full validation split.
N_FINALISTS = 5
#: Subset size for the first generation pass. Large enough to rank, small
#: enough that 30 candidates stay affordable.
VALIDATION_SUBSET = 60

GENERATION_KWARGS = {"max_new_tokens": 96, "do_sample": False}

#: Token-level audit thresholds, same shape as v2's (Amendment 1 there).
MAX_TOKEN_MEAN_GAP_RATIO = 0.05
MAX_TOKEN_MEDIAN_GAP_TOKENS = 1.0
MAX_TOKEN_LABEL_CORR = 0.15
MIN_TOKEN_LONGER_SHARE = 0.45
MAX_TOKEN_LONGER_SHARE = 0.55


def _results_path(name: str) -> Any:
    from pathlib import Path

    directory = Path(VOL_MOUNT) / RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def _load(split_names: tuple[str, ...]) -> Any:
    from brainpatch.datasets import load_contrast_set
    from brainpatch.research.antisycophancy import split_by_topic

    splits = split_by_topic(load_contrast_set(DATASET))
    missing = [s for s in split_names if s not in splits]
    if missing:
        raise RuntimeError(f"dataset is missing splits {missing}")
    return {name: splits[name] for name in split_names}


def _token_length_audit(pairs: Any) -> dict[str, Any]:
    """Length audit on real tokens; runs before any activation is captured."""
    import statistics

    desired = [p.n_desired for p in pairs]
    undesired = [p.n_undesired for p in pairs]
    gaps = [d - u for d, u in zip(desired, undesired)]
    n = len(gaps)
    mean_gap = sum(gaps) / n
    mean_len = (sum(desired) + sum(undesired)) / (2 * n)

    values = desired + undesired
    mv = sum(values) / len(values)
    classes = [1.0] * n + [0.0] * n
    cov = sum((v - mv) * (c - 0.5) for v, c in zip(values, classes))
    den = (sum((v - mv) ** 2 for v in values) ** 0.5) * (
        sum((c - 0.5) ** 2 for c in classes) ** 0.5
    )

    stats = {
        "n_pairs": n,
        "mean_gap_tokens": mean_gap,
        "median_gap_tokens": statistics.median(gaps),
        "mean_gap_ratio": mean_gap / mean_len if mean_len else 0.0,
        "label_length_corr": cov / den if den else 0.0,
        "desired_longer_share": sum(1 for g in gaps if g > 0) / n,
    }
    failures = []
    if abs(stats["mean_gap_ratio"]) > MAX_TOKEN_MEAN_GAP_RATIO:
        failures.append(f"mean gap ratio {stats['mean_gap_ratio']:+.4f}")
    if abs(stats["median_gap_tokens"]) > MAX_TOKEN_MEDIAN_GAP_TOKENS:
        failures.append(f"median gap {stats['median_gap_tokens']:+.1f} tokens")
    if abs(stats["label_length_corr"]) > MAX_TOKEN_LABEL_CORR:
        failures.append(f"label/length corr {stats['label_length_corr']:+.4f}")
    if not MIN_TOKEN_LONGER_SHARE <= stats["desired_longer_share"] <= MAX_TOKEN_LONGER_SHARE:
        failures.append(f"desired-longer share {stats['desired_longer_share']:.3f}")
    stats["failures"] = failures
    stats["ok"] = not failures
    return stats


def _generate(
    model: Any,
    tokenizer: Any,
    examples: Any,
    *,
    pad_id: int,
    vector: Any = None,
    strength: float = 0.0,
    site: str = PRIMARY_SITE,
    layer_module: Any = None,
    batch_size: int = 12,
) -> list[str]:
    """Deterministic greedy generation, optionally steered.

    Prompts, template, token budget and decoding settings are identical between
    baseline and patched runs; the only difference is whether a hook is
    attached. That is what makes the two conditions comparable.
    """
    import torch

    from brainpatch.research.behaviour_eval import GenerationInjector

    injector = None
    if vector is not None:
        injector = GenerationInjector(vector.cuda(), strength, site).attach(layer_module)

    outputs: list[str] = []
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": e.prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for e in chunk
            ]
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to("cuda")
            if injector is not None:
                injector.reset()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, pad_token_id=pad_id, **GENERATION_KWARGS
                )
            for row in range(len(chunk)):
                new_tokens = generated[row, encoded["input_ids"].shape[1] :]
                outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = original_side
        if injector is not None:
            injector.remove()
    return outputs


def _behaviour(examples: Any, texts: list[str]) -> dict[str, Any]:
    from brainpatch.research.generation_eval import summarise

    polarities = [str(e.metadata.get("polarity", "false_claim")) for e in examples]
    return summarise(polarities, texts)


def _bootstrap_rate_ci(
    flags: list[int], *, resamples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    import torch

    if not flags:
        return (float("nan"), float("nan"))
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.tensor(flags, dtype=torch.float64)
    idx = torch.randint(0, len(flags), (resamples, len(flags)), generator=generator)
    means = tensor[idx].mean(dim=1)
    return (
        float(torch.quantile(means, 0.025)),
        float(torch.quantile(means, 0.975)),
    )


@app.function(**gpu_kwargs(timeout=60 * 50, image=RESEARCH_IMAGE))
def stage_a_baseline(batch_size: int = 12) -> dict[str, Any]:
    """Baseline behaviour on train and validation. Test is never loaded.

    This runs *before* the success criteria are written, because the minimum
    meaningful effect size has to be calibrated against how much the baseline
    rate itself moves between splits.
    """
    from brainpatch.backends.transformers_backend import TransformersBackend
    from brainpatch.research.behaviour_eval import encode_pairs, score_pairs
    from brainpatch.research.generation_eval import per_item_labels

    backend = TransformersBackend()
    backend.load_model(MODEL, revision=REVISION, device="cuda")
    model, tokenizer = backend.model, backend.tokenizer
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    splits = _load(("train", "validation"))
    encoded = {k: encode_pairs(tokenizer, v) for k, v in splits.items()}
    print(f"[stage-a] train={len(encoded['train'])} validation={len(encoded['validation'])}")
    print("[stage-a] test split deliberately not loaded")

    audits = {k: _token_length_audit(v) for k, v in encoded.items()}
    combined = _token_length_audit(encoded["train"] + encoded["validation"])
    print("[stage-a] TOKEN-LEVEL LENGTH AUDIT")
    for name, stats in list(audits.items()) + [("combined", combined)]:
        print(
            f"  {name:<11} mean={stats['mean_gap_tokens']:+.2f}tok "
            f"({stats['mean_gap_ratio']:+.4f}) median={stats['median_gap_tokens']:+.1f} "
            f"corr={stats['label_length_corr']:+.4f} "
            f"longer={stats['desired_longer_share']:.3f} "
            f"{'OK' if stats['ok'] else 'FAIL ' + '; '.join(stats['failures'])}"
        )
    if not combined["ok"]:
        raise RuntimeError("token-level length audit FAILED: " + "; ".join(combined["failures"]))

    baseline: dict[str, Any] = {}
    stored: dict[str, Any] = {}
    for name, pairs in encoded.items():
        texts = _generate(model, tokenizer, splits[name], pad_id=pad_id, batch_size=batch_size)
        behaviour = _behaviour(splits[name], texts)

        polarities = [str(e.metadata.get("polarity", "false_claim")) for e in splits[name]]
        labels = per_item_labels(polarities, texts)
        corrected = [
            1 if row["label"] == "CORRECT_CHALLENGE" else 0
            for row in labels
            if row["polarity"] == "false_claim"
        ]
        ci = _bootstrap_rate_ci(corrected)
        behaviour["correction_rate_ci"] = list(ci)

        scores = score_pairs(model, pairs, pad_id=pad_id, device="cuda", batch_size=batch_size)
        behaviour["logprob_diagnostic"] = {
            polarity: {
                "n": sum(1 for s in scores if s.polarity == polarity),
                "mean_normalized_margin": sum(
                    s.margin for s in scores if s.polarity == polarity
                ) / max(1, sum(1 for s in scores if s.polarity == polarity)),
                "mean_total_margin": sum(
                    s.total_margin for s in scores if s.polarity == polarity
                ) / max(1, sum(1 for s in scores if s.polarity == polarity)),
            }
            for polarity in ("false_claim", "true_claim")
        }
        baseline[name] = behaviour
        stored[name] = [
            {"topic": e.metadata.get("topic"), "polarity": p, "label": row["label"], "text": t}
            for e, p, t, row in zip(splits[name], polarities, texts, labels)
        ]

        print(
            f"[stage-a] {name}: correction={behaviour['correction_rate_false_claims']:.3f} "
            f"CI[{ci[0]:.3f},{ci[1]:.3f}] "
            f"sycophantic={behaviour['sycophantic_agreement_rate_false_claims']:.3f} "
            f"hedge={behaviour['hedge_rate_false_claims']:.3f} "
            f"other={behaviour['other_rate_false_claims']:.3f}"
        )
        print(
            f"[stage-a] {name}: true correct_agree="
            f"{behaviour['correct_agreement_rate_true_claims']:.3f} "
            f"false_disagree={behaviour['false_disagreement_rate_true_claims']:.3f} "
            f"SIS={behaviour['selective_independence_score']:+.3f}"
        )
        print(
            f"[stage-a] {name}: evaluator agreement="
            f"{behaviour['evaluator_agreement_rate']:.3f} "
            f"({behaviour['n_evaluator_disagreements']} disagreements) "
            f"degenerate={behaviour['degenerate_rate']:.3f} "
            f"mean_chars={behaviour['mean_response_chars']:.0f}"
        )

    payload = {
        "model": MODEL,
        "revision": REVISION,
        "dataset": DATASET,
        "generation_settings": {**GENERATION_KWARGS, "temperature": 0.0, "greedy": True},
        "token_length_audit": {**audits, "combined": combined},
        "baseline": baseline,
        "responses": stored,
    }
    path = _results_path("baseline_results.json")
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    volume.commit()
    print(f"[stage-a] wrote {path}")
    return {
        "audit_ok": combined["ok"],
        "train_correction": baseline["train"]["correction_rate_false_claims"],
        "validation_correction": baseline["validation"]["correction_rate_false_claims"],
        "validation_ci": baseline["validation"]["correction_rate_ci"],
    }
