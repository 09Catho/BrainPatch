"""Causal validation of feature interventions.

The question this module answers is *not* "does steering this feature change the
output" -- almost any large enough perturbation changes the output. It is
"does steering **this direction** change the output in a way that a
**scale-matched perturbation in another direction** does not".

Conditions run for every prompt:

======================  =========================================================
``baseline``            no hook installed at all
``zero``                hook installed, strength 0 -- must equal ``baseline``
``positive``            the feature direction at ``+strength``
``negative``            the feature direction at ``-strength``
``random_positive``     a random unit direction at ``+strength`` (same L2 norm)
``random_negative``     a random unit direction at ``-strength``
``unrelated_positive``  a *different, real* feature at ``+strength``
======================  =========================================================

The random control isolates "is this direction special". The unrelated-feature
control isolates "is this feature special, or does any dictionary direction do
this". The ``zero`` condition is a correctness check on the harness itself.

Every generation is stored. There is no filtering, no best-of, and no
cherry-picking: the artifacts contain what the model actually produced under
each condition, including the incoherent ones.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from brainpatch.evaluation.metrics import compare_generations, score_generation
from brainpatch.ml.generation import GenerationConfig
from brainpatch.ml.runtime import BrainPatchedModel
from brainpatch.paths import VolumePaths
from brainpatch.schemas.patch import BrainPatchSpec, FeatureEdit, SAEReference

#: Conditions that constitute the intervention itself.
INTERVENTION_CONDITIONS = ("positive", "negative")
#: Conditions that exist to rule out alternative explanations.
CONTROL_CONDITIONS = ("zero", "random_positive", "random_negative", "unrelated_positive")


@dataclass
class ConditionResult:
    """One (prompt, condition) generation with its model-free metrics."""

    condition: str
    prompt_index: int
    prompt: str
    text: str
    feature_id: int | None
    strength: float
    steering_stats: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "prompt_index": self.prompt_index,
            "prompt": self.prompt,
            "text": self.text,
            "feature_id": self.feature_id,
            "strength": self.strength,
            "steering_stats": self.steering_stats,
            "metrics": self.metrics,
        }


def _spec_for(
    model: BrainPatchedModel, feature_id: int, strength: float, name: str
) -> BrainPatchSpec:
    """Build a throwaway single-feature patch spec for one condition."""
    assert model.sae is not None and model.sae_reference is not None
    return BrainPatchSpec(
        name=name,
        base_model=model.bundle.model_id,
        model_revision=model.bundle.revision,
        sae=SAEReference(
            reference=model.sae_reference,
            layer=int(model.sae.config.layer),
            hook=model.sae.config.hook or "residual_post",
            d_in=model.sae.d_in,
            d_sae=model.sae.d_sae,
            input_scale=model.input_scale,
        ),
        features=[FeatureEdit(feature_id=feature_id, strength=strength)],
        description="Transient experimental condition; not a shipped patch.",
        evidence_level="none",
    )


def run_intervention_experiment(
    model: BrainPatchedModel,
    prompts: Sequence[str],
    *,
    feature_id: int,
    strength: float,
    unrelated_feature_id: int,
    generation: GenerationConfig | None = None,
    control_seed: int = 1234,
) -> list[ConditionResult]:
    """Run every condition over every prompt and return all generations.

    Generation settings are identical across conditions by construction: the
    same :class:`GenerationConfig` object is passed to each call.
    """
    cfg = generation or GenerationConfig()
    results: list[ConditionResult] = []

    def run(condition: str, prompt_index: int, prompt: str, **kwargs) -> ConditionResult:
        text = model.generate(prompt, config=cfg, **kwargs)
        return ConditionResult(
            condition=condition,
            prompt_index=prompt_index,
            prompt=prompt,
            text=text,
            feature_id=kwargs.get("_feature_id"),
            strength=kwargs.get("_strength", 0.0),
            steering_stats=model.last_steering_stats,
            metrics=score_generation(text).to_dict(),
        )

    for i, prompt in enumerate(prompts):
        # --- baseline: nothing installed --------------------------------------
        saved = dict(model.plan.patches)
        model.plan.patches = {}
        baseline_text = model.generate(prompt, config=cfg)
        results.append(
            ConditionResult(
                condition="baseline",
                prompt_index=i,
                prompt=prompt,
                text=baseline_text,
                feature_id=None,
                strength=0.0,
                steering_stats={},
                metrics=score_generation(baseline_text).to_dict(),
            )
        )
        model.plan.patches = saved

        conditions: list[tuple[str, int, float, str]] = [
            ("zero", feature_id, 0.0, "none"),
            ("positive", feature_id, strength, "none"),
            ("negative", feature_id, -strength, "none"),
            ("random_positive", feature_id, strength, "random"),
            ("random_negative", feature_id, -strength, "random"),
            ("unrelated_positive", unrelated_feature_id, strength, "none"),
        ]

        for condition, fid, magnitude, control in conditions:
            model.plan.patches = {}
            model.install(_spec_for(model, fid, magnitude, f"cond-{condition}"))
            text = model.generate(prompt, config=cfg, control=control, control_seed=control_seed)
            results.append(
                ConditionResult(
                    condition=condition,
                    prompt_index=i,
                    prompt=prompt,
                    text=text,
                    feature_id=fid,
                    strength=magnitude,
                    steering_stats=model.last_steering_stats,
                    metrics=score_generation(text).to_dict(),
                )
            )
            model.plan.patches = {}

        model.plan.patches = saved

    return results


def summarize_experiment(results: Sequence[ConditionResult]) -> dict[str, Any]:
    """Aggregate per-condition statistics and the zero-strength sanity check.

    ``divergence_from_baseline`` is ``1 - Jaccard(3-gram)`` between a condition's
    output and the baseline for the same prompt: 0 means identical text, 1 means
    no shared trigrams. Comparing an intervention's divergence against its
    scale-matched random control is the core of the causal claim.
    """
    by_prompt: dict[int, dict[str, ConditionResult]] = {}
    for r in results:
        by_prompt.setdefault(r.prompt_index, {})[r.condition] = r

    conditions = sorted({r.condition for r in results})
    summary: dict[str, Any] = {"num_prompts": len(by_prompt), "conditions": {}}

    zero_identical = 0
    zero_total = 0

    for condition in conditions:
        divergences: list[float] = []
        lengths: list[int] = []
        degenerations = 0
        distinct2: list[float] = []
        delta_norms: list[float] = []
        count = 0

        for prompt_results in by_prompt.values():
            result = prompt_results.get(condition)
            baseline = prompt_results.get("baseline")
            if result is None or baseline is None:
                continue
            count += 1
            comparison = compare_generations(baseline.text, result.text)
            divergences.append(1.0 - comparison["jaccard_3"])
            lengths.append(result.metrics["num_words"])
            distinct2.append(result.metrics["distinct_2"])
            if result.metrics["degeneration_flag"]:
                degenerations += 1
            if result.steering_stats:
                delta_norms.append(result.steering_stats.get("mean_delta_norm", 0.0))
            if condition == "zero":
                zero_total += 1
                zero_identical += int(comparison["identical"])

        summary["conditions"][condition] = {
            "n": count,
            "mean_divergence_from_baseline": _mean(divergences),
            "mean_num_words": _mean(lengths),
            "mean_distinct_2": _mean(distinct2),
            "degeneration_count": degenerations,
            "degeneration_rate": degenerations / count if count else 0.0,
            "mean_delta_norm": _mean(delta_norms) if delta_norms else None,
        }

    summary["zero_strength_matches_baseline"] = {
        "identical": zero_identical,
        "total": zero_total,
        "all_identical": zero_total > 0 and zero_identical == zero_total,
    }

    # The comparison that licenses (or refuses) a causal claim.
    def divergence(condition: str) -> float | None:
        entry = summary["conditions"].get(condition)
        return entry["mean_divergence_from_baseline"] if entry else None

    pos = divergence("positive")
    rand = divergence("random_positive")
    unrelated = divergence("unrelated_positive")
    summary["effect_vs_controls"] = {
        "positive_divergence": pos,
        "random_control_divergence": rand,
        "unrelated_feature_divergence": unrelated,
        "positive_minus_random": (pos - rand) if pos is not None and rand is not None else None,
        "positive_minus_unrelated": (
            (pos - unrelated) if pos is not None and unrelated is not None else None
        ),
        "interpretation_note": (
            "A positive difference means the feature direction moved the output "
            "further from baseline than a scale-matched control. It does NOT by "
            "itself identify what changed, or establish any semantic label. "
            "With a handful of prompts and no repeated sampling these differences "
            "carry no statistical significance."
        ),
    }
    return summary


def write_experiment_artifacts(
    paths: VolumePaths,
    experiment: str,
    config: dict[str, Any],
    results: Sequence[ConditionResult],
    summary: dict[str, Any],
) -> dict[str, str]:
    """Persist config, all generations, metrics and a markdown report."""
    out_dir = Path(paths.experiment(experiment))
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}

    def dump_jsonl(filename: str, rows: Sequence[ConditionResult]) -> None:
        path = out_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        written[filename] = str(path)

    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    written["config.json"] = str(out_dir / "config.json")

    dump_jsonl("baseline.jsonl", [r for r in results if r.condition == "baseline"])
    dump_jsonl("interventions.jsonl", [r for r in results if r.condition in INTERVENTION_CONDITIONS])
    dump_jsonl("controls.jsonl", [r for r in results if r.condition in CONTROL_CONDITIONS])
    dump_jsonl("all_generations.jsonl", list(results))

    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    written["metrics.json"] = str(out_dir / "metrics.json")

    report = render_report(experiment, config, summary, results)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    written["report.md"] = str(out_dir / "report.md")
    return written


def render_report(
    experiment: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    results: Sequence[ConditionResult],
) -> str:
    """Render a markdown report that states what was measured and nothing more."""
    lines: list[str] = [
        f"# Intervention experiment: `{experiment}`",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## Harness sanity check",
        "",
    ]
    zero = summary["zero_strength_matches_baseline"]
    status = "PASS" if zero["all_identical"] else "FAIL"
    lines += [
        f"`strength=0` reproduces baseline exactly: **{status}** "
        f"({zero['identical']}/{zero['total']} prompts identical).",
        "",
        "A failure here invalidates every other number on this page, because it "
        "would mean the hook perturbs the model even when asked to do nothing.",
        "",
        "## Per-condition summary",
        "",
        "| condition | n | divergence from baseline | mean words | distinct-2 | degenerate | mean delta norm |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition, entry in summary["conditions"].items():
        norm = entry["mean_delta_norm"]
        lines.append(
            f"| `{condition}` | {entry['n']} | {entry['mean_divergence_from_baseline']:.3f} | "
            f"{entry['mean_num_words']:.1f} | {entry['mean_distinct_2']:.3f} | "
            f"{entry['degeneration_count']} | {f'{norm:.3f}' if norm is not None else '-'} |"
        )

    effect = summary["effect_vs_controls"]
    lines += [
        "",
        "## Effect versus controls",
        "",
        f"- positive intervention divergence: `{_fmt(effect['positive_divergence'])}`",
        f"- scale-matched random direction:   `{_fmt(effect['random_control_divergence'])}`",
        f"- unrelated real feature:           `{_fmt(effect['unrelated_feature_divergence'])}`",
        f"- positive minus random:            `{_fmt(effect['positive_minus_random'])}`",
        f"- positive minus unrelated:         `{_fmt(effect['positive_minus_unrelated'])}`",
        "",
        "> " + effect["interpretation_note"],
        "",
        "## Sample generations",
        "",
        "Unfiltered. The first prompt is shown under every condition.",
        "",
    ]
    for result in [r for r in results if r.prompt_index == 0]:
        lines += [
            f"### `{result.condition}` (feature={result.feature_id}, strength={result.strength:+.2f})",
            "",
            "```text",
            result.text.strip() or "(empty)",
            "```",
            "",
        ]
    return "\n".join(lines)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
