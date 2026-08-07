# Methodology

How BrainPatch decides whether an intervention did anything, and why the
controls are shaped the way they are.

## The claim structure

Four distinct things get confused in interpretability work. BrainPatch keeps
them apart, and `evidence_level` records which one a given statement has earned.

1. **Correlation.** A feature fires on tokens that look like *X*. This is what
   top-activating-examples tables show. It is the weakest evidence there is, and
   it is where most feature "labels" in the wild stop.
2. **Predictive usefulness.** The feature's activation predicts *X* on held-out
   data. Stronger, still not causal.
3. **Intervention evidence.** Steering the feature changes the output. Necessary
   but nowhere near sufficient — *any* large enough perturbation changes output.
4. **Reproducible causal effect.** Steering changes behaviour in a way that
   scale-matched controls in other directions do **not**.

Only (4) licenses a behavioural name. In `smoke_v0` we reached (3) and failed
(4), so the shipped patches are named after their feature IDs.

## Control design

Three conditions exist purely to rule out alternative explanations.

### `zero` — is the harness itself clean?

A patch installed at strength 0, everything else identical. Must reproduce the
no-hook baseline **byte-for-byte**.

Implemented so this is true by construction, not by luck: when every coefficient
resolves to zero the plan returns an empty edit list, the hook returns `None`,
and the residual tensor is never touched. No `x + 0.0`, no float round trip.

If this condition ever fails, every baseline in every experiment is contaminated
by an unknown perturbation, and no measured effect can be attributed to
anything. It is checked first and the pipeline aborts on failure.

Measured in `smoke_v0`: 6/6 prompts identical, `applied_passes = 0`.

### `random_positive` / `random_negative` — is this *direction* special?

A random unit vector from a seeded generator, injected through the identical
coefficient path.

The scale-matching is exact rather than approximate. Decoder columns are
constrained to unit L2 norm, random directions are normalised to unit L2 norm,
and both are multiplied by the same `coefficient / input_scale`. The injected
vectors therefore have **identical** norms — measured at 28.5178 for every
condition in `smoke_v0`. The only difference is direction.

This matters because the naive alternative — adding Gaussian noise of arbitrary
scale — controls for nothing. If the noise is smaller than the intervention, the
intervention "wins" trivially; if larger, it "loses" trivially. Neither tells you
anything about direction.

### `unrelated_positive` — is this *feature* special?

A different, real SAE feature at the same strength. Distinguishes "this
particular feature does something" from "any dictionary direction does this".

## Effect measurement

`divergence = 1 − Jaccard(3-grams of baseline, 3-grams of condition)`.

- 0.0 — identical text
- 1.0 — no shared trigrams

Chosen because it needs no judge model and no paid API, and because it is
symmetric and bounded. Its weakness is real and worth stating: **it measures
that the output changed, not how.** A genuine behavioural effect that preserves
phrasing would be invisible to it, and a fluent-but-irrelevant rewrite scores
the same as a meaningful one.

Alongside it, every generation gets model-free metrics: distinct-1/2/3,
trigram repetition rate, longest repeated n-gram, type-token ratio, unigram
entropy, and top-bigram fraction. These are **degeneration tripwires**, not
quality measures. A high `distinct_2` does not mean the answer is good; a low one
strongly suggests it is broken.

They are heuristics and they have missed real degeneration — see the strength-32
case in the research log. A `degeneration_flag` of `False` means "no obvious
breakage detected", not "output is fine".

## Utility retention

Steering toward a target behaviour is worthless if it breaks arithmetic or
instruction-following. Ten hand-written probes across arithmetic, factual QA,
instruction-following and reasoning, plus three open-ended continuations scored
only for degeneration.

They are deliberately easy — a healthy Qwen2.5-1.5B-Instruct should get nearly
all of them — so a drop is signal rather than noise. They are development
fixtures, not a benchmark, and an absolute score on them means nothing.

## Generation settings

Greedy (`do_sample=False`) throughout, identical `GenerationConfig` object
passed to every condition. Sampling would add variance that many more paid
generations would be needed to average out before an effect could be
distinguished from noise.

## Choosing a strength

Strength is measured, not assumed. The right magnitude depends on the
residual-stream norm at the hooked layer, which is a property of the model and
layer.

`sweep_strength` runs a dose–response curve and reports divergence and
degeneration rate per strength. The usable window is where divergence has risen
but degeneration has not. For `smoke_v0` that was strength 8–16; below 8 the
greedy output was unchanged, at 32 the model looped, at 64 it collapsed.

## Statistical honesty

The `smoke_v0` numbers come from 6 prompts with one greedy generation per
condition. There is no repeated sampling and no significance testing. They are
sufficient to observe that the controls did not separate. They are **not** effect
sizes and should not be quoted as such.

Any future claim of a positive effect needs many more prompts, repeated
sampling, and an actual test.

## What is deliberately not done

- **No automatic semantic labelling.** Feature records are written with
  `hypothesis: null` and `evidence_level: "none"`. A labelling heuristic that
  reads top contexts and emits "the uncertainty feature" would manufacture
  exactly the false confidence this project is built to avoid.
- **No cherry-picking.** Every generation from every condition is persisted to
  `all_generations.jsonl`. Reports show the first prompt under every condition,
  not a selected one.
- **No filtering of negative results.** The `smoke_v0` result is negative and is
  reported as the headline.
