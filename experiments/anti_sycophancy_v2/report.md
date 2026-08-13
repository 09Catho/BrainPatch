# `anti_sycophancy_v2` — report

**Verdict: NEGATIVE. The test split was never opened, and no patch ships.**

Under the pre-registered selection rule, none of the six shortlisted
configurations improved the free-generation correction rate on validation, so
[`success_criteria.md`](success_criteria.md) §4 closed the test split. It is
still unscored and remains usable as a genuinely held-out set.

The interesting part is *why*, and v2 answers a question v1 could only raise.

---

## 1. The length confound is fixed

This was the defect that decided v1.

| | v1 | v2 |
|---|---|---|
| preferred response longer | **96%** of pairs | **53.2%** (chars) / 53.0% (tokens) |
| mean length gap | large, one-directional | **−0.93 chars** / **+0.28 tokens** |
| median length gap | — | **+2 chars** / **+1 token** |
| class predicts length (corr) | +0.10 (chars) | **−0.014** (chars) / **+0.023** (tokens) |
| result-level `corr(Δ, length gap)` | **+0.457** → disqualified | **mean 0.129, max 0.43** across 330 configurations; the best configuration per method sat at **+0.055, −0.020, −0.025, +0.176, +0.016** |

The fix worked. Length is no longer a plausible explanation for anything
measured here, and the normalized and total margins now agree in sign
throughout (e.g. CAA at layer 26: normalized **+0.355**, total **+3.74**).

## 2. Baseline: the behaviour is real, and there is room to move

| Split | false claims: prefers sycophantic | true claims: prefers correct | generation correction rate | false-disagreement on true |
|---|---|---|---|---|
| train (155) | **72 / 99** (72.7%) | 27 / 56 (48.2%) | **32.3%** | 1.8% |
| validation (77) | **34 / 44** (77.3%) | 10 / 33 (30.3%) | **18.2%** | 3.0% |

Nowhere near ceiling, so the benchmark did not need hardening. Note the
generation correction rate is **18–32%** here against v1's **5.7%** — v1's
generation metric was pinned near the floor, which is part of why it could not
detect movement.

One honest oddity: on *true* claims the paired log-probability prefers the
manufactured disagreement (validation −0.308), while free generation agrees with
true claims **97%** of the time. Forced choice between two specific
continuations and what the model actually emits are not the same measurement —
a theme that turns out to be the whole story below.

## 3. Method comparison (validation, 330 configurations)

Best configuration per method, all at the strongest strength tested:

| Method | Layer | Position | Site | normalized `Δ_false` | 95% CI | total `Δ_false` | `Δ_true` | length *r* | probe acc |
|---|---|---|---|---|---|---|---|---|---|
| **CAA** (difference-of-means) | 26 | last_prompt | prompt | **+0.3553** | [+0.273, +0.442] | +3.739 | −0.241 | +0.055 | — |
| PCA | 16 | cont_last | prompt | +0.2911 | [+0.190, +0.393] | +3.194 | −0.463 | −0.020 | — |
| Linear probe | 26 | last_prompt | prompt | +0.1746 | [+0.098, +0.256] | +1.747 | −0.167 | −0.025 | 0.810 |
| SAE single feature | 18 | last_prompt | prompt | +0.1270 | [+0.079, +0.178] | +1.388 | −0.002 | +0.176 | — |
| SAE sparse (8 features) | 18 | cont_last | prompt | +0.0807 | [+0.029, +0.133] | +0.915 | −0.088 | +0.016 | — |

**CAA now wins, where PCA won in v1.** On the clean dataset the simplest method —
difference of class means — is the strongest, and it dominates the survivors
(21 of 27). The v1 ordering was PCA > probe > CAA > SAE; the v2 ordering is
CAA > PCA > probe > SAE. The one stable part is that **SAE is last in both.**

**Probe accuracy is not steerability, again.** Probe accuracy ranged 0.784 to
**1.000** (mean 0.937) while steering at half CAA's strength. v1's lesson
replicates on independent data.

**Injection site replicates and sharpens.** Maximum `Δ_false` by site:

| Site | max `Δ_false` |
|---|---|
| prompt tokens | **+0.3553** |
| prompt + generated | +0.3055 |
| generated tokens only | **−0.0002** |

Steering generated tokens does essentially *nothing* — even more starkly than
v1's ~6× gap. These directions work by changing how the prompt is read.

**Most configurations are contrarian, and the true-claim guard caught them.**
Only **27 of 330** survived stage 1, and the dominant cause of rejection was
`Δ_true` — the top log-prob movers have `Δ_true` of −0.24 (CAA) and −0.46 (PCA),
i.e. they push the model to disagree with *true* statements too. Without the
true-assertion controls, those would have looked like the best results in the
experiment.

## 4. Why the experiment stopped: log-probability and generation disagree

The six shortlisted survivors, in the pre-registered order (top 6 by `Δ_false`):

| Method | Layer | Site | ratio | `Δ_false` | correction-rate gain | selective gain |
|---|---|---|---|---|---|---|
| CAA | 22 | all | 0.35 | +0.2649 | **+0.000** | −0.030 |
| CAA | 16 | prompt | 0.35 | +0.2429 | **−0.045** | −0.015 |
| CAA | 20 | prompt | 0.35 | +0.2144 | **−0.068** | −0.038 |
| CAA | 18 | prompt | 0.35 | +0.2077 | **+0.000** | +0.000 |
| CAA | 22 | all | 0.20 | +0.1462 | **−0.023** | +0.008 |
| CAA | 20 | prompt | 0.20 | +0.1311 | **+0.000** | +0.030 |

Not one improved the correction rate. The rule closed the test split.

### The exploratory diagnostic

To distinguish "the shortlist narrowly missed" from "these directions do not
move generation", free generation was run on **all 27** survivors. This was
explicitly labelled exploratory and **could not promote anything** — the
pre-registered rule had already resolved the experiment.

```
corr(normalized Δ_false, free-generation correction gain) = −0.298
configurations improving generation:                        7 / 27
best correction-rate gain observed:                        +0.068
```

- Top 8 by log-probability effect → **mean correction gain −0.023**
- Top 8 by correction gain → **mean log-probability effect +0.069**

The relationship is **negative**. Selecting directions by paired
log-probability does not merely fail to select for generation improvement — it
*actively anti-selects* for it. The pre-registered shortlist rule, which ranks
by `Δ_false`, therefore sampled precisely the configurations least likely to
pass the generation gate.

This is the single most useful thing v2 produced, and it explains v1 exactly:
v1's winner had a strong log-prob effect (+0.074 on test, CI excluding zero,
beating every control) and a **falling** correction rate (5.7% → 3.8%). That was
not bad luck. It is what this correlation predicts.

No degeneration was observed anywhere (max degenerate fraction **0.00** across
all 27 configurations), so the failure is not the directions breaking the model.

## 5. Pre-registered criteria: status

| # | Gate | Status |
|---|---|---|
| G1–G6, G8–G10 | held-out effect, controls, length, contrarianism, degeneration, utility | **not evaluated** — test split never opened |
| **G7** | **free generation improves (hard gate)** | **FAILED at validation**, which is what stopped the experiment |

Amendment 1 (dataset audit: median gap measured in tokens rather than as an
unsatisfiable ratio) was made before any candidate was evaluated and is recorded
in [`success_criteria.md`](success_criteria.md#6a-amendment-1--dataset-audit-median-gap-measured-in-tokens-not-as-a-ratio),
including the fact that it is a relaxation in ratio terms. **No threshold was
loosened after seeing a result**, and G6 — the gate that actually decides
length confounding — was untouched.

## 6. What this says about the research question

> Can a tiny activation-space intervention make Qwen more likely to
> independently correct false user assertions, without simply disagreeing more,
> and without damaging unrelated behaviour?

On a clean, length-balanced dataset, with five discovery methods and 330
configurations:

- Directions that strongly move the **paired preference** exist and are easy to find. CAA at layer 26 shifts the normalized margin by +0.355 with a tight interval.
- Almost all of them achieve it **by becoming contrarian** — the true-claim guard rejects 303 of 330.
- Of the 27 that are selective, **the ones with the largest preference shift make free generation worse**, and the best generation improvement available anywhere in the surviving set is **+6.8 percentage points** from a direction with a weak preference shift.

So the honest answer from v2 is: **not demonstrated, and now we know the metric
that made it look otherwise.** A forced-choice log-probability margin is a
convenient, low-variance proxy that is *negatively* correlated with the thing it
is a proxy for, at least for this behaviour on this model.

## 7. What would settle it

1. **Select on generation from the start.** The obvious v3: skip log-probability ranking entirely and search directly against the free-generation correction rate, accepting the higher variance and cost. The exploratory data says the useful directions are in the *weak* tail of the log-prob ranking, which no log-prob-first design will surface.
2. **The +0.068 candidate is a hypothesis, not a result.** `CAA, layer 16, last_prompt, prompt-site, ratio 0.35` improved validation correction from 18.2% to 25.0%. It cannot be promoted here — that would be exactly the post-hoc selection the protocol exists to prevent. It should be a pre-registered *prediction* in a fresh experiment with its own test split.
3. **A stance judge instead of a keyword rubric.** The rubric scores "Yes, you're correct! …" as agreement even when the following text corrects the user, so the correction rate is probably biased low.
4. **A second model**, to separate properties of the behaviour from properties of Qwen2.5-1.5B.

## 8. Reproducing

```bash
python scripts/build_sycophancy_v2.py            # regenerates fixture + manifest, audits as a gate
modal run modal_app/antisycophancy_v2.py::stage_a_baseline
modal run modal_app/antisycophancy_v2.py::stage_b_discovery
modal run modal_app/antisycophancy_v2.py::stage_b_dissociation   # exploratory
```

`stage_c_test` exists but was never invoked; the test split has no scores
attached to it.

## 9. Cost

Modal metered spend for v2: **$0.31** (project total $1.06 → **$1.37** of the
$10 budget). L4, one GPU throughout. Activations were captured once and shared
across all five methods; the token-level length audit runs before any activation
is captured, so a confounded dataset would have cost nothing.

Artifacts: [`dataset_manifest.json`](dataset_manifest.json),
[`train_results.json`](train_results.json),
[`validation_results.json`](validation_results.json),
[`free_generation_results.json`](free_generation_results.json),
[`controls.json`](controls.json), [`test_results.json`](test_results.json).
