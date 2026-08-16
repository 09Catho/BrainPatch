# `anti_sycophancy_v3`

**All 11 pre-registered gates passed.** The first positive result in this
research programme — and the one caveat below is as important as the headline.

> **The specificity is the weak link.** The best of ten norm-matched random
> directions scored **+0.158**; the real direction scored **+0.167**. That is a
> margin of about **one item in 120**. The random directions span **−0.133 to
> +0.158**, so an intervention of this size moves the correction rate
> substantially *whichever way you push*. G4 asks only that the real direction
> beat the maximum random control, and it does — but a reader is entitled to
> regard the direction-specificity as unestablished.

## The result

**SAE single feature 204, layer 18, `last_prompt` extraction, prompt-token
injection, strength ratio 0.35.**

| | baseline | patched |
|---|---|---|
| **correction rate on false claims** | **0.233** | **0.400** |
| SIS | +0.221 | +0.338 |
| false disagreement on true claims | 0.013 | 0.062 |
| correct agreement on true claims | 0.500 | 0.613 |
| degenerate rate | 0.000 | 0.000 |

**+0.167 absolute, +71.4% relative**, CI [+0.092, +0.242]. Paired over the same
120 items: 22 improved, 2 worsened, 96 unchanged, McNemar **p = 3.6 × 10⁻⁵**.
Response length changed **+0.21%**. Utility 1.000 → 1.000 (n=32), zero refusals.
The zero-strength control reproduced the baseline character-for-character.

Two gates pass narrowly: **G3 at exactly its +0.05 threshold** (true-claim false
disagreement rose from 1/80 to 5/80) and **G4 by one item**.

## The finding that generalises

The method ranking **inverts** depending on what you measure.

```
by log-probability steering (v1, v2):  CAA / PCA  >  probe  >  SAE   (SAE last)
by free-generation behaviour (v3):     SAE  >  probe  >  PCA  >  CAA (CAA last, +0.000)
```

The winning direction's own log-prob diagnostic is **−0.0136** — the patch that
improves generation by 71% makes the paired log-probability margin slightly
*worse*. `corr(log-prob, generation gain)` was **+0.163** here against
**−0.298** in v2: not stable in sign, never large.

```
representation quality  ≠  log-prob steerability  ≠  behavioural usefulness
```

A linear probe hit **1.000** predictive accuracy and delivered **+0.011** on
full validation.

## Read in this order

| File | What it is |
|---|---|
| [`report.md`](report.md) | **The results.** Start here |
| [`protocol.md`](protocol.md) | What was run and why |
| [`success_criteria.md`](success_criteria.md) | Pre-registration, committed before test access |
| [`dataset_manifest.json`](dataset_manifest.json) | Composition, audits, hashes, overlap checks |
| [`baseline_results.json`](baseline_results.json) | Baseline generations and rates |
| [`discovery_results.json`](discovery_results.json) | Candidate search: cheap filter, generation ranking |
| [`validation_results.json`](validation_results.json) | Finalists on full validation; frozen winner |
| [`test_results.json`](test_results.json) | The single test pass, with every stored response |
| [`controls.json`](controls.json) | Random, unrelated, sign, shuffled, zero-strength |
| [`utility_results.json`](utility_results.json) | Utility battery |

## Dataset

550 fresh propositions, 11 categories, **200 / 150 / 200**, **40%** true-claim
controls, zero near-duplicates, **zero** topic or assertion overlap with v1 or
v2. Mean length gap +1.9 chars, class/length correlation +0.043.

## Method

Selection was by **generated behaviour**, never by log-probability. Cheap
metrics only filtered; ≤30 candidates reached generation (declared in advance);
the top 5 were confirmed on the full validation split. That confirmation step
mattered: probe and PCA candidates that looked strong on the 60-item ranking
subset (+0.167, +0.139) collapsed to +0.011 and +0.000 on full validation.

## Shipped

`anti-sycophancy.brainpatch` — **7,382 bytes**, `controlled_interventional`,
one intervention at layer 18 with `site: prompt`.

Transformers is marked **`implemented`**, not `verified`: the behaviour was
measured on that backend in stage C, but the artifact-level reproduction check
did not complete within budget. llama.cpp and vLLM are **`unsupported`** —
neither can express prompt-token-only injection, so applying it there would be
a configuration with no test evidence.

Modal spend for v3: **$0.66**; project total **$2.03** of $10.
