# `anti_sycophancy_v3` — report

**Verdict: all 11 pre-registered gates PASS. This is the first positive result
in the BrainPatch research programme.**

> **Read this caveat before the headline.** The best of ten norm-matched random
> directions scored **+0.158**. The real direction scored **+0.167**. That
> margin is about **one item in 120**. G4 asks whether the effect beats the
> maximum random control, and it does — but only just. The random directions
> span **−0.133 to +0.158**, an enormous null distribution, which means an
> intervention of this size at this strength moves the correction rate a great
> deal *whatever direction you push in*. The specificity of the discovered
> direction is the weakest link in this result, and no amount of significance
> elsewhere repairs it.

Everything below should be read against that.

---

## 1. What was measured

v2 established that ranking candidates by continuation log-probability
*anti-selects* for generated behaviour (`corr = −0.298`). v3 therefore optimises
free generation directly and demotes log-probability to a diagnostic.

Deterministic throughout: greedy decoding, `max_new_tokens=96`, one fixed chat
template, identical prompts between conditions. **Every generated response is
stored** in `test_results.json`.

Classification uses two independent evaluators plus a degeneracy layer; where
both return **definite and opposing** verdicts the item is labelled `OTHER`
rather than forced. Evaluator agreement was **1.000** at baseline and **0.995**
patched.

## 2. Dataset

550 fresh propositions, 11 categories, **200 train / 150 validation / 200 test**,
**40%** true-assertion controls. **Zero** topic and **zero** assertion overlap
with v1 or v2, asserted by test.

Length audit (the v2 fix, carried over because log-probability is still
reported): mean gap **+1.9 chars**, median **+1.5**, class/length correlation
**+0.043**, preferred longer in **50.9%**. Zero near-duplicates.

## 3. Baseline

| Split | correction rate | 95% CI | sycophantic | true correct-agreement | false disagreement | SIS |
|---|---|---|---|---|---|---|
| train (200) | 0.308 | [0.225, 0.392] | 0.517 | 0.950 | 0.013 | +0.296 |
| validation (150) | 0.144 | [0.078, 0.222] | 0.622 | 0.967 | 0.000 | +0.144 |
| **test (200)** | **0.233** | — | 0.317 | 0.500 | 0.013 | +0.221 |

The three splits differ substantially in difficulty. The test split's invitation
phrasings are more challenge-inviting ("Push back if you disagree", "Am I
wrong?"), which raises its baseline correction rate and pushes many true-claim
responses out of the confident-agreement class into `OTHER` (correct agreement
0.500 on test against 0.95–0.97 elsewhere). Comparisons are always within a
split on identical prompts, so this does not bias the effect estimate — but it
means the three baselines are not interchangeable.

## 4. The result

**Frozen configuration** (written to disk before test, read from disk by the
test stage): SAE **single feature 204**, layer **18**, extraction `last_prompt`,
injection **prompt tokens**, strength ratio **0.35** (‖δ‖ = 22.39 against a
median residual norm of 63.97).

| | baseline | patched | change |
|---|---|---|---|
| **correction rate (false claims)** | **0.233** | **0.400** | **+0.167**, CI [+0.092, +0.242], **+71.4%** relative |
| SIS | +0.221 | +0.338 | **+0.117** |
| false-disagreement (true claims) | 0.013 | 0.062 | **+0.050** |
| correct agreement (true claims) | 0.500 | 0.613 | +0.113 |
| degenerate rate | 0.000 | 0.000 | 0.000 |
| mean response length | — | — | **+0.21%** |

Paired over the same 120 false-claim items: **22 improved, 2 worsened, 96
unchanged**, exact McNemar **p = 3.6 × 10⁻⁵**.

The zero-strength control reproduced the baseline **character-for-character**,
which is what makes the comparison trustworthy at all.

## 5. Gates

| # | Gate | Threshold | Observed | |
|---|---|---|---|---|
| G1 | correction improves | ≥ +0.10, CI excludes 0 | **+0.167**, CI [+0.092, +0.242] | PASS |
| G2 | SIS improves | ≥ +0.08 | **+0.117** | PASS |
| G3 | false-disagreement bounded | ≤ +0.05 | **+0.050** | PASS *(exactly at the threshold)* |
| G4 | beats random | > max of 10 | +0.167 vs **+0.158** | PASS *(by one item)* |
| G5 | beats unrelated real | > max of 3 | +0.167 vs **+0.025** | PASS |
| G6 | sign control | no gain ≥ +0.05 | **−0.075** | PASS |
| G7 | shuffled-label | not credibly positive | **−0.108**, CI [−0.183, −0.033] | PASS |
| G8 | no degeneration | ≤ 0.02 | **0.000** | PASS |
| G9 | utility preserved | ≤ 5pp drop | **1.000 → 1.000** (n=32), refusals 0 | PASS |
| G10 | measured in real generation | — | stored greedy generations | PASS |
| G11 | not explained by length | < 20% | **+0.21%** | PASS |

**Two gates pass on a knife edge and must be read that way.** G3 landed at
exactly +0.050 against a ≤ +0.05 threshold: false disagreement went from 1/80 to
5/80 true-claim items. G4 passes by 0.009. Neither is a comfortable margin.

## 6. Controls in full

| Control | correction gain |
|---|---|
| 10 norm-matched random directions | −0.133, −0.133, −0.100, −0.092, −0.083, −0.025, −0.008, +0.033, +0.083, **+0.158** |
| unrelated: verbosity (cos −0.071) | −0.058 |
| unrelated: contradiction (cos −0.042) | −0.050 |
| unrelated: verification (cos −0.006) | +0.025 |
| sign flipped | −0.075 |
| shuffled labels (re-selected SAE feature **1726**, real was **204**) | −0.108, CI [−0.183, −0.033] |
| zero strength | identical to baseline |

The shuffled-label control genuinely re-ran SAE feature selection on permuted
labels and picked a different feature, so it tests the pipeline rather than a
substitute. Its effect is clearly negative, which is the safe direction.

## 7. The PatchBench finding: the method ranking inverts

This is the most transferable result in the whole programme.

**Ranked by log-probability steering** (v1 and v2, two independent datasets):

```
CAA / PCA  >  linear probe  >  SAE          (SAE last in both)
```

**Ranked by free-generation correction gain** (v3, same five methods):

| Method | best subset gain | log-prob effect of that config |
|---|---|---|
| **SAE single feature** | **+0.194** | −0.124 |
| linear probe | +0.167 | −0.188 |
| PCA | +0.139 | −0.136 |
| SAE sparse | +0.111 | −0.239 |
| **CAA (difference-of-means)** | **+0.000** | −0.314 |

**The ordering is essentially reversed.** SAE — worst on log-probability in both
earlier experiments — is best on generation. CAA — best or near-best on
log-probability — produces no generation improvement at all. And the winning
direction's own log-prob diagnostic on the test split is **−0.0136**: the patch
that improves generation by 71% makes the paired log-probability margin
slightly *worse*.

`corr(log-prob effect, generation gain)` over 25 candidates was **+0.163** here,
against **−0.298** in v2. Across the two experiments the correlation is not
stable in sign and never large. The honest summary is not "log-probability is
negatively predictive" but **"log-probability is not predictive"**:

```
representation quality  ≠  log-prob steerability  ≠  behavioural usefulness
```

A linear probe reached **1.000** predictive accuracy (mean 0.814 across the
grid) and its full-validation gain was **+0.011**.

## 8. The validation subset was noisy, and the confirmation step earned its keep

Three of five finalists shrank sharply between the 60-item ranking subset and
the full 150-item validation split:

| Finalist | subset gain | full-validation gain |
|---|---|---|
| sae_single L18 prompt | +0.167 | **+0.156** |
| sae_single L18 all | +0.194 | +0.133 |
| probe L18 all | +0.167 | **+0.011** |
| pca L24 all | +0.139 | **+0.000** |
| sae_single L18 cont_mean all | +0.111 | +0.022 |

Ranking on 60 items (36 false claims, baseline 2 corrections) was close to the
resolution limit. Only the SAE candidates survived confirmation. Without the
full-validation step, v3 would have frozen a configuration worth +0.011.

## 9. What this does and does not establish

**Does:** on a fresh, length-balanced, contamination-checked dataset, with the
configuration frozen before the split was opened, a single SAE feature injected
at the prompt raised the free-generation correction rate from 23.3% to 40.0%,
paired p = 3.6 × 10⁻⁵, with no degeneration, no measured utility loss, no length
inflation, and true-claim agreement *improving* rather than degrading. It
reverses under sign flip and is not reproduced by shuffled labels or by
unrelated real directions.

**Does not:** establish that the effect is specific to *this* direction. One of
ten random directions came within one item of it. Nor does it establish
replication — a second model or an independent dataset was pre-declared out of
scope and is not claimed.

**Also worth stating plainly:** G3 sits exactly on its threshold. Contrarianism
increased, from 1 to 5 of 80 true claims. The intervention is not free.

## 10. What would settle the specificity question

1. **More random controls.** Ten gives a poor estimate of a null whose observed range is 0.29 wide. A hundred would put the real direction at a properly estimated percentile.
2. **Strength-matched nulls at several strengths.** The winner sits at ratio 0.35, the largest scanned; random directions at that strength are evidently disruptive. Testing whether the real/random gap widens at lower strengths would separate "this direction" from "any large perturbation".
3. **A second model**, for replication.
4. **A model-based stance judge** to cross-check the keyword evaluator, which cannot see whether the correction is factually right.

## 11. Reproducing

```bash
python scripts/build_sycophancy_v3.py
modal run modal_app/antisycophancy_v3.py::stage_a_baseline
modal run modal_app/antisycophancy_v3.py::stage_b_discovery
modal run modal_app/antisycophancy_v3.py::stage_c_test
```

Modal spend for v3: **$0.71** (project total $1.37 → **$2.08** of $10).

## 12. The shipped artifact was wrong, and the behavioural check caught it

Worth recording in full, because the failure mode is general.

SAE feature selection multiplies the decoder column by the sign of the contrast
effect. Feature 204's effect is **−0.1003**, so discovery used the **negated**
column. `compile_from_sae` emits the *unsigned* column and carries sign in the
coefficient — and the first spec was written with a **positive** coefficient.

The first artifact was therefore the **sign control**:

| | correction rate |
|---|---|
| baseline | 0.233 |
| stage C, measured | **0.400** |
| first artifact (wrong sign) | **0.150** |
| stage C sign control | 0.158 |

**The first integrity check passed it.** That check compared the compiled vector
against `feature_direction(204)` — the *unsigned* column — and reported
`cosine = 1.0`. It reproduced the exact bug it was meant to catch, because a
check that ignores sign cannot see a sign error. Only regenerating the split and
comparing *behaviour* exposed it.

After correcting the coefficient to **−12.5621**:

| Check | Result |
|---|---|
| cosine to the tested direction | **+0.99999998** (was −0.99999998) |
| ‖δ‖ applied | 22.38996 vs stage C's 22.39017 |
| site restriction | 1 edit on the prompt pass, 0 on continuation |
| **correction rate** | **0.3917** vs stage C's **0.4000** — within one item in 120 |
| exact string matches | 64/200, up from 1/200 |

Exact string identity is neither required nor achievable: re-batching changes
left-padding and F16 storage perturbs the vector by ~7×10⁻⁴, and either flips
argmax on near-ties. The behavioural rate is the meaningful comparison, and it
agrees.

`tests/test_patch_sign.py` now pins the sign against the recorded
`direction_sign`, so a spec and an artifact cannot disagree silently again.

## 13. What shipped, and what the artifact does not claim

`anti-sycophancy.brainpatch`, **7,598 bytes**, `evidence_level:
controlled_interventional`, discovery method `sae_single`, one intervention at
layer 18 with `site: prompt` and coefficient **−12.5621**. Full provenance is in the manifest, including the
training-split hash and the strength calibration.

Backend statuses are deliberately conservative:

| Backend | Status | Why |
|---|---|---|
| Transformers | `verified` | the compiled delta is numerically the tested direction (cosine +0.99999998), the prompt-only site restriction is honoured, and the artifact reproduces the test correction rate to within one item (0.3917 vs 0.4000) |
| llama.cpp | `unsupported` | a control vector binds for a whole run; it cannot express `site: prompt` |
| vLLM | `unsupported` | continuous batching shares one forward pass, so prompt and generated positions cannot be separated |
| MLX-LM | `experimental` | no Apple Silicon available |

The validated configuration is therefore **not** portable across engines today.
That is a real limitation of the result, not a packaging detail: the two
backends that cannot express it would silently apply the direction to every
token, a configuration with no test evidence behind it.
