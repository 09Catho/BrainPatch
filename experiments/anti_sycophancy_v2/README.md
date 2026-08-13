# `anti_sycophancy_v2`

A second, cleaner attempt at the same question `anti_sycophancy_v1` failed to
answer. **Result: negative — the held-out test split was never opened, and no
patch ships.**

> Can a tiny activation-space intervention make Qwen more likely to
> independently correct false user assertions, **without simply making it
> disagree more often**, and **without materially damaging unrelated
> behaviour**?

## Read in this order

| File | What it is |
|---|---|
| [`protocol.md`](protocol.md) | What was run and why, including what v1 taught |
| [`success_criteria.md`](success_criteria.md) | Pre-registration, committed before any test access |
| [`report.md`](report.md) | **The results.** Start here if you only read one |
| [`dataset_manifest.json`](dataset_manifest.json) | Composition, audits, split hashes |
| [`train_results.json`](train_results.json) | Baseline on train, calibration, probe accuracy |
| [`validation_results.json`](validation_results.json) | All 330 scanned configurations |
| [`free_generation_results.json`](free_generation_results.json) | The gate that stopped the experiment |
| [`controls.json`](controls.json) | Why the test-side controls were not run |
| [`test_results.json`](test_results.json) | Record that the split stays closed |

## The three things worth knowing

**1. The v1 length confound is gone.** v1's preferred response was longer in 96%
of pairs and its result was disqualified at `corr(Δ, length gap) = +0.457`. v2's
pairs are balanced by construction: mean gap **+0.28 tokens**, median **+1
token**, class/length correlation **+0.023**, preferred longer in **53%**. Across
330 configurations the mean length correlation was **0.129**.

**2. Log-probability and free generation are negatively correlated.**

```
corr(normalized Δ_false, free-generation correction gain) = −0.298   (27 configurations)
top 8 by log-prob effect  →  mean correction gain  −0.023
top 8 by correction gain  →  mean log-prob effect  +0.069
```

Ranking directions by paired log-probability *anti-selects* for the behaviour
you actually want. This explains v1's outcome exactly, and it means a
log-prob-first search cannot find what this experiment was looking for.

**3. Most "successful" directions are just contrarian.** Only **27 of 330**
configurations survived the true-claim guard. The strongest log-prob movers push
the model to disagree with **true** statements as well (`Δ_true` −0.24 for CAA,
−0.46 for PCA). Without true-assertion controls those would have looked like the
best results in the study.

## Method ordering, on a clean dataset

**CAA > PCA > probe > SAE single > SAE sparse.**

CAA (difference-of-means) wins where PCA won in v1, and takes 21 of the 27
surviving configurations. A linear probe reached **100%** predictive accuracy
while steering at half CAA's strength — predictive accuracy is not steering
efficacy, replicated on independent data. **SAE was last in both experiments.**
BrainPatch ships portable direction vectors and is agnostic about where a
direction comes from; a simple difference of means winning is a fine outcome.

Injection site replicates v1 and sharpens it: prompt tokens **+0.355**, prompt +
generated **+0.306**, generated tokens only **−0.0002**.

## Relationship to v1

v1 is frozen and untouched. Its test split is opened and is **not** reused: the
two pools share **zero** propositions and **zero** topics, asserted by test in
`tests/test_sycophancy_v2_dataset.py`.

## Dataset

387 fresh propositions, 11 categories, **155 train / 77 validation / 155 test**,
**37.5%** true-assertion controls, zero near-duplicates. Built and audited by
`scripts/build_sycophancy_v2.py`, which **refuses to emit the dataset** if the
length audit fails.

The test split has never been scored. It remains a genuinely held-out set for a
future pre-registered experiment.

Modal spend for v2: **$0.31**; project total **$1.37** of a $10 budget.
