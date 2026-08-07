# BrainPatch research log

Chronological record of what was built, what was measured, what broke, and what
remains unresolved. No marketing. Negative results are recorded as results.

All work below ran on Modal (workspace `ashketchume45`, environment
`brainpatch-dev`), on a single NVIDIA L4. The local machine never executed a
model, never downloaded weights, and had no packages installed for this project.

---

## 2026-08-07 — Phase 0: environment audit

Modal CLI 1.5.3, authenticated. Environment `brainpatch-dev` active. Volume
`brainpatch-data` and Secret `huggingface-secret` both present. `gh` 2.68.1
authenticated. Not a git repository yet.

Local Python 3.11 already carried `torch 2.6.0+cu126` and `transformers 4.47.0`
from unrelated prior work. **Decision:** do not use them, do not import them,
install nothing. The pre-existing install makes the import boundary *more*
important, not less — without an enforced boundary, code that accidentally
imports torch would work on this machine and fail everywhere else. Hence the
meta-path blocker in `tests/conftest.py` and the AST check over `modal_app/`.

`pyyaml`, `typer`, `rich`, `pytest` and `modal` were already present, so the
project needed **zero local installs**.

---

## Design decisions

### Two packages, one boundary

`brainpatch/` is pure Python; `brainpatch/ml/` is torch-only and is never
imported eagerly. `brainpatch/__init__.py` exposes `BrainPatchedModel` through
`__getattr__` so the convenient import works on a GPU without being mandatory
elsewhere. Verified by importing the package with `torch`, `transformers`,
`datasets` and `numpy` blocked at the import hook.

### Storage layout

Activation metadata is `(example_index, token_position, token_id)` as int32 —
12 bytes per row — with text stored once in `examples.jsonl`. Measured cost
came out at **3084.01 bytes/token**, exactly `1536 × 2 + 12`, confirming no
hidden overhead. Duplicating context strings per token would have multiplied
corpus size several-fold for information already recoverable by lookup.

Shards are immutable once recorded in the manifest. Resume is the default;
`--force` is required to discard. For work that costs money, an accidental
re-run should continue, not restart.

### Why unit-norm decoder columns are not optional

Without the constraint, the SAE can halve every decoder column and double every
activation with no change in loss. For reconstruction that is merely untidy. For
*interventions* it is fatal: a patch says "add strength 16 along feature 727's
direction", and if the direction's length is arbitrary then so is the strength.
Measured post-training norms: mean 1.0, min 0.9999992, max 1.0000007.

Decoder gradients have their column-parallel component projected out before each
optimizer step, otherwise the optimizer spends part of every update changing
norms that renormalisation immediately undoes.

### Why `input_scale` is stored in the checkpoint

Inputs are normalised so `E[||x||₂] = √d_in`, so decoder columns live in
normalised space. Injecting into the raw residual stream divides by that scale.
If it were not recorded, "strength 1.0" would mean a different physical
magnitude for every SAE and no patch would be portable even between runs on the
same model. Measured for `smoke_v0`: **0.5610531069008018**.

### Why `strength = 0` had to be *exactly* baseline

Implemented by making the plan return an empty edit list when all coefficients
are zero, so the hook returns `None` and the residual tensor is never touched.
Not "add 0.0" — no arithmetic at all.

The reason is not elegance. If zeroed patches were only *approximately*
baseline, every baseline in every experiment would be contaminated by an
unknown perturbation, and no measured effect could be attributed. Verified
empirically at 6/6 prompts byte-identical with `applied_passes = 0`.

---

## Modal SDK findings

### `hf_transfer` fails against the Volume

First `cache_model` run died with `RuntimeError: An error occurred while
downloading using hf_transfer`. Its parallel range-writes do not work against
the Volume's network filesystem under gVisor. Disabled via
`HF_HUB_ENABLE_HF_TRANSFER=0`. A model is downloaded exactly once ever, so the
throughput gain was worth nothing against the reliability loss. Download then
completed in 64.6 s for 2.886 GB.

### Duplicate `local_entrypoint` names break aggregation

Each `modal_app/*.py` module initially defined its own `main()` local
entrypoint. Aggregating them in `modal_app/app.py` produced
`InvalidError: Duplicate local entrypoint name: main`, which broke **every**
`modal run modal_app/app.py::...` command.

This was caught only because the documented commands were actually executed
rather than assumed to work. Fixed by removing the per-module entrypoints —
`modal run file.py::function` works directly on an `@app.function`, so they
were redundant anyway. `smoke_pipeline` is now the single local entrypoint.

### `from __future__ import annotations` breaks Modal class parameters

`modal_app/web.py` declared `experiment: str = modal.parameter(default=...)`.
With the future import active, the annotation is the *string* `"str"`, and
Modal's parameter validator calls `.__name__` on it:
`AttributeError: 'str' object has no attribute '__name__'`.

Because `app.py` imports `web.py`, this broke every aggregated entry point.
Fixed by dropping the future import from that one file, with a comment
explaining why it must not be re-added. Python 3.11 evaluates the remaining
PEP 604 unions natively.

### Images cannot be extended after `add_local_*`

`WEB_IMAGE = ML_IMAGE.pip_install("gradio", ...)` raised
`InvalidError: An image tried to run a build step after using image.add_local_*`.
Local sources attach at container start rather than as a layer, so nothing may
follow them. This failed at *run* time, not import time, so it only surfaced
when a function whose module imported `web.py` was actually invoked.

Fixed with a `_build(*packages)` helper that composes each image from its own
package list, always adding local sources last.

---

## Measurements

### L4 probe

NVIDIA L4, compute capability 8.9, 58 SMs, 22.03 GB total / 21.84 GB free.
torch 2.6.0+cu124, CUDA 12.4, cuDNN 9.1.0. bf16 supported and finite.
GPU matmul max absolute error vs a CPU reference: **0.0**.
Naive bf16 matmul loop: 18.08 TFLOPS (a crude loop, not a tuned benchmark).

### Model cache

Download 2.886 GB / 10 files in 64.6 s on a **CPU** container — paying L4 rates
to wait on a network transfer would be waste.

Reload from the Volume in a **separate** L4 invocation took **7.72 s** with
`cache_present_before_load: true`. The separation is the actual test: it proves
the cache survives container teardown, which is the point of the Volume.

Architecture discovered, not assumed: hidden 1536, 28 decoder blocks. Layer 18
validated against that depth before any extraction ran.

Model load peak VRAM 2945.3 MB; inference peak 2955.2 MB.

### Attention sink confirmed empirically

The hook smoke test on `"The capital of France is"` reported a mean activation
norm of 2267 across 5 tokens with the **first token at 11052**. Excluding
position 0 gives ~70.8, which matches the corpus mean of ~70 implied
independently by the measured `input_scale` (`√1536 / 0.5611 = 69.85`).

Position 0 is therefore ~156× the typical activation norm. Extraction drops it
(`skip_first_n_tokens = 1`); including it would have dominated the input-scale
normalisation and spent dictionary capacity on a single positional artifact.
Two independent measurements agreeing on ~70 is a useful consistency check on
the whole extraction-and-normalisation path.

### Extraction

`tiny_v0` (2,000 tokens) ran first as a cheap bug-catcher before the real run —
272.6 tokens/s, but that figure includes the one-off dataset build inside the
timed region and should not be quoted as throughput.

`smoke_v0`: 20,000 tokens at layer 18, seq len 256, batch 8, in 8.367 s →
**2390.4 tokens/s**. 58.8 MB in one shard. Peak VRAM 3553.4 MB.

### SAE training

d_in 1536, d_sae 2048 (expansion 1.33×), k 32, 6,295,040 parameters.
2220 steps / 60 epochs in 78.6 s → **28.229 steps/s**. Peak VRAM **295.4 MB**
(the SAE is negligible next to the model).

| | train | validation |
|---|---|---|
| explained variance | 0.762 | 0.658 |
| cosine similarity | 0.925 | 0.890 |
| normalised MSE | 0.145 | 0.211 |

L0 exactly 32.0 (Top-K working as specified). **0 dead features of 2048.**

**Observation.** The 0.104 explained-variance gap between train and validation
is genuine overfitting. 19,000 training rows against a 2048-feature dictionary
is far too little data — roughly 9 rows per feature. Zero dead features at this
scale is not a sign of health either; with so few examples every feature can
find something to fire on. Both are expected consequences of the corpus size and
neither should be read as SAE quality.

### Dose-response sweep

Strength is not something to guess: the right magnitude depends on the
residual-stream norm at the hooked layer. Measured on feature 727, 3 prompts:

| strength | delta norm | divergence | note |
|---|---|---|---|
| 2 | 3.565 | 0.434 | probe prompt output **unchanged** |
| 4 | 7.129 | 0.443 | probe prompt output **unchanged** |
| 8 | 14.259 | 0.563 | first wording change |
| 16 | 28.518 | 0.770 | rewritten, coherent, still correct |
| 32 | 57.036 | 0.958 | looping |
| 64 | 114.071 | 1.000 | total collapse |

Against a raw residual norm of ~70, strength 16 is a ~41% perturbation. The
initial `intervention_smoke` at strength 2.0 produced **byte-identical** output
to baseline — a 5% perturbation is simply too small to change greedy decoding.
That is why the sweep exists rather than a guessed default.

### The degeneration detector was wrong, and was fixed

At strength 32 the model produced:

> `"as they encounter each other, as they interact with each other, as they collide, as they merge, as they combine, as they fuse, ..."`

The detector scored this **clean**. Each clause ends differently, so `distinct_2`
stayed at 0.700 and `longest_repeated_ngram` at 5 — set-based diversity measures
do not see a repeated *syntactic frame*.

Added `most_common_ngram_fraction`: the share of all bigrams taken by the single
most frequent one. Measured 0.167 for the looping text against 0.028 for a
healthy generation. Threshold set at 0.10 for texts of ≥30 words. The strength-16
output remains correctly unflagged.

This is worth recording as a limitation, not just a fix: these metrics are
heuristics, they have demonstrably missed real degeneration once, and a `False`
means "no obvious breakage detected", not "output is fine".

---

## The headline result — NEGATIVE

`smoke_v0_intervention`: feature 727 vs unrelated feature 1270, strength ±16,
6 prompts, greedy decoding, 96 new tokens, identical generation settings across
all conditions.

| condition | divergence from baseline | delta norm |
|---|---|---|
| `zero` | **0.000** (6/6 byte-identical) | 0.0 |
| `positive` | 0.710 | 28.5178 |
| `negative` | 0.731 | 28.5178 |
| **`random_positive`** | **0.847** | 28.5178 |
| `random_negative` | 0.698 | 28.5178 |
| `unrelated_positive` | 0.681 | 28.5178 |

```
positive − random_control    = −0.137
positive − unrelated_feature = +0.029
```

**A scale-matched random direction moved the output further from baseline than
the real SAE feature direction did.** An unrelated real feature was
indistinguishable from the target. All conditions shared an identical delta
norm of 28.5178 by construction, so magnitude is fully controlled for and the
comparison is purely about direction.

**Conclusion: no evidence that feature 727's direction carries specific
behavioural meaning.** What was demonstrated is that *perturbing the residual
stream at layer 18 with sufficient magnitude changes the output* — which is
unsurprising and requires no SAE.

### Why this is the expected outcome, and what it does not mean

Most likely explanations, in order of my confidence:

1. **The SAE is undertrained.** 20k activations, measurable overfitting. Its
   directions may not be meaningfully different from arbitrary ones.
2. **The corpus is wrong for the question.** `wikitext` is generic encyclopedic
   prose; the base model is instruction-tuned. Features that steer *behaviour*
   would more plausibly emerge from instruction-formatted data.
3. **The selection rule was wrong.** Feature 727 was chosen by max activation on
   the wikitext corpus — nothing about behaviour entered the choice. The
   contrast-driven search in `patch_search.py` is implemented but was not the
   rule used here.
4. **The metric is coarse.** 3-gram divergence measures *that* output changed,
   not *how*. A real behavioural effect could be invisible to it.

What this does **not** establish: that activation steering does not work, or
that SAE features are never causally meaningful. This is one feature, one layer,
one undertrained SAE, six prompts. It is a null result at this scale, not
evidence of a general negative.

### Utility retention

10 hand-written probes, feature 727 at strength 16: 9/10 → 8/10. The single loss
was in factual QA (2/3 → 1/3); arithmetic, instruction-following and reasoning
were unchanged. Mean continuation length rose from 57.3 to 69.7 words.

One item on ten probes carries no statistical weight. Directionally consistent
with steering degrading unrelated capability; nothing stronger is supportable.

### Dynamic steering — works, and is verified precisely

Schedule `{0: 0.0, 24: 1.0, 48: 2.0}` at base strength 16, traced during a real
generation:

| generated token | measured delta norm |
|---|---|
| 0 – 23 | 0.0 |
| 24 | 28.5178 |
| 48 | 57.0356 |

Maximum absolute error against the predicted schedule: **3.6 × 10⁻⁶**.

The generated text visibly drifts once the schedule engages, beginning clean and
degrading into `"as described in Newton's law of..."` repetition after the
keyframe.

An implementation note worth keeping: the trace initially recorded only passes
where a delta was *applied*, so list index silently stopped matching token index
whenever the schedule was off. Now every pass is recorded as
`(generated_index, norm)`, including zeros.

---

## Costs

| stage | compute | wall time |
|---|---|---|
| model download | CPU | 64.6 s |
| L4 probe, model verify, tiny + full extraction, SAE training, intervention smoke, sweep, full experiment, dynamic demo | L4 | ~20 min total, including container starts and model loads |
| feature analysis, volume report | CPU | ~2 min |

Volume: 6080.24 MB total (5938.53 MB of it the Hugging Face cache).

**Actual spend, read from `modal billing summary`:**

```
Metered Cost:      0.11
  Ephemeral Apps:  0.10
  Volumes:         0.01
Credits:          -0.10
Free Storage:     -0.01
Billed Cost:      $0.00
```

**$0.11 metered** for the entire project — infrastructure, all GPU work,
`smoke_v0` end to end, and both Hugging Face uploads. That is 1.1% of the $10
budget, and it landed at $0.00 billed after credits.

What kept it there: CPU containers for the model download, feature analysis and
publishing; `retries=0` so no failed GPU job ever silently ran twice; a
2,000-token `tiny_v0` extraction to catch bugs before the real run; a 60-second
scaledown window; and no deployed demo.

---

## Unresolved scientific issues

1. **Does any SAE feature here carry causal behavioural meaning?** Unknown. The
   one tested does not, at this scale.
2. **Would 500k activations change the answer?** Untested. `serious_v1` is
   configured but not run (it exceeds the 50k unapproved token ceiling).
3. **Is `residual_post` at layer 18 the right site?** Chosen as a reasonable
   mid-depth default and validated to exist. Never compared against alternatives.
4. **Is 3-gram divergence the right effect measure?** It detects *that* output
   changed, not *what* changed. Log-probability differences on contrast pairs
   would be lower-variance per unit of compute.
5. **How much of the utility drop is the intervention versus generation noise?**
   Undetermined at n=10 with no repeats.
6. **Would contrast-driven selection find better candidates?** The machinery
   exists and is untested against the max-activation rule.

## What should be run next

In the order most likely to change the result:

1. Re-extract from an **instruction-formatted corpus** at the same token count.
   Cheap, and tests the most likely explanation for the null.
2. Contrast-driven candidate selection instead of max-activation ranking.
3. Log-probability measurement on the contrast fixtures rather than free
   generation only.
4. Only then scale to `serious_v1`. Scaling an SAE trained on the wrong
   distribution mostly buys a better model of the wrong thing.
