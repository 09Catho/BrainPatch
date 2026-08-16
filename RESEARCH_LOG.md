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

### Extreme position-0 activation outlier, measured

The hook smoke test on `"The capital of France is"` reported a mean activation
norm of 2267 across 5 tokens with the **first token at 11052**. Excluding
position 0 gives ~70.8, which matches the corpus mean of ~70 implied
independently by the measured `input_scale` (`√1536 / 0.5611 = 69.85`).

Position 0 is therefore ~156× the typical activation norm. Extraction drops it
(`skip_first_n_tokens = 1`); including it would have dominated the input-scale
normalisation and spent dictionary capacity on a single positional artifact.
Two independent measurements agreeing on ~70 is a useful consistency check on
the whole extraction-and-normalisation path.

**On the mechanism.** An earlier draft of this log called this an "attention
sink". That was an unearned inference: the literature commonly attributes
first-token residual outliers to attention-sink behaviour, and it is a plausible
explanation here, but **no attention weights were measured in this project**.
What was measured is an activation-norm outlier at position 0. The wording has
been corrected throughout the repository to say only that.

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

Measured L0 32.0 -- note L0 is bounded above by `k`, not identically `k`; a
later audit confirmed 0 zero-valued Top-K selections in this run. **0 dead
features of 2048.**

**Observation.** The 0.104 explained-variance gap between train and validation
is genuine overfitting. 19,000 training rows against a 2048-feature dictionary
is far too little data — roughly 9 rows per feature. Zero dead features at this
scale is not a sign of health either; with so few examples every feature can
find something to fire on. Both are expected consequences of the corpus size and
neither should be read as SAE quality.

### Dose-response sweep

Strength is not something to guess: the right magnitude depends on the
residual-stream norm at the hooked layer. Measured on feature 727, 3 prompts,
64 new tokens. The divergence column is the **3-prompt mean**; the note column
describes **prompt #0 only**, the one whose generations the sweep records
verbatim. Those are different measurements and the first write-up presented
them adjacently without saying so.

| strength | delta norm | mean divergence (3 prompts) | prompt #0 |
|---|---|---|---|
| 2 | 3.565 | 0.434 | byte-identical to baseline |
| 4 | 7.129 | 0.443 | byte-identical to baseline |
| 8 | 14.259 | 0.563 | first wording change |
| 16 | 28.518 | 0.770 | rewritten, coherent, still correct |
| 32 | 57.036 | 0.958 | looping |
| 64 | 114.071 | 1.000 | total collapse |

Against a raw residual norm of ~70, strength 16 is a ~41% perturbation. The
initial `intervention_smoke` at strength 2.0 produced **byte-identical** output
to baseline on that same prompt — a 5% perturbation is simply too small to
change greedy decoding there. That is why the sweep exists rather than a
guessed default.

**Prompt sensitivity varies enormously, and the table half-hides it.** At
strength 2 prompt #0 contributed exactly 0.000 divergence, yet the 3-prompt mean
was 0.434. Arithmetically the other two prompts averaged about 0.65 — a large
change — at the same perturbation that left prompt #0 untouched. So "below
strength 8 the output was unchanged" is true of prompt #0 and false of the
sweep as a whole. Both the README and the model card now label the aggregation
explicitly, because side by side the two numbers read as a contradiction.

This is also a caution about the sweep design: recording only
``generations[0]`` makes the cheapest-to-read evidence the least
representative.

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
the real SAE feature direction did.** All conditions shared an identical delta
norm of 28.5178 by construction, so magnitude is fully controlled for and the
comparison is purely about direction.

**Conclusion: no evidence that feature 727's direction carries specific
behavioural meaning.** What was demonstrated is that *perturbing the residual
stream at layer 18 with sufficient magnitude changes the output* — which is
unsurprising and requires no SAE.

> **Retraction (added during the post-release quality pass).** The
> `unrelated_positive` condition was **not** an unrelated feature. Feature 1270
> was selected by the same `max_activation` ranking as the target and is a
> near-duplicate of it. That comparison is uninformative and is retracted; see
> the root-cause section below. The random-direction control is unaffected and
> the headline conclusion rests on it.

### Root cause found: the selection rule picked a degenerate cluster

Found by reading the published feature table, which only became inspectable once
the dataset viewer was fixed. This supersedes the earlier speculation.

| | feature 727 | feature 1270 (the "control") | dictionary |
|---|---|---|---|
| fire count | 5 / 20,000 | 3 / 20,000 | median 271, mean 312.5 |
| firing rate | 0.00025 | 0.00015 | mean 0.015625 |
| max activation | 1429.77 | 1385.83 | median 9.06 |
| top token | `" Bd"` | `" Bd"` | — |

The top **32** features by `max_activation` all fire on 3–6 tokens and all share
the top token `" Bd"` — chess notation appearing in a handful of wikitext
articles. An undertrained SAE shatters a few rare, high-norm tokens across many
near-duplicate features, and `max_activation` ranking finds exactly that cluster.

Two consequences:

1. The intervention target was the dictionary's single most extreme outlier,
   firing on 0.025% of tokens — about as unrepresentative a direction as the
   dictionary contains.
2. The "unrelated feature" control was drawn from the same ranking and landed on
   a near-duplicate. It was never a control.

So `smoke_v0` is better read as **"this selection rule picks degenerate
features"** than as **"SAE feature directions carry no behavioural meaning"**.
The two are very different claims and the earlier write-up conflated them.

`rank_features()` now accepts `min_firing_rate` and carries an explicit warning
about this failure mode.

### Remaining explanations, re-ranked

1. **The selection rule** — now demonstrated, not speculated. Fixing it requires
   no new extraction and no new SAE.
2. **The corpus is wrong for the question.** `wikitext` is generic encyclopedic
   prose; the base model is instruction-tuned.
3. **The SAE is undertrained.** 20k activations, measurable overfitting — and
   the ` Bd` shattering is itself a symptom of exactly that.
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

One item on ten probes carries no statistical weight, and **this sample cannot
establish degradation**. The correct reading is that the intervention *may*
affect unrelated capabilities and that a properly-powered experiment should
check; the observed direction is not evidence in itself.

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

---

## Post-release quality pass

Four corrections after the first publication, none requiring new experiments.

### The dataset viewer was broken on arrival

`09Catho/BrainPatch-Features-Qwen2.5-1.5B` returned
`preview: false, viewer: false` from the datasets-server. The first layout put
three files with unrelated schemas in one directory:

```
smoke_v0/features.jsonl            one row per feature
smoke_v0/summary.json              a single aggregate object
smoke_v0/activation_manifest.json  corpus provenance
```

Auto-detection globbed all three into one config and failed to cast them to a
common schema (`StreamingRowsError` / `CastError`). Published, but unbrowsable.

Restructured into two **flat** Parquet tables — `features` (2048 rows) and
`contexts` (16,238 rows, joinable on `feature_id`) — with explicit `configs:` in
the dataset card pinning exactly which files each config reads, so
auto-detection never runs again. Metadata moved to `metadata/` and the original
JSONL preserved verbatim at `raw/smoke_v0/features.jsonl`, both outside every
config glob.

Contexts were deliberately *not* kept as a `list<struct>` column on the feature
table. Parquet round-trips that fine, but the viewer renders it as opaque JSON
and it blocks search, filter and statistics. Two flat tables cost one join and
make both fully browsable.

### The documented quickstart did not run, then ran wrong

The published example downloaded the SAE checkpoint from the Hub and then called
`model.install("patches/experimental-feature-727.json")` against a **local
relative path that a new user would not have**. It could never have worked from
a clean environment.

Fixed to fetch both artifacts via `hf_hub_download`, and added
`verify_model_card_example` — a Modal function that executes the published
snippet verbatim in a fresh container, deliberately reading from the Hub rather
than from `/vol` so a broken upload fails the check.

Running it exposed a second, worse problem. `set_patch_strength` is a
**multiplier** on the patch's declared strength, not an absolute value. The
example used `1.5` against a patch declaring `16.0`, giving an effective
coefficient of **24** — a ~34% residual perturbation. Measured output:

```
baseline:  "The sum of 17 and 25 is 42."
patched:   "17 + 25 = 32 ... This is an example of the commutative property
            of multiplication ..."
```

A wrong arithmetic answer and a confused digression, shipped as the flagship
usage example. Corrected to `1.0` (effective 16, delta norm 28.5178), re-verified
on Modal: answer 42, coherent, `zero_strength_matches_baseline: true`. The
failure case is now documented in the card as a warning rather than deleted,
because it is a useful measurement of how little headroom there is.

### Two overclaims in the wording

**"interventions damage unrelated capabilities"** — 9/10 versus 8/10 on ten
hand-written probes is a one-item difference and cannot establish degradation in
either direction. Changed throughout to "may affect unrelated capabilities",
with the sample size stated inline.

**"position 0 is an attention sink"** — the measurement was an activation-norm
outlier (11052 against a corpus mean of ~70). Attributing it to attention-sink
behaviour is a mechanism claim, and **no attention weights were measured in this
project**. Changed throughout to "position 0 exhibited an extreme
residual-stream activation outlier", with the attention-sink explanation
mentioned only as a plausible-but-unverified hypothesis.

Both were inferences that ran ahead of the data. Recording them here because the
same failure mode — a measurement quietly promoted into a mechanism — is exactly
what the evidence ladder elsewhere in this project exists to prevent.

---

## Correctness pass: Top-K liveness accounting

### The bug

`torch.topk` always returns exactly `k` indices. Because the SAE applies ReLU
*before* Top-K, some of those selected values can be zero whenever a token has
fewer than `k` positive encoder pre-activations.

`update_liveness` counted **every selected index** as a firing:

```python
fired[indices.reshape(-1)] = True                    # zero-valued ones included
counts = torch.bincount(indices.reshape(-1), ...)    # zero-valued ones counted
self.tokens_since_fired[fired] = 0                   # clock reset for non-firings
```

Consequences, in increasing severity:

1. `fire_count` inflated by the number of zero-valued selections.
2. `tokens_since_fired` reset for features that did not activate.
3. Therefore `dead_mask()` under-reports dead features — **a permanently silent
   feature that Top-K keeps padding into its selection would be reported alive
   forever**, never flagged dead, and never revived by AuxK.

(3) is the one that matters. The dead-feature mechanism exists precisely to
catch features that stop firing, and this bug could blind it to exactly those.

### Audit of what was and was not affected

| Surface | Affected? | Why |
|---|---|---|
| `fire_count` buffer | yes, in principle | counted zero-valued selections |
| `tokens_since_fired` / `dead_mask` | yes, in principle | clock reset by non-firings |
| AuxK | indirectly | consumes `dead_mask`; its own zero-scatter is a numerical no-op |
| reported `l0` | **no** | already computed as `feature_acts > 0`, which is correct |
| reconstruction / loss | **no** | zeros scattered into a zeros tensor change nothing |
| feature database (`features.jsonl`) | **no** | built from `acts > 0`, never from `update_liveness` |

### Did it affect the persisted smoke_v0 run? No.

Determined empirically by `audit_topk_liveness`, not by argument:

```
training evidence   89 logged steps, min l0 = max l0 = 32.0 = k
corpus evidence     640,000 Top-K selections over all 20,000 stored activations
                    zero-valued selections:            0
                    rows with fewer than k positive:   0
                    min positive selections in a row:  32
verdict             NOT AFFECTED
```

Every Top-K selection in the run had a strictly positive value, so the buggy and
fixed accounting produce **identical** numbers here. This is unsurprising at
`d_sae=2048, k=32`: fewer than 32 of 2048 pre-activations would have to be
positive, and roughly half are. **All published smoke_v0 metrics remain valid
exactly as reported and none were changed.**

The checkpoint's stored `fire_count` total equals `k × tokens_seen`
(36,372,480 = 32 × 1,136,640). That is expected under the *old* code regardless,
so it is not evidence either way — the corpus re-encode is what settles it.

### The fix

`update_liveness` now takes `values` alongside `indices` and filters on
`values > 0`; passing the old single-argument form raises. `SAEOutput` gains
`active_mask()` and `l0()`, and `reconstruction_metrics` now also reports
`l0_min`, `l0_max` and `zero_selection_rate` — so if this condition ever does
arise, it appears in the training log instead of having to be inferred.

Docstrings claiming "exactly `k` non-zeros by construction" were wrong and are
corrected to "at most `k`" throughout, including the README and model card. The
*measured* L0 of exactly 32.0 stays, because it is a measurement.

15 regression tests in `tests/remote/test_sae_liveness.py` pin the behaviour,
driving the SAE into the fewer-than-k-positive regime deliberately (which needs
a rigged encoder — it does not occur naturally at realistic width). They run
inside Modal via `sae_unit_tests` because they need torch, which the local suite
blocks by design.

---

## Correctness pass: evidence terminology

The ladder's top rung was `causal`, defined as "steering changes behaviour and
scale-matched controls do not". That definition describes an *experimental
outcome*; the label asserts *causation*, which a single experiment on one model
at one layer with one prompt set does not establish.

Replaced with:

```
none → correlational → predictive → interventional
     → controlled_interventional → replicated
```

`controlled_interventional` names what was actually done (controls ran and
passed, with adequate power). `replicated` requires independent repetition.
`is_validated` is now True only at `replicated`; `has_controlled_evidence`
covers the top two rungs.

**No patch was retroactively promoted.** Both published patches remain
`evidence_level: none`, which is what their controls support — and `"causal"` is
now rejected by schema validation, so old files naming it fail loudly rather
than loading with a silently unrecognised label.

---

## Backend verification: llama.cpp and vLLM

Both moved from *implemented* to *verified* against real engines. Several
upstream behaviours had to be discovered empirically rather than assumed, and
each would have produced a wrong or fake-looking result if guessed.

### llama.cpp (upstream b10344, 7a20b417f)

Verified on a real 1.12 GB **Q4_K_M** GGUF (`Qwen/Qwen2.5-1.5B-Instruct-GGUF`),
driving the upstream `llama-cli` binary -- no fork.

| check | result |
|---|---|
| layer mapping: BrainPatch L18 (0-based) to `direction.19` (1-based) | pass |
| scale 0 matches baseline | pass (character-identical generation) |
| non-zero scale changes output | pass |
| layer range honoured | pass |
| no crash / no model corruption | pass |

Three things upstream does that the first implementation got wrong:

1. **`--control-vector-scaled` takes one `FNAME:SCALE` token**, not two
   arguments. Passing them separately fails with
   `control-vector-scaled format: FNAME:SCALE`.
2. **`-no-cnv` alone does not terminate the process.** llama-cli stayed in
   interactive mode printing `> ` forever against EOF stdin -- indistinguishable
   from a hang, and it cost several runs to diagnose. `-st/--single-turn` is
   what makes it exit.
3. **stdout is full of non-deterministic chrome**: an animated loading spinner,
   an ASCII logo, and a trailing `[ Prompt: 262.4 t/s | Generation: 47.9 t/s ]`
   line. Comparing raw stdout reported baseline and scale-0 as *different* when
   their generations were character-identical and only the measured throughput
   differed. The test now extracts just the answer.

Also worth recording: the GGUF must be staged to container-local disk.
llama.cpp mmaps the model, and mmap against the Volume's network filesystem
turns every page fault into a network round trip.

**Quantization honesty.** The direction still changes output at Q4_K_M. Whether
it produces the *same* behavioural effect as at bf16 is untested, and the patch
manifest says exactly that.

### vLLM 0.11.0

| check | result |
|---|---|
| hooks installed **inside the vLLM worker process** | pass |
| scale 0 matches baseline | pass |
| non-zero scale changes output | pass |
| batched requests agree with single requests | pass |
| OpenAI server: two concurrent requests, no state leak | pass |
| mismatched per-request strength rejected with 400 | pass |

The worker itself reports `Qwen2ForCausalLM`, 28 layers, `active_hooks: 1`,
`hooked_layers: [18]`, `cuda:0`, `torch.bfloat16`. That report is the evidence
the intervention runs inside vLLM rather than in a substituted Transformers
model -- output changes alone could not distinguish the two.

Two constraints shaped the design:

* **V1 runs the model in a separate process**, so attribute traversal off the
  `LLM` object cannot reach it.
* **The RPC channel is msgpack and refuses to serialize a callable**, suggesting
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1`. That would turn the engine's control
  channel into an arbitrary-code path for our convenience, so it was not used.
  The supported `worker_extension_cls` mechanism calls methods **by name** with
  msgpack-safe arguments instead.

Also fixed: the OpenAI server returned `422 Field required` for every POST,
because `from __future__ import annotations` plus locally-defined Pydantic
models left FastAPI unable to resolve the body type, silently demoting it to a
query parameter. This is the second time postponed annotations broke a framework
that introspects them at runtime (the first was `modal.parameter`).

**Throughput was not measured reliably on vLLM.** The server benchmark ran the
patched condition first and the baseline second, and vLLM's prefix cache makes a
second pass over identical prompts much faster. The resulting "+82.7% overhead"
is an artifact of ordering, not a property of the patch, and is deliberately not
reported as an overhead figure. The Transformers benchmark (-1.3%, within noise)
was ordered correctly and stands.

---

## Anti-sycophancy attempt: negative, with a much better method

Stage A of the staged plan: reuse the `smoke_v0` SAE and search it with a
behaviour-specific objective. Total cost well under a dollar.

### What was done differently from the first experiment

* **Objective**: paired log-probability margin,
  `log P(independent | prompt) - log P(sycophantic | prompt)`, per token -- not
  n-gram divergence, which rewards *any* perturbation and is how a random
  direction beat the real feature last time.
* **Splits by topic**, not by row, so no topic appears in more than one split.
  20 hand-written items: 8 train / 5 validation / 7 test.
* **Screening**: 2013 of 2048 features passed a firing-rate band of
  [0.002, 0.30], deliberately excluding the rare-token cluster that invalidated
  the feature-727 result. Then cosine deduplication at 0.6, leaving 6 candidates
  with corpus firing rates 0.0101-0.0171 -- right around the dictionary median
  of 0.0136, not the pathological tail.
* **Strength calibrated to each feature's own p90 activation**, so the
  intervention stays on-manifold rather than relying on a huge off-distribution
  coefficient.
* **Controls chosen properly**: three unrelated SAE features screened to cosine
  < 0.15 against the target *and* against each other, plus three scale-matched
  random directions.

### Result

| stage | outcome |
|---|---|
| train (n=8) | feature 1848, standardised effect -0.463, corpus firing rate 0.0160 |
| validation (n=5) | best config f1848 at coefficient +10.66: mean delta **+0.187**, win rate 0.60 |
| **held-out test (n=7)** | mean delta **-0.0021**, bootstrap 95% CI **[-0.154, +0.156]** |

**NEGATIVE.** The validation improvement did not replicate on the topic-disjoint
held-out split, and the confidence interval spans zero. The most likely
explanation is that +0.187 was selection noise over five validation examples --
which is exactly what a held-out split exists to catch.

The patch is published as `experimental-independent-criticism-candidate` at
`evidence_level: correlational`: the train-split activation difference is a real
correlation, and the behavioural claim is not supported. It does **not** get the
name `anti-sycophancy`.

### A bug that would have faked a different negative

The first Stage A run reported *exactly* `+0.0000` delta for every candidate at
every strength. That was not a result: log-probability scoring calls the model
directly for a single forward pass, while the backend attaches its hooks inside
`generate()`. No intervention was ever applied.

A zero delta for every condition looks exactly like "the feature does nothing",
which is a conclusion one might well have published. `score_examples` now
installs the hooks explicitly and raises if patches are installed but no hooks
are attached, so the failure cannot recur silently.

### What to try next

1. **An instruction-formatted activation corpus.** The SAE was trained on
   wikitext -- generic encyclopedic prose -- while sycophancy is an
   instruction-following property. This is the leading explanation for the null
   and is Stage B of the plan (~50k tokens, d_sae ~4096, one L4).
2. **More held-out items.** Seven examples can fail to demonstrate an effect;
   they cannot rule one out.
3. Only then consider scaling the dictionary.


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

Re-ordered after the selection-rule root cause was found.

1. **Re-run the intervention experiment on the existing SAE with a sane feature
   selection** — features near the median firing rate (~271 fires / 20,000), or
   contrast-driven selection from `patch_search.py`. Requires **no new
   extraction and no new SAE training**: every artifact already exists on the
   Volume. Cheapest possible experiment, and it directly tests the leading
   explanation for the null. Also needs a genuinely unrelated control feature,
   chosen from a different firing-rate band than the target.
2. Re-extract from an **instruction-formatted corpus** at the same token count,
   so features can plausibly relate to behaviour at all.
3. Log-probability measurement on the contrast fixtures rather than free
   generation only — lower variance per unit of compute.
4. Statistical power: many prompts, repeated sampling, actual significance
   testing.
5. Only then consider `serious_v1`. Scaling an SAE whose features are selected
   by a broken rule, on the wrong corpus, mostly buys a better model of the
   wrong thing.

---

# `anti_sycophancy_v1` — a properly powered attempt, and a negative result

Full write-up and raw artifacts:
[`experiments/anti_sycophancy_v1/`](experiments/anti_sycophancy_v1/).
Criteria were [pre-registered](experiments/anti_sycophancy_v1/success_criteria.md)
and committed before the test split was scored.

## What changed from the previous attempt

The earlier anti-sycophancy run had 20 items, no true-assertion control, and one
discovery method. Every one of those was load-bearing:

- **198 distinct propositions**, 134 false and 64 true, 13 categories, 78 train / 42 validation / 78 test. No proposition is restated in a second wording, so the item count is a count of independent observations and the bootstrap interval means what it says.
- **True-assertion controls** in every split. Without them a direction that simply increases disagreement scores as increased independence — which is exactly what both SAE variants turned out to do.
- **Four discovery methods** on identical splits: difference-of-means (CAA), PCA, a linear probe, and SAE features single and sparse.
- **1,035 configurations** scanned on validation: 7 layers × 3 extraction positions × 3 injection sites × 5 strengths, calibrated to the residual norm the model naturally carries at each layer.

## The baseline justified the dataset

Qwen2.5-1.5B-Instruct preferred the sycophantic continuation on **31 of 53**
false-claim test items (mean per-token margin −0.129) while correctly preferring
agreement on true ones (+0.239). The behaviour is present and asymmetric in the
way the target describes; the dataset is not measuring an already-solved problem.

## Result

The selected direction — PCA, layer 24, extraction `cont_mean`, injection on
prompt tokens, strength 10% of median residual norm — produced on the held-out
test split:

- `Δ_false` **+0.0744**, CI [+0.0637, +0.0850], **94%** of items improved, *d* = 1.84
- `Δ_true` **+0.0391**, CI [+0.0085, +0.0717] — not contrarian
- beat the best of **10** scale-matched random directions (+0.0305) and all three unrelated real directions
- reversed under sign inversion (−0.0477)
- cost **2.02%** perplexity on neutral text

And failed anyway, on two pre-registered criteria:

- **length-gap correlation +0.457**, over the 0.3 threshold. The dataset's desired responses are longer in ~96% of pairs, and per-item gain tracks that gap. Per-token normalisation, the two-polarity design and random controls were all insufficient to separate the effect from a preference for longer text.
- **free generation moved the wrong way**: correction rate 5.7% → 3.8%. A 0.074 nat/token log-probability preference does not change what a greedy decoder emits.

The shuffled-label control also failed, though not in the dangerous direction:
its effect was significantly *negative* (−0.011) rather than zero, most likely
because permuting labels leaves the length asymmetry partly intact. A pipeline
manufacturing the target effect from noise would have produced a *positive*
result there.

## Three findings that outlive the null

1. **The SAE was the worst method tested.** Beaten by PCA, probe and CAA, and both SAE variants failed the true-claim guard. This project began as an SAE product; the honest measurement says the dictionary is not what makes steering work here.
2. **Probe accuracy is not steerability.** 100% training separation, mediocre steering. Mean probe accuracy across the grid was 0.949 with essentially no relationship to effect.
3. **Injection site dominates.** Prompt-token steering was ~6× stronger than generated-token steering (+0.44 vs +0.067 max). The intervention that llama.cpp and vLLM can both express is the more effective one.

## What would settle it

Length-balanced response pairs first — that single authoring change is what
would make the disqualifying criterion informative instead of fatal. Then a
stance judge instead of a keyword rubric (the rubric scores "Yes, you're
correct! …the product is indeed a positive number" as agreement when the content
corrects the user), and a second model to say whether any of this is a property
of the behaviour or of Qwen2.5-1.5B.

## Also corrected in this round

The vLLM throughput benchmark reported "+82.7% throughput from patching", which
was never published. Four separate defects: cache-warming across conditions, one
sample per condition, fixed condition order, and — the real one — the two
conditions generating different amounts of text, because the patch changes the
output and completions stop at EOS at different lengths. Normalised by generated
tokens, the honest figure is **+2.15% overhead** (60.74 → 59.43 tok/s).

---

# `anti_sycophancy_v2` — the confound is fixed, and the metric turns out to be the problem

Full write-up: [`experiments/anti_sycophancy_v2/`](experiments/anti_sycophancy_v2/).
Pre-registered before any test access. **Negative result; the test split was
never opened and is still unscored.**

## The dataset fix worked

v1 died on a length confound: the preferred response was longer in ~96% of pairs
and per-item steering gain correlated **+0.457** with that gap. v2 authored 387
fresh propositions — **zero** shared topics or assertions with v1 — each pair
written to a declared length polarity and then trimmed to near-parity.

| | v1 | v2 |
|---|---|---|
| preferred longer | 96% | **53%** |
| mean gap | one-directional | **+0.28 tokens** |
| class predicts length | +0.10 | **+0.023** |
| result-level `corr(Δ, gap)` | +0.457 (disqualifying) | mean **0.129** over 330 configurations |

The audit is a **gate**, not a report: `scripts/build_sycophancy_v2.py` refuses
to emit the dataset if it fails, and a token-level audit runs on Modal before
any activation is captured. Both fired during development — the token audit
caught that my own criterion (`|median gap| / mean length ≤ 0.05`) was
unsatisfiable for an integer gap on ~14-token continuations, which is recorded
as Amendment 1 along with the fact that it is a relaxation in ratio terms.

## The finding

330 configurations, five methods, 27 survivors of the true-claim guard. Free
generation on every survivor:

```
corr(normalized Δ_false, free-generation correction gain) = −0.298
configurations improving generation:                        7 / 27
best correction gain:                                      +0.068  (from a WEAK log-prob config)
top 8 by log-prob effect →  mean correction gain  −0.023
top 8 by correction gain →  mean log-prob effect  +0.069
```

**Ranking directions by paired log-probability anti-selects for generation
improvement.** The pre-registered shortlist rule ranks by `Δ_false`, so it
sampled exactly the configurations least likely to pass the generation gate, and
none of the six did — which closed the test split.

This retro-explains v1 completely. v1's winner had a strong log-prob effect
(+0.074 on test, CI excluding zero, beating ten random directions, three
unrelated directions and reversing under sign flip) and a **falling** correction
rate, 5.7% → 3.8%. That was not bad luck; it is what this correlation predicts.

## Method ordering changed; two lessons held

**CAA > PCA > probe > SAE single > SAE sparse.** Difference-of-means wins on the
clean dataset where PCA won on the confounded one, and takes 21 of 27 survivors.
What replicated:

- **Probe accuracy is not steerability.** Up to **100%** predictive accuracy (mean 0.937) at half CAA's steering strength.
- **SAE is last**, in both experiments.
- **Injection site dominates**: prompt +0.355, prompt+generated +0.306, generated-only **−0.0002**. Steering generated tokens does essentially nothing.

## Most "working" directions are just contrarian

Only 27 of 330 configurations passed the true-claim guard. The strongest
log-prob movers have `Δ_true` of −0.24 (CAA) and −0.46 (PCA): they make the
model disagree with **true** statements too. Without true-assertion controls
those would have been the headline results.

## What v3 should do

Select on generation from the start. The useful directions live in the *weak*
tail of the log-prob ranking, which no log-probability-first search will
surface. The best candidate seen (CAA, layer 16, `last_prompt`, prompt-site,
ratio 0.35; validation correction 18.2% → 25.0%) is a **hypothesis, not a
result** — promoting it here would be exactly the post-hoc selection the
protocol exists to prevent. It should be a pre-registered prediction in a fresh
experiment with its own test split.

Modal spend for v2: **$0.31**; project total **$1.37** of $10.

---

# `anti_sycophancy_v3` — the first positive result, and its one weak joint

Full write-up: [`experiments/anti_sycophancy_v3/`](experiments/anti_sycophancy_v3/).
Pre-registered before test access, with the minimum effect size calibrated to
measured baseline variance. **All 11 gates passed.**

## What changed

v2 showed that ranking candidates by continuation log-probability
*anti-selects* for generated behaviour (`corr = −0.298`). v3 therefore selects
on **generated behaviour** and demotes log-probability to a diagnostic. Cheap
metrics only filter; ≤30 candidates (declared in advance) reach generation; the
top 5 are confirmed on the full validation split.

Fresh dataset: 550 propositions, 11 categories, 200/150/200, 40% true-claim
controls, **zero** topic or assertion overlap with v1 or v2.

## Result

**SAE feature 204, layer 18, `last_prompt` extraction, prompt-token injection,
0.35 of the median residual norm.**

| | baseline | patched |
|---|---|---|
| correction rate (false claims) | **0.233** | **0.400** |
| SIS | +0.221 | +0.338 |
| false disagreement (true claims) | 0.013 | 0.062 |
| correct agreement (true claims) | 0.500 | 0.613 |

+0.167 absolute (+71.4% relative), CI [+0.092, +0.242]. Paired over the same 120
items: **22 improved, 2 worsened, 96 unchanged**, McNemar **p = 3.6 × 10⁻⁵**.
Length +0.21%, degeneration 0.000, utility 32/32 both conditions, zero refusals.
The zero-strength control reproduced the baseline character-for-character.

## The weak joint

The best of ten norm-matched **random** directions scored **+0.158**. The real
direction scored **+0.167**. That is about **one item in 120**, and the random
null spans **−0.133 to +0.158**. G4 only asks that the real direction beat the
maximum random control, and it does — but an intervention of this size moves the
correction rate substantially *whichever way you push*. The effect is well
measured; the direction-specificity is not. Ten random controls is too few to
estimate a null that wide, and that is the first thing a follow-up should fix.

G3 also landed exactly on its threshold: true-claim false disagreement rose from
1/80 to 5/80, precisely the pre-registered +0.05 limit.

## The PatchBench finding: the ranking inverts

```
by log-probability steering (v1, v2):  CAA / PCA  >  probe  >  SAE   (SAE last)
by free-generation behaviour (v3):     SAE  >  probe  >  PCA  >  CAA (CAA last, +0.000)
```

The winning direction's own log-prob diagnostic on test is **−0.0136**: the
patch that improves generation by 71% makes the paired log-probability margin
slightly *worse*. `corr(log-prob, generation gain)` was **+0.163** here against
**−0.298** in v2 — not stable in sign, never large. The honest conclusion is not
"log-probability is negatively predictive" but **"log-probability is not
predictive"**:

```
representation quality ≠ log-prob steerability ≠ behavioural usefulness
```

A linear probe hit **1.000** predictive accuracy and delivered **+0.011** on
full validation.

## The subset nearly picked the wrong winner

Three of five finalists collapsed between the 60-item ranking subset and the
full 150-item validation split (probe +0.167 → +0.011, PCA +0.139 → +0.000).
Only the SAE candidates survived. Without the confirmation step v3 would have
frozen a configuration worth +0.011.

## A format gap the result exposed

The validated configuration is **prompt-token-only** injection, which the
`.brainpatch` v1 format could not express: a schedule cannot separate the prompt
pass from generated token 0, since they share an index. Interventions now carry
an explicit `site` field (default `all`, so every existing patch is unchanged).

llama.cpp binds a control vector for a whole run and vLLM shares one forward
pass across a batch, so **neither can honour `site: prompt`**. The patch records
them as `unsupported` rather than claiming cross-backend support it does not
have. Applying it there would steer every token — a configuration with no test
evidence behind it.

## Shipped

`anti-sycophancy.brainpatch`, **7,382 bytes**, `evidence_level:
controlled_interventional`, discovery method `sae_single`. Its own README leads
with the random-control caveat.

Transformers is marked `implemented` rather than `verified`: the behaviour was
measured on that backend in stage C, but the artifact-level reproduction check
(compiled file through `backend.generate`) did not complete within budget, and
the stronger word is reserved for checks that finish.

Modal spend for v3: **$0.66**; project total **$2.03** of $10.
