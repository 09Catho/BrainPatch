---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
  - mechanistic-interpretability
  - sparse-autoencoder
  - activation-steering
  - interpretability
  - qwen2
library_name: brainpatch
---

# BrainPatch — Qwen2.5-1.5B-Instruct

**A Top-K sparse autoencoder over layer 18 of a frozen Qwen2.5-1.5B-Instruct, plus the runtime for injecting its feature directions back into the residual stream.**

GitHub: **https://github.com/09Catho/BrainPatch**

---

## Read this first

This repository contains **verified infrastructure with a negative behavioural result.**

The pipeline works end to end and is reproducible. The behavioural claim does not hold: in the `smoke_v0` intervention experiment, steering the selected SAE feature moved the model's output away from baseline — but a **scale-matched random direction of identical L2 norm moved it further**.

There is no evidence here that any feature direction carries specific behavioural meaning. That is why the shipped patches are named `experimental-feature-727.json` and not `anti-sycophancy.json`.

---

## What this is

BrainPatch applies small activation-space interventions to a **frozen** language model. Nothing is fine-tuned; the base weights are never touched.

```
frozen LLM
   +  residual-stream activations   (layer 18, post-block)
   +  Top-K sparse autoencoder      (d_sae 2048, k 32)
   +  feature directions            (unit-norm decoder columns)
   +  runtime intervention hooks    (add / ablate, with schedules)
```

An intervention adds, at the hooked layer:

```
delta_raw = strength × unit_decoder_column / input_scale
```

`input_scale` normalises activations so `E[||x||₂] = √d_in`. Dividing by it maps the direction back to the raw residual stream, which is what makes `strength` mean the same physical thing across SAEs. For this SAE, `input_scale = 0.5610531069008018`.

## What this is **not**

- Not evidence that SAE features correspond to human concepts.
- Not a claim that steering demonstrates beliefs, intentions, or any mental property.
- Not a source of validated behavioural labels. Every feature description is a hypothesis and is marked as one.
- Not free of side effects. Capability probes dropped from 9/10 to 8/10 at the strength used.
- Not portable to other models, layers, or SAEs. The format refuses mismatched application.

---

## Contents

| Path | What |
|---|---|
| `sae/smoke_v0/sae_latest.pt` | SAE weights, optimizer state, liveness buffers, config |
| `sae/smoke_v0/config.json` | Architecture and training configuration |
| `sae/smoke_v0/metrics.jsonl` | Per-step training metrics |
| `feature-db/smoke_v0/features.jsonl` | Per-feature statistics and top-activating contexts |
| `activations/smoke_v0/manifest.json` | Corpus provenance (metadata only — no shards) |
| `experiments/smoke_v0_intervention/` | All generations, metrics, and the report |
| `patches/` | BrainPatch JSON files |

The Qwen base weights are **not** duplicated here. Load them from [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

Raw activation shards are not published: 58.8 MB derived from a CC BY-SA corpus, fully reproducible from the recorded config.

---

## Usage

```python
from huggingface_hub import hf_hub_download
from brainpatch import BrainPatchedModel

checkpoint = hf_hub_download(
    "09Catho/BrainPatch-Qwen2.5-1.5B", "sae/smoke_v0/sae_latest.pt"
)

model = BrainPatchedModel.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
)
model.load_sae(checkpoint, reference="smoke_v0")

model.install("patches/experimental-feature-727.json")
model.set_patch_strength("experimental-feature-727", 1.5)

print(model.generate("Solve this problem..."))
```

Ad-hoc single-feature steering:

```python
model.add_feature(layer=18, feature_id=727, strength=16.0)
```

Dynamic mid-generation steering, keyed on **generated**-token index:

```python
from brainpatch.steering import StrengthSchedule

model.set_patch_schedule("experimental-feature-727", StrengthSchedule({0: 0.0, 24: 1.0, 48: 2.0}))
```

---

## Modal reproduction

Every experiment ran on [Modal](https://modal.com) on a single **NVIDIA L4**. No local GPU is required, and no model weights or activations are ever downloaded to a developer machine.

```bash
modal run modal_app/app.py::smoke_pipeline
```

Stage by stage:

```bash
modal run modal_app/app.py::cache_model
modal run modal_app/app.py::extract_activations --experiment smoke_v0 --target-tokens 20000
modal run modal_app/app.py::train_sae --experiment smoke_v0 --d-sae 2048 --k 32 --epochs 60
modal run modal_app/app.py::analyze_features --experiment smoke_v0
modal run modal_app/app.py::intervention_experiment --experiment smoke_v0 --strength 16
```

---

## Experimental results

All figures below were measured. None are estimates.

### Environment

| | |
|---|---|
| GPU | NVIDIA L4, compute capability 8.9, 22.03 GB VRAM |
| Stack | torch 2.6.0+cu124, CUDA 12.4, transformers 4.51.3 |
| GPU correctness | matmul max abs error vs CPU reference: 0.0 |
| Base model | Qwen2.5-1.5B-Instruct @ `989aa798…`, hidden 1536, 28 layers |

### Extraction

20,000 tokens from layer 18 (`residual_post`), sequence length 256, in 8.367 s → **2390.4 tokens/s**, at **3084.01 bytes/token** (= 1536 × 2 bf16 + 12 bytes int32 metadata). Peak VRAM 3553.4 MB.

Position 0 is excluded: its measured activation norm at layer 18 is **11052 against a corpus mean of ~70**, a factor of 156. It is an attention sink, and including it would dominate the input-scale normalisation.

### SAE training

| | |
|---|---|
| Architecture | d_in 1536, d_sae 2048 (1.33× expansion), k 32, 6,295,040 params |
| Training | 2220 steps / 60 epochs in 78.6 s → 28.229 steps/s |
| Peak VRAM | 295.4 MB |
| Train | explained variance **0.762**, cosine 0.925 |
| Validation | explained variance **0.658**, cosine 0.890 |
| L0 | exactly 32.0 |
| Dead features | 0 of 2048 |
| Decoder norms | mean 1.0, min 0.9999992, max 1.0000007 |

The **0.104 train/validation explained-variance gap is genuine overfitting.** 19,000 training rows against a 2048-feature dictionary is roughly 9 rows per feature. Zero dead features at this scale is not a sign of health either. `smoke_v0` exists to prove the pipeline, not to produce a good SAE.

### Dose–response

Residual-stream L2 norm at layer 18 is ~70 raw units.

| strength | delta norm | divergence from baseline | degenerate |
|---|---|---|---|
| 2 | 3.565 | 0.434 | no |
| 8 | 14.259 | 0.563 | no |
| 16 | 28.518 | 0.770 | no |
| 32 | 57.036 | 0.958 | yes |
| 64 | 114.071 | 1.000 | yes |

Below strength 8 the greedy output on the probe prompt was unchanged.

### Intervention experiment — the headline

Feature 727 vs unrelated feature 1270, strength ±16, 6 prompts, greedy decoding.

| condition | divergence from baseline | delta norm |
|---|---|---|
| `zero` | **0.000** (6/6 byte-identical) | 0.0 |
| `positive` | 0.710 | 28.5178 |
| `negative` | 0.731 | 28.5178 |
| **`random_positive`** | **0.847** | 28.5178 |
| `random_negative` | 0.698 | 28.5178 |
| `unrelated_positive` | 0.681 | 28.5178 |

```
positive − random_control    = −0.137   ← wrong sign
positive − unrelated_feature = +0.029   ← negligible
```

All conditions share an **identical** injected delta norm by construction (decoder columns and random directions are both unit-norm, through the same coefficient path), so magnitude is fully controlled and the comparison is purely about direction.

### Utility retention

10 hand-written probes, feature 727 at strength 16: **9/10 → 8/10**. The loss was in factual QA (2/3 → 1/3); arithmetic, instruction-following and reasoning unchanged. Mean continuation length rose 57.3 → 69.7 words.

### Dynamic steering

Schedule `{0: 0.0, 24: 1.0, 48: 2.0}` at base strength 16, traced during a real generation:

| generated token | measured delta norm |
|---|---|
| 0 – 23 | 0.0 |
| 24 | 28.5178 |
| 48 | 57.0356 |

Maximum absolute error against the predicted schedule: **3.6 × 10⁻⁶**.

---

## Scientific status

| Claim | Status |
|---|---|
| The pipeline runs end to end and is reproducible | **Verified** |
| `strength = 0` is byte-identical to baseline | **Verified** (6/6 prompts, 0 applied passes) |
| Injected delta norm equals `\|strength\| / input_scale` | **Verified** to 7 significant figures |
| Token-indexed schedules fire at the correct index | **Verified** (error 3.6 × 10⁻⁶) |
| Decoder columns hold unit norm | **Verified** (min 0.9999992, max 1.0000007) |
| Perturbing layer 18 with sufficient magnitude changes output | **Verified** — and unsurprising; needs no SAE |
| Feature 727's *direction* carries specific behavioural meaning | **Not supported.** Controls failed. |
| Any feature here maps to a human concept | **Not tested, not claimed** |

The evidence ladder used throughout: `none` → `correlational` → `predictive` → `interventional` → `causal`. Nothing advances past `correlational` automatically. Both published patches are `none`.

---

## Limitations

1. **The behavioural result is negative.** Scale-matched controls did not separate from the intervention.
2. **The SAE is undertrained by design** — 20k activations, measurable overfitting.
3. **The corpus is wrong for the goal.** `wikitext` is generic prose; the model is instruction-tuned. Features that steer *behaviour* would more plausibly emerge from instruction-formatted data. This is the most likely explanation for the null.
4. **Sample sizes are tiny.** 6 prompts, 10 utility probes, one greedy generation per condition, no repeated sampling, no significance testing. These are not effect sizes.
5. **The selection rule was not behavioural.** Feature 727 was chosen by max activation on wikitext; nothing about behaviour entered the choice.
6. **The effect metric is coarse.** 3-gram divergence detects *that* output changed, not *what* changed.
7. **Model-free degeneration metrics are heuristics** and demonstrably missed one looping generation before being corrected.
8. **SAE features may be polysemantic.** Feature entanglement is expected.
9. **One layer, one model, one hook site.** No cross-layer or cross-model claims are made or supported.
10. **Results depend on model revision and generation configuration.** Both are pinned and recorded.

Activation steering does not demonstrate human-like mental properties, and nothing in this repository should be read as evidence that it does.

---

## Reproducibility

| | |
|---|---|
| GitHub | https://github.com/09Catho/BrainPatch |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Base model revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Hook | layer 18, `residual_post` |
| SAE | d_in 1536, d_sae 2048, k 32, `input_scale` 0.5610531069008018 |
| Seed | 0 (Python, NumPy, torch, dataset sampling, SAE init) |
| Corpus | `Salesforce/wikitext` / `wikitext-2-raw-v1`, train split, 20,000 tokens |
| Packages | torch 2.6.0, transformers 4.51.3, datasets 3.5.0, accelerate 1.6.0, numpy 2.1.3 |
| Config | `configs/experiments/smoke_v0.yaml` |

**Known non-determinism:** GPU training is only approximately reproducible — cuDNN kernel selection and floating-point atomics make bitwise-identical reruns unlikely across containers or drivers. All experimental generation is greedy, which removes sampling variance from every comparison.

---

## Attribution and licensing

Base model: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), Apache-2.0. Not redistributed here.

Corpus: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) (`wikitext-2-raw-v1`), CC BY-SA 3.0, derived from Wikipedia. Only derived numerical artifacts and short attributed context snippets are published; the corpus itself is referenced, not redistributed.

Method influences: sparse dictionary learning on transformer activations, and the Top-K SAE formulation with AuxK dead-feature revival.

This repository: Apache-2.0.
