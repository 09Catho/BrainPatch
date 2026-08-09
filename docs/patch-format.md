# The BrainPatch file format (v0.1)

A BrainPatch is a JSON document describing an activation-space intervention on
one model, at one layer, under one SAE. It is small enough to read, diff, and
review by hand — which is the point.

## Full example

```json
{
  "format_version": "0.1",
  "name": "experimental-feature-727",
  "description": "Single-feature steering direction from the smoke_v0 SAE. ...",
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
  "sae": {
    "reference": "smoke_v0",
    "layer": 18,
    "hook": "residual_post",
    "d_in": 1536,
    "d_sae": 2048,
    "input_scale": 0.5610531069008018,
    "sha256": null
  },
  "features": [
    { "feature_id": 727, "strength": 16.0, "mode": "add" }
  ],
  "schedule": { "0": 0.0, "24": 1.0, "48": 2.0 },
  "evaluation": { "...": "measured numbers, or {} for 'not evaluated'" },
  "evidence_level": "none",
  "license": "Apache-2.0",
  "authors": [],
  "metadata": {}
}
```

## Fields

| Field | Required | Meaning |
|---|---|---|
| `format_version` | yes | Must be a version this build supports (`0.1`). |
| `name` | yes | `^[a-z0-9][a-z0-9._-]{0,63}$`. Lowercase, no spaces. |
| `base_model` | yes | Hugging Face id. Enforced at install time. |
| `model_revision` | no | Commit SHA. Enforced under `strict_revision=True`. |
| `sae` | yes | Identity of the dictionary the feature IDs index into. |
| `features` | yes | Non-empty list of feature edits, unique IDs. |
| `schedule` | no | Generated-token index → strength multiplier. |
| `evaluation` | no | Measured results. `{}` means *not evaluated*, never *it works*. |
| `evidence_level` | no | See the evidence ladder below. Defaults to `none`. |
| `license`, `authors`, `metadata` | no | Provenance. |

### `features[]`

| Field | Meaning |
|---|---|
| `feature_id` | Index into the SAE dictionary. Must be `< d_sae`. |
| `strength` | Coefficient in normalised activation-space units. Negative suppresses. |
| `mode` | `add` (inject the direction) or `ablate` (remove the feature's measured contribution). |

### What `strength` physically means

The runtime adds:

```
delta_raw = strength × unit_decoder_column / input_scale
```

`input_scale` normalises activations so `E[||x||₂] = √d_in`. Dividing by it maps
the direction back to the raw residual stream. So `strength = 1.0` adds one unit
of normalised activation-space distance, regardless of how large the residual
stream happens to be at that layer.

For `smoke_v0` (`input_scale = 0.5611`, raw residual norm ≈ 70):

| strength | injected L2 norm | fraction of residual norm |
|---|---|---|
| 2 | 3.565 | ~5% |
| 16 | 28.518 | ~41% |
| 64 | 114.071 | ~163% |

Measured behaviour across that range is in the README's dose–response table.

### `schedule`

Keys are **generated**-token indices — the prompt is not counted. Semantics are
step-hold: the value at index `n` is the value of the largest keyframe `≤ n`.

```
{"0": 0.0, "24": 1.0, "48": 2.0}

token:     0 ... 23   24 ... 47   48 ...
multiplier: 0.0        1.0         2.0
```

The multiplier scales every `features[].strength` in the patch.

## The evidence ladder

`evidence_level` is the field that stops a patch's *name* from becoming a claim.

| Level | Requires |
|---|---|
| `none` | nothing |
| `correlational` | top-activating contexts look suggestive |
| `predictive` | activation predicts a behaviour on held-out data |
| `interventional` | steering changes behaviour; controls absent, incomplete, or not yet run |
| `controlled_interventional` | steering changes behaviour **and** scale-matched controls did not, in one adequately-powered experiment |
| `replicated` | that controlled result held up on independent repetition |

`spec.has_controlled_evidence` is True from `controlled_interventional` upward.
`spec.is_validated` is True only at `replicated` — one passing controlled
experiment is a result, not a validation.

Nothing advances a patch past `correlational` automatically. Both patches
shipped in this repository are `none`, because their controls came back
negative — see [`../RESEARCH_LOG.md`](../RESEARCH_LOG.md).

A patch named `honesty.json` at `evidence_level: none` would be dishonest. A
patch named `experimental-feature-1207.json` is not.

## Compatibility enforcement

`BrainPatchSpec.check_compatibility()` raises `PatchCompatibilityError` on:

- a different `base_model`
- a hidden size ≠ `sae.d_in`
- `sae.layer` beyond the model's depth
- a different SAE reference or dictionary size
- a revision mismatch, when `strict_revision=True`

This is a hard refusal, not a warning. Feature directions are properties of one
specific set of weights at one specific layer under one specific dictionary.
Applying a Qwen patch to Gemma is not degraded performance — it is adding an
arbitrary vector to an unrelated coordinate system, and the result would look
like output, not like an error.

`BrainPatchSpec.validate()` separately rejects malformed files: unknown format
versions, bad names, empty or duplicate feature lists, out-of-range IDs, unknown
modes, unknown evidence levels, and malformed schedules.

## Validating a file

```bash
brainpatch validate patches/experimental-feature-727.json
```

```bash
brainpatch inspect patches/experimental-feature-727.json
```

```bash
brainpatch compare patches/experimental-feature-727.json patches/experimental-feature-727-scheduled.json
```
