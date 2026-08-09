---
license: apache-2.0
pretty_name: BrainPatch feature database (Qwen2.5-1.5B-Instruct, layer 18)
size_categories:
  - 1K<n<10K
tags:
  - mechanistic-interpretability
  - sparse-autoencoder
  - interpretability
  - activation-steering
configs:
  - config_name: features
    default: true
    data_files:
      - split: train
        path: data/features/train-*.parquet
  - config_name: contexts
    data_files:
      - split: train
        path: data/contexts/train-*.parquet
---

# BrainPatch feature database — Qwen2.5-1.5B-Instruct

Per-feature statistics for a Top-K sparse autoencoder trained on layer 18
(`residual_post`) of a frozen
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

- **GitHub:** https://github.com/09Catho/BrainPatch
- **Model / SAE checkpoint:** https://huggingface.co/09Catho/BrainPatch-Qwen2.5-1.5B

## Loading

```python
from datasets import load_dataset

features = load_dataset("09Catho/BrainPatch-Features-Qwen2.5-1.5B", "features", split="train")
contexts = load_dataset("09Catho/BrainPatch-Features-Qwen2.5-1.5B", "contexts", split="train")

print(features[727])
print(contexts.filter(lambda r: r["feature_id"] == 727)["context_preview"])
```

The two configs join on `feature_id`.

## Configs

### `features` — 2048 rows, one per SAE feature (default)

| Column | Type | Meaning |
|---|---|---|
| `feature_id` | int32 | Index into the SAE dictionary |
| `fire_count` | int32 | Tokens on which the feature was active |
| `total_tokens` | int32 | Tokens analysed (20,000) |
| `firing_rate` | float64 | `fire_count / total_tokens` |
| `mean_activation` | float32 | Mean over **firing tokens only** |
| `max_activation` | float32 | Largest activation observed |
| `std_activation` | float32 | Std over **firing tokens only** |
| `decoder_norm` | float32 | L2 norm of the decoder column (constrained to 1) |
| `is_dead` | bool | Never fired over the corpus |
| `num_top_contexts` | int32 | Contexts recorded in the `contexts` config |
| `top_activation` | float32 | Activation of the strongest context |
| `top_token` | string | Token text of the strongest context |
| `top_context_preview` | string | Strongest context, token marked `[[like this]]` |
| `hypothesis` | string | Candidate description. **Null for every row.** |
| `evidence_level` | string | Evidence behind `hypothesis`. `"none"` for every row. |

### `contexts` — one row per top-activating occurrence

| Column | Type | Meaning |
|---|---|---|
| `feature_id` | int32 | Joins to the `features` config |
| `rank` | int32 | 0 = strongest activation for that feature |
| `activation` | float32 | Activation value |
| `example_index` | int32 | Index into the extraction corpus |
| `token_position` | int32 | Position within that token block |
| `token_id` | int32 | Tokenizer id |
| `token_text` | string | The activating token |
| `context_before` / `context_after` | string | ±12 tokens either side |
| `context_preview` | string | Rendered as `before[[token]]after` |

Both tables are flat — no nested structs — so the viewer, search, filter and
statistics all work.

## Files outside the configs

These are preserved for provenance and are deliberately **not** part of any
config, because their schemas differ from the feature table and merging them is
what broke the viewer in the first published version:

| Path | What |
|---|---|
| `metadata/smoke_v0/summary.json` | Aggregate dictionary statistics |
| `metadata/smoke_v0/activation_manifest.json` | Corpus provenance, shard index, seeds |
| `metadata/smoke_v0/sae_config.json` | SAE architecture and training configuration |
| `raw/smoke_v0/features.jsonl` | The original nested JSONL, verbatim |

Nothing was lost in the restructure; `raw/smoke_v0/features.jsonl` is the exact
file the Parquet tables were built from.

## Dictionary statistics

| | |
|---|---|
| Features | 2048 (d_in 1536, expansion 1.33×) |
| Top-K | k=32 (measured L0 32.0; L0 is bounded by k, not equal to it) |
| Tokens analysed | 20,000 |
| Alive / dead | 2048 / 0 |
| Mean firing rate (alive) | 0.015625 |
| Median firing rate (alive) | 0.01355 |
| Max firing rate | 0.0766 |
| Decoder norms | mean 1.0, min 0.9999992, max 1.0000007 |
| `input_scale` | 0.5610531069008018 |

`mean_activation` and `std_activation` are computed over **firing tokens only**.
Averaging in a Top-K SAE's structural zeros would report `k/d_sae` times the
true magnitude.

## Scientific status

**Every `hypothesis` is null and every `evidence_level` is `"none"`.**

Top-activating contexts are *correlational* evidence. A feature whose top
examples look thematically coherent is a feature that correlates with that theme
in this corpus — it is not "the X feature" until steering it changes behaviour
and scale-matched controls do not.

In the accompanying intervention experiment the **control failed**: a
scale-matched random direction of identical L2 norm moved the model's output
further from baseline (0.847) than the tested feature direction did (0.710).
Treat these statistics as descriptive only.

The SAE is also deliberately undertrained — 20,000 activations against a
2048-feature dictionary, train explained variance 0.762 versus validation 0.658.
It exists to prove the pipeline, not to produce a good dictionary.

### Read this before ranking by `max_activation`

This table contains a documented pathology, and it is visible directly in the
viewer:

```
sort by max_activation, descending
```

The top **32** features all fire on **3–6 tokens out of 20,000** and all share
the top token `" Bd"` — chess notation from a handful of wikitext articles —
at activations above 1200 against a dictionary median of **9.06**. The SAE has
shattered a few rare, high-norm tokens across many near-duplicate features,
which is a characteristic symptom of training on too little data.

| | feature 727 | dictionary |
|---|---|---|
| `fire_count` | 5 | median 271, mean 312.5 |
| `max_activation` | 1429.77 | median 9.06 |

The published intervention experiment selected its target *and* its
"unrelated feature" control from this cluster, which is why that control was
retracted. **Filter on `firing_rate` before using this table to choose
intervention candidates.** Features near the median firing rate are far more
likely to be representative.

See the research log on GitHub for the full account.

## Corpus and licensing

Statistics are derived from
[`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext)
(`wikitext-2-raw-v1`, train split), CC BY-SA 3.0, derived from Wikipedia.

The corpus is **not** redistributed here. The `contexts` config carries short
attributed snippets (±12 tokens around each activating token) for human
inspection; everything else is derived numerical metadata.

This repository: Apache-2.0.
