# Modal infrastructure

All compute, all model weights, all data. The local machine only submits jobs.

## Resources

| | |
|---|---|
| Environment | `brainpatch-dev` |
| Volume | `brainpatch-data`, mounted at `/vol` |
| Secret | `huggingface-secret`, exposes `HF_TOKEN` |
| GPU | `L4`, single, always |

```bash
brainpatch modal status
```

## Volume layout

```
/vol/
├── hf-cache/                     HF model + dataset cache (HF_HOME points here)
├── datasets/                     preprocessed corpora
├── activations/<experiment>/     manifest.json, examples.jsonl, shard_*.safetensors
├── sae/<experiment>/             sae_latest.pt, config.json, metrics.jsonl, summary.json
├── feature-db/<experiment>/      features.jsonl, summary.json
├── patches/                      published patch files
├── experiments/<experiment>/     config, all generations, metrics.json, report.md
└── reports/
```

`brainpatch.paths.VolumePaths` is the single source of truth for these paths and
is used identically on both sides.

## Images

Three, built by `_build(*packages)` in `modal_app/image.py`, which always adds
local Python sources **last**:

| Image | Contents | Used by |
|---|---|---|
| `CPU_IMAGE` | pyyaml, typer, rich, huggingface_hub, numpy, safetensors | volume bookkeeping, model download, publishing |
| `ML_IMAGE` | torch 2.6.0, transformers 4.51.3, datasets 3.5.0, accelerate 1.6.0 | extraction, training, interventions, feature analysis |
| `WEB_IMAGE` | `ML_IMAGE` packages + gradio, fastapi | the demo |

Images may not be extended after `add_local_*`, so each is composed from its own
package list rather than by appending to another. Versions are pinned and
recorded in every experiment's provenance block via `pinned_versions()`.

`HF_HUB_ENABLE_HF_TRANSFER` is set to `0` deliberately — its parallel
range-writes fail against the Volume's network filesystem under gVisor.

## Cost controls

The project runs under a hard ~$10 budget, so the defaults are conservative and
several limits are enforced in code rather than by convention:

- `gpu_kwargs()` **raises** on A100, H100, H200, B200, L40S, or any multi-GPU
  request. Escalation is a human decision, not a code path.
- `retries=0` on every GPU function. An automatic retry of an expensive job is a
  silent doubling of cost; failures should be read and fixed.
- `scaledown_window=60` so containers do not idle warm.
- `assert_token_budget()` refuses extraction above **50,000 tokens** unless
  `approved=True` is passed explicitly. It is called at the entry point, so no
  code path can bypass it.
- CPU functions are used wherever a GPU is not required: model download, feature
  analysis, volume reports, Hugging Face publishing.

## Entry points

Every command below was executed against this repository and verified to resolve.

```bash
modal run modal_app/app.py::cpu_smoke
```

```bash
modal run modal_app/app.py::gpu_info
```

```bash
modal run modal_app/app.py::cache_model
```

```bash
modal run modal_app/app.py::model_architecture
```

```bash
modal run modal_app/app.py::verify_cached_model --layer 18
```

```bash
modal run modal_app/app.py::extract_activations --experiment smoke_v0 --target-tokens 20000
```

```bash
modal run modal_app/app.py::train_sae --experiment smoke_v0 --d-sae 2048 --k 32 --epochs 60
```

```bash
modal run modal_app/app.py::analyze_features --experiment smoke_v0
```

```bash
modal run modal_app/app.py::top_features --experiment smoke_v0
```

```bash
modal run modal_app/app.py::intervention_smoke --experiment smoke_v0
```

```bash
modal run modal_app/app.py::sweep_strength --experiment smoke_v0
```

```bash
modal run modal_app/app.py::intervention_experiment --experiment smoke_v0 --strength 16
```

```bash
modal run modal_app/app.py::dynamic_steering_demo --experiment smoke_v0 --strength 16
```

```bash
modal run modal_app/app.py::volume_report
```

```bash
modal run modal_app/app.py::smoke_pipeline
```

The demo is served, not deployed:

```bash
modal serve modal_app/web.py
```

## Credential handling

`HF_TOKEN` reaches only containers that declare `secrets=[hf_secret]`. It is
never hardcoded, never printed, never written to the Volume, and never included
in a published artifact. `modal_app/publish.py` additionally scans every staged
file for credential-shaped strings (`hf_…`, `sk-…`, `ghp_…`, `ak-…`) and refuses
to upload if any match.

## Checkpointing and resume

Both expensive stages resume by default; `--force` is required to discard.

**Extraction** rewrites `manifest.json` after every shard and calls
`volume.commit()`. Shards are immutable once recorded. A resumed run
re-validates model, layer, hook, dtype, sequence length and dataset against the
manifest and refuses to append incompatible activations to an existing corpus.

**SAE training** checkpoints weights, optimizer state, step, epoch, liveness
buffers and RNG state every 200 steps, written to a temp file and renamed so a
crash cannot truncate a good checkpoint. Resume re-validates `d_in`, `d_sae` and
`k`.
