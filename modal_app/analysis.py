"""Feature-database construction and reporting.

Runs on **CPU**. Encoding 20k activations through a 2048-feature dictionary is
two matrix multiplies; paying GPU rates for that would be waste. The tokenizer
is loaded (also CPU) so that top-activating contexts can be decoded back into
readable text.

Nothing in here assigns a semantic label to a feature. See
:mod:`brainpatch.ml.feature_analysis`.
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.resources import VOL_MOUNT, app, cpu_kwargs, volume

#: The analysis image needs torch (for the SAE) but never a GPU. Reuse the ML
#: image on a CPU container rather than maintaining a third image.
from modal_app.image import ML_IMAGE  # noqa: E402


@app.function(**cpu_kwargs(timeout=60 * 30, image=ML_IMAGE, cpu=4, memory=8192))
def analyze_features(
    experiment: str = "smoke_v0",
    top_k_contexts: int = 8,
    context_window: int = 12,
    with_tokenizer: bool = True,
) -> dict[str, Any]:
    """Build ``/vol/feature-db/<experiment>/features.jsonl`` from a trained SAE."""
    import torch

    from brainpatch.ml.activation_store import ActivationSubset
    from brainpatch.ml.feature_analysis import build_feature_database
    from brainpatch.ml.sae import TopKSAE
    from brainpatch.paths import VolumePaths

    paths = VolumePaths(VOL_MOUNT)
    checkpoint_path = str(paths.sae_checkpoint(experiment))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sae = TopKSAE.from_checkpoint(checkpoint, device="cpu")

    input_scale = sae.config.input_scale
    if input_scale is None:
        raise ValueError(f"SAE checkpoint {checkpoint_path} has no recorded input_scale")

    subset = ActivationSubset.load(paths, experiment, dtype=torch.float32)

    tokenizer = None
    if with_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            sae.config.model, revision=sae.config.model_revision or None
        )

    summary = build_feature_database(
        sae,
        subset,
        paths,
        experiment,
        input_scale=float(input_scale),
        top_k_contexts=top_k_contexts,
        context_window=context_window,
        tokenizer=tokenizer,
    )
    volume.commit()
    print(json.dumps(summary, indent=2))
    return summary


@app.function(**cpu_kwargs(timeout=600, image=ML_IMAGE))
def top_features(
    experiment: str = "smoke_v0",
    by: str = "max_activation",
    limit: int = 20,
    max_firing_rate: float = 0.5,
) -> list[dict[str, Any]]:
    """List the highest-ranked features with their top contexts."""
    from brainpatch.ml.feature_analysis import rank_features
    from brainpatch.paths import VolumePaths

    paths = VolumePaths(VOL_MOUNT)
    records = rank_features(
        paths, experiment, by=by, limit=limit, max_firing_rate=max_firing_rate
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        contexts = [
            {
                "token": c.token_text,
                "activation": round(c.activation, 3),
                "context": f"{c.context_before}[[{c.token_text}]]{c.context_after}",
            }
            for c in record.top_contexts[:3]
        ]
        rows.append(
            {
                "feature_id": record.feature_id,
                "max_activation": round(record.stats.max_activation, 3),
                "mean_activation": round(record.stats.mean_activation, 3),
                "firing_rate": round(record.stats.firing_rate, 5),
                "fire_count": record.stats.fire_count,
                "evidence_level": record.evidence_level,
                "top_contexts": contexts,
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return rows


@app.function(**cpu_kwargs(timeout=600))
def volume_report() -> dict[str, Any]:
    """Inventory of what is stored on the Volume, with sizes."""
    from pathlib import Path

    root = Path(VOL_MOUNT)
    report: dict[str, Any] = {"root": str(root), "trees": {}}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        files = [p for p in child.rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        report["trees"][child.name] = {
            "num_files": len(files),
            "total_mb": round(total / 1024**2, 2),
            "entries": sorted(p.name for p in child.iterdir())[:25],
        }
    report["total_mb"] = round(sum(t["total_mb"] for t in report["trees"].values()), 2)
    print(json.dumps(report, indent=2))
    return report

