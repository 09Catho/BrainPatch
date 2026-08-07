"""Building a viewer-compatible Hugging Face dataset release.

The first published layout put three files with unrelated schemas side by side
under one directory::

    smoke_v0/features.jsonl            one row per SAE feature
    smoke_v0/summary.json              a single aggregate object
    smoke_v0/activation_manifest.json  corpus provenance

The Hub's auto-detection globbed all three into one config and tried to cast
them to a common schema, which fails with ``StreamingRowsError`` /
``CastError``. The dataset was published but unbrowsable.

The fix has three parts:

1. **Parquet tables under ``data/``** hold the browsable rows.
2. **Explicit ``configs:`` in the dataset card** pin exactly which files each
   config reads, so auto-detection never runs and can never re-merge anything.
3. **Metadata moves to ``metadata/``** and the original JSONL to ``raw/``,
   outside every config's glob. Nothing is deleted; it is just no longer in the
   viewer's path.

Both tables are deliberately **flat** -- no nested structs, no lists. A
``list<struct>`` column for top-activating contexts would round-trip fine
through Parquet, but it renders as opaque JSON in the viewer and blocks the
search/filter/statistics features. Splitting contexts into their own row-per-
context table keeps both tables trivially browsable and joinable on
``feature_id``.
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.image import CPU_IMAGE
from modal_app.resources import VOL_MOUNT, app, cpu_kwargs

#: Directory layout of the published dataset repository.
FEATURES_PATH = "data/features/train-00000-of-00001.parquet"
CONTEXTS_PATH = "data/contexts/train-00000-of-00001.parquet"


def _features_schema():
    """Explicit Arrow schema for the per-feature table.

    Declared rather than inferred: ``hypothesis`` is null for every row at this
    stage, and pyarrow would infer ``null`` type from the data, which the Hub
    then cannot render as a string column.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("feature_id", pa.int32()),
            pa.field("fire_count", pa.int32()),
            pa.field("total_tokens", pa.int32()),
            pa.field("firing_rate", pa.float64()),
            pa.field("mean_activation", pa.float32()),
            pa.field("max_activation", pa.float32()),
            pa.field("std_activation", pa.float32()),
            pa.field("decoder_norm", pa.float32()),
            pa.field("is_dead", pa.bool_()),
            pa.field("num_top_contexts", pa.int32()),
            pa.field("top_activation", pa.float32()),
            pa.field("top_token", pa.string()),
            pa.field("top_context_preview", pa.string()),
            pa.field("hypothesis", pa.string()),
            pa.field("evidence_level", pa.string()),
        ]
    )


def _contexts_schema():
    """Explicit Arrow schema for the per-context table."""
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("feature_id", pa.int32()),
            pa.field("rank", pa.int32()),
            pa.field("activation", pa.float32()),
            pa.field("example_index", pa.int32()),
            pa.field("token_position", pa.int32()),
            pa.field("token_id", pa.int32()),
            pa.field("token_text", pa.string()),
            pa.field("context_before", pa.string()),
            pa.field("context_after", pa.string()),
            pa.field("context_preview", pa.string()),
        ]
    )


def _preview(before: str, token: str, after: str) -> str:
    """Render one context as a single readable string with the token marked."""
    return f"{before}[[{token}]]{after}"


@app.function(**cpu_kwargs(timeout=60 * 20, image=CPU_IMAGE, memory=4096))
def build_dataset_tables(experiment: str = "smoke_v0") -> dict[str, Any]:
    """Convert ``features.jsonl`` into two flat Parquet tables on the Volume.

    Writes to ``/vol/reports/dataset-release/<experiment>/`` so the source
    feature database is left untouched.
    """
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from brainpatch.paths import VolumePaths
    from modal_app.resources import volume

    paths = VolumePaths(VOL_MOUNT)
    source = Path(paths.features_jsonl(experiment))
    if not source.is_file():
        raise FileNotFoundError(f"feature database not found: {source}")

    feature_rows: dict[str, list] = {name: [] for name in _features_schema().names}
    context_rows: dict[str, list] = {name: [] for name in _contexts_schema().names}

    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            stats = record.get("stats", {})
            contexts = record.get("top_contexts", []) or []
            feature_id = int(record["feature_id"])

            top = contexts[0] if contexts else None
            feature_rows["feature_id"].append(feature_id)
            feature_rows["fire_count"].append(int(stats.get("fire_count", 0)))
            feature_rows["total_tokens"].append(int(stats.get("total_tokens", 0)))
            feature_rows["firing_rate"].append(float(stats.get("firing_rate", 0.0)))
            feature_rows["mean_activation"].append(float(stats.get("mean_activation", 0.0)))
            feature_rows["max_activation"].append(float(stats.get("max_activation", 0.0)))
            feature_rows["std_activation"].append(float(stats.get("std_activation", 0.0)))
            feature_rows["decoder_norm"].append(float(stats.get("decoder_norm", 0.0)))
            feature_rows["is_dead"].append(bool(stats.get("is_dead", False)))
            feature_rows["num_top_contexts"].append(len(contexts))
            feature_rows["top_activation"].append(
                float(top["activation"]) if top else 0.0
            )
            feature_rows["top_token"].append(str(top["token_text"]) if top else "")
            feature_rows["top_context_preview"].append(
                _preview(
                    str(top.get("context_before", "")),
                    str(top.get("token_text", "")),
                    str(top.get("context_after", "")),
                )
                if top
                else ""
            )
            # Stays null until a causal experiment supports a description.
            feature_rows["hypothesis"].append(record.get("hypothesis"))
            feature_rows["evidence_level"].append(str(record.get("evidence_level", "none")))

            for rank, context in enumerate(contexts):
                context_rows["feature_id"].append(feature_id)
                context_rows["rank"].append(rank)
                context_rows["activation"].append(float(context.get("activation", 0.0)))
                context_rows["example_index"].append(int(context.get("example_index", -1)))
                context_rows["token_position"].append(int(context.get("token_position", -1)))
                context_rows["token_id"].append(int(context.get("token_id", -1)))
                context_rows["token_text"].append(str(context.get("token_text", "")))
                context_rows["context_before"].append(str(context.get("context_before", "")))
                context_rows["context_after"].append(str(context.get("context_after", "")))
                context_rows["context_preview"].append(
                    _preview(
                        str(context.get("context_before", "")),
                        str(context.get("token_text", "")),
                        str(context.get("context_after", "")),
                    )
                )

    out_dir = Path(VOL_MOUNT) / "reports" / "dataset-release" / experiment
    (out_dir / "data" / "features").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "contexts").mkdir(parents=True, exist_ok=True)

    features_table = pa.Table.from_pydict(feature_rows, schema=_features_schema())
    contexts_table = pa.Table.from_pydict(context_rows, schema=_contexts_schema())
    pq.write_table(features_table, out_dir / FEATURES_PATH, compression="snappy")
    pq.write_table(contexts_table, out_dir / CONTEXTS_PATH, compression="snappy")

    volume.commit()

    result = {
        "ok": True,
        "experiment": experiment,
        "features_rows": features_table.num_rows,
        "features_columns": features_table.column_names,
        "features_bytes": (out_dir / FEATURES_PATH).stat().st_size,
        "contexts_rows": contexts_table.num_rows,
        "contexts_columns": contexts_table.column_names,
        "contexts_bytes": (out_dir / CONTEXTS_PATH).stat().st_size,
        "output_dir": str(out_dir),
    }
    print(json.dumps(result, indent=2))
    return result


@app.function(**cpu_kwargs(timeout=60 * 25, image=CPU_IMAGE, secrets=True, memory=4096))
def publish_dataset_release(
    experiment: str = "smoke_v0",
    repo_id: str | None = None,
    dry_run: bool = True,
    dataset_card: str | None = None,
) -> dict[str, Any]:
    """Upload the restructured dataset, removing the broken flat layout.

    The old ``smoke_v0/*`` paths are deleted in the same commit that adds the
    new structure, so the repository is never briefly in a state where both
    layouts are present and the viewer could pick the wrong one.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    from huggingface_hub import HfApi

    from brainpatch.paths import VolumePaths
    from modal_app.publish import _scan_for_secrets

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing; check the huggingface-secret Modal Secret")

    api = HfApi(token=token)
    username = api.whoami()["name"]
    target = repo_id or f"{username}/BrainPatch-Features-Qwen2.5-1.5B"

    vol = Path(VOL_MOUNT)
    paths = VolumePaths(VOL_MOUNT)
    release_dir = vol / "reports" / "dataset-release" / experiment
    if not (release_dir / FEATURES_PATH).is_file():
        raise FileNotFoundError(
            f"parquet tables not found at {release_dir}. Run build_dataset_tables first."
        )

    staging = Path(tempfile.mkdtemp(prefix="bp-dsrelease-"))

    # 1. browsable Parquet tables
    shutil.copytree(release_dir / "data", staging / "data")

    # 2. metadata, deliberately outside every config glob
    metadata_dir = staging / "metadata" / experiment
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for source, name in (
        (Path(paths.feature_summary(experiment)), "summary.json"),
        (Path(paths.activation_manifest(experiment)), "activation_manifest.json"),
        (Path(paths.sae_config(experiment)), "sae_config.json"),
    ):
        if source.is_file():
            shutil.copy2(source, metadata_dir / name)

    # 3. the original JSONL, preserved verbatim so nothing is lost
    raw_dir = staging / "raw" / experiment
    raw_dir.mkdir(parents=True, exist_ok=True)
    features_jsonl = Path(paths.features_jsonl(experiment))
    if features_jsonl.is_file():
        shutil.copy2(features_jsonl, raw_dir / "features.jsonl")

    if dataset_card:
        (staging / "README.md").write_text(dataset_card, encoding="utf-8")

    leaked = _scan_for_secrets(staging)
    if leaked:
        raise RuntimeError(f"refusing to publish: possible credential material in {leaked}")

    files = sorted(str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())

    result: dict[str, Any] = {
        "ok": True,
        "dry_run": dry_run,
        "repo_id": target,
        "files": files,
        "total_mb": round(total / 1024**2, 3),
        "deletes": ["smoke_v0/*"],
    }

    if dry_run:
        print(json.dumps(result, indent=2))
        shutil.rmtree(staging, ignore_errors=True)
        return result

    api.create_repo(repo_id=target, repo_type="dataset", private=False, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(staging),
        repo_id=target,
        repo_type="dataset",
        commit_message=(
            "Restructure for the dataset viewer: Parquet tables under data/, "
            "metadata and raw JSONL moved out of the config globs"
        ),
        # Removes the original flat layout whose mixed schemas broke the viewer.
        delete_patterns=["smoke_v0/*"],
    )
    shutil.rmtree(staging, ignore_errors=True)

    result["commit_url"] = str(getattr(commit, "commit_url", commit))
    result["repo_url"] = f"https://huggingface.co/datasets/{target}"
    print(json.dumps(result, indent=2))
    return result
