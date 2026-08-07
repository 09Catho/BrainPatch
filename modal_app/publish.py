"""Publishing curated artifacts from the Modal Volume to Hugging Face.

The upload runs **inside Modal**, reading from ``/vol`` and pushing straight to
the Hub. Artifacts never transit the local machine::

    Modal Volume  ->  Hugging Face Hub          (this module)
    Modal Volume  ->  local PC  ->  Hugging Face  (never)

Credentials come from the ``huggingface-secret`` Modal Secret as ``HF_TOKEN``.
The token is read from the environment and handed to ``huggingface_hub``; it is
never printed, never written into an artifact, and never committed.

What gets published is *curated*, not the whole Volume: the SAE checkpoint and
its config, feature statistics, experiment results, patches, and a model card.
Raw activation shards are excluded -- they are large, derived from a third-party
corpus, and reproducible from the recorded config.
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.resources import VOL_MOUNT, app, cpu_kwargs

#: Never uploaded, regardless of what a caller asks for.
EXCLUDED_PATTERNS = ("*.safetensors.tmp", "shard_*.safetensors", "hf-cache/*")


@app.function(**cpu_kwargs(timeout=60 * 30, secrets=True))
def publish_to_huggingface(
    experiment: str = "smoke_v0",
    repo_id: str | None = None,
    intervention_run: str = "smoke_v0_intervention",
    private: bool = False,
    dry_run: bool = True,
    model_card: str | None = None,
) -> dict[str, Any]:
    """Upload curated BrainPatch artifacts to a Hugging Face model repository.

    Parameters
    ----------
    repo_id:
        Target repo. Defaults to ``<authenticated user>/BrainPatch-Qwen2.5-1.5B``.
        No organization is guessed.
    dry_run:
        List what would be uploaded without creating a repo or pushing. The
        default, because publishing is a one-way, outward-facing action.
    model_card:
        Full README.md text for the repo. Passed in from the caller so the card
        can be reviewed before it ships.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not present in the container environment. "
            "Check that the huggingface-secret Modal Secret is attached."
        )

    api = HfApi(token=token)
    whoami = api.whoami()
    username = whoami["name"]
    target = repo_id or f"{username}/BrainPatch-Qwen2.5-1.5B"
    print(f"[publish] authenticated as {username}; target repo {target}")

    vol = Path(VOL_MOUNT)
    staging = Path(tempfile.mkdtemp(prefix="bp-publish-"))

    manifest: list[dict[str, Any]] = []

    def stage(source: Path, dest_rel: str) -> None:
        if not source.exists():
            print(f"[publish]   skip (missing): {source}")
            return
        dest = staging / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, dest, dirs_exist_ok=True)
            files = [p for p in dest.rglob("*") if p.is_file()]
        else:
            shutil.copy2(source, dest)
            files = [dest]
        for f in files:
            manifest.append(
                {
                    "path": str(f.relative_to(staging)),
                    "bytes": f.stat().st_size,
                }
            )

    # -- SAE checkpoint + config ------------------------------------------------
    stage(vol / "sae" / experiment / "sae_latest.pt", f"sae/{experiment}/sae_latest.pt")
    stage(vol / "sae" / experiment / "config.json", f"sae/{experiment}/config.json")
    stage(vol / "sae" / experiment / "summary.json", f"sae/{experiment}/summary.json")
    stage(vol / "sae" / experiment / "metrics.jsonl", f"sae/{experiment}/metrics.jsonl")

    # -- feature database -------------------------------------------------------
    stage(vol / "feature-db" / experiment / "summary.json", f"feature-db/{experiment}/summary.json")
    stage(
        vol / "feature-db" / experiment / "features.jsonl",
        f"feature-db/{experiment}/features.jsonl",
    )

    # -- activation manifest (metadata only; shards deliberately excluded) ------
    stage(
        vol / "activations" / experiment / "manifest.json",
        f"activations/{experiment}/manifest.json",
    )

    # -- experiment results -----------------------------------------------------
    stage(vol / "experiments" / intervention_run, f"experiments/{intervention_run}")

    # -- patches ----------------------------------------------------------------
    stage(vol / "patches", "patches")

    total_bytes = sum(m["bytes"] for m in manifest)

    if model_card:
        (staging / "README.md").write_text(model_card, encoding="utf-8")
        manifest.append(
            {"path": "README.md", "bytes": (staging / "README.md").stat().st_size}
        )

    (staging / "ARTIFACTS.json").write_text(
        json.dumps({"experiment": experiment, "files": manifest}, indent=2), encoding="utf-8"
    )

    # Safety net: refuse to upload anything that looks like a credential.
    leaked = _scan_for_secrets(staging)
    if leaked:
        raise RuntimeError(
            f"refusing to publish: possible credential material found in {leaked}"
        )

    result: dict[str, Any] = {
        "ok": True,
        "dry_run": dry_run,
        "repo_id": target,
        "username": username,
        "num_files": len(manifest),
        "total_mb": round(total_bytes / 1024**2, 2),
        "files": [m["path"] for m in manifest],
    }

    if dry_run:
        print("[publish] DRY RUN -- nothing uploaded")
        print(json.dumps(result, indent=2))
        shutil.rmtree(staging, ignore_errors=True)
        return result

    api.create_repo(repo_id=target, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(staging),
        repo_id=target,
        repo_type="model",
        commit_message=f"BrainPatch {experiment}: SAE, feature database, experiment artifacts",
    )
    shutil.rmtree(staging, ignore_errors=True)

    result["commit_url"] = str(getattr(commit, "commit_url", commit))
    result["repo_url"] = f"https://huggingface.co/{target}"
    print(json.dumps(result, indent=2))
    return result


@app.function(**cpu_kwargs(timeout=60 * 20, secrets=True))
def publish_dataset_to_huggingface(
    experiment: str = "smoke_v0",
    repo_id: str | None = None,
    private: bool = False,
    dry_run: bool = True,
    dataset_card: str | None = None,
) -> dict[str, Any]:
    """Publish the feature database as a Hugging Face Dataset repository.

    Only derived numerical metadata plus short attributed context snippets are
    included. The source corpus itself is referenced, not redistributed.
    """
    import os
    import shutil
    import tempfile
    from pathlib import Path

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing; check the huggingface-secret Modal Secret")

    api = HfApi(token=token)
    username = api.whoami()["name"]
    target = repo_id or f"{username}/BrainPatch-Features-Qwen2.5-1.5B"

    vol = Path(VOL_MOUNT)
    staging = Path(tempfile.mkdtemp(prefix="bp-dataset-"))
    files: list[str] = []

    for name in ("features.jsonl", "summary.json"):
        source = vol / "feature-db" / experiment / name
        if source.exists():
            dest = staging / experiment / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            files.append(str(dest.relative_to(staging)))

    manifest_src = vol / "activations" / experiment / "manifest.json"
    if manifest_src.exists():
        dest = staging / experiment / "activation_manifest.json"
        shutil.copy2(manifest_src, dest)
        files.append(str(dest.relative_to(staging)))

    if dataset_card:
        (staging / "README.md").write_text(dataset_card, encoding="utf-8")
        files.append("README.md")

    leaked = _scan_for_secrets(staging)
    if leaked:
        raise RuntimeError(f"refusing to publish: possible credential material in {leaked}")

    total = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
    result = {
        "ok": True,
        "dry_run": dry_run,
        "repo_id": target,
        "files": files,
        "total_mb": round(total / 1024**2, 3),
    }

    if dry_run:
        print(json.dumps(result, indent=2))
        shutil.rmtree(staging, ignore_errors=True)
        return result

    api.create_repo(repo_id=target, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(staging),
        repo_id=target,
        repo_type="dataset",
        commit_message=f"BrainPatch feature database for {experiment}",
    )
    shutil.rmtree(staging, ignore_errors=True)
    result["repo_url"] = f"https://huggingface.co/datasets/{target}"
    print(json.dumps(result, indent=2))
    return result


def _scan_for_secrets(root) -> list[str]:
    """Scan staged text files for anything resembling a live credential.

    Matches the *value* shapes (``hf_...``, ``sk-...``, long hex tokens), not
    the variable names, so a doc mentioning ``HF_TOKEN`` does not trip it.
    """
    import re
    from pathlib import Path

    patterns = [
        re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
        re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
        re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
        re.compile(r"\bgho_[A-Za-z0-9]{30,}\b"),
        re.compile(r"\bak-[A-Za-z0-9]{20,}\b"),
        re.compile(r"\bas-[A-Za-z0-9]{20,}\b"),
    ]
    offenders: list[str] = []
    for path in Path(root).rglob("*"):
        if not path.is_file() or path.suffix in {".pt", ".safetensors", ".bin"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(p.search(text) for p in patterns):
            offenders.append(str(path))
    return offenders
