"""Remote SAE unit tests and an artifact audit of the Top-K liveness bug.

Two CPU functions, neither of which is a research experiment:

``sae_unit_tests``
    Runs ``tests/remote/`` inside Modal. Those tests need torch, which the local
    suite blocks, so this is where they execute. No GPU, no model, no Volume
    data -- just SAE arithmetic on tiny tensors.

``audit_topk_liveness``
    Answers one specific question about already-persisted artifacts: *did the
    liveness bug actually corrupt the smoke_v0 run?* Read-only. It re-derives
    the answer from the training metrics log and from a deterministic re-encode
    of the stored activation corpus with the stored checkpoint, rather than
    arguing from first principles.
"""

from __future__ import annotations

import json
from typing import Any

from modal_app.image import ML_IMAGE, TEST_IMAGE
from modal_app.resources import VOL_MOUNT, app, cpu_kwargs


@app.function(**cpu_kwargs(timeout=60 * 15, image=TEST_IMAGE, cpu=2, memory=4096))
def sae_unit_tests(verbose: bool = True) -> dict[str, Any]:
    """Run the remote pytest suite (``tests/remote/``) in a Modal container."""
    import os
    import subprocess
    import sys

    args = [sys.executable, "-m", "pytest", "/root/tests/remote", "-p", "no:cacheprovider"]
    args.append("-v" if verbose else "-q")

    # tests/conftest.py hides tests/remote/ and installs a torch import blocker
    # unless this is set. Both behaviours are correct locally and wrong here.
    env = {**os.environ, "BRAINPATCH_REMOTE_TESTS": "1"}
    completed = subprocess.run(args, capture_output=True, text=True, cwd="/root", env=env)
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)

    result = {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "summary": completed.stdout.strip().splitlines()[-1] if completed.stdout else "",
    }
    if completed.returncode != 0:
        raise RuntimeError(f"remote SAE tests failed:\n{completed.stdout}\n{completed.stderr}")
    print(json.dumps(result, indent=2))
    return result


@app.function(**cpu_kwargs(timeout=60 * 20, image=ML_IMAGE, cpu=4, memory=8192))
def audit_topk_liveness(experiment: str = "smoke_v0") -> dict[str, Any]:
    """Determine whether the Top-K liveness bug affected a persisted run.

    The bug only has an effect when a token has fewer than ``k`` positive
    encoder pre-activations, because only then does ``torch.topk`` return a
    zero-valued selection. This function checks for that condition two ways:

    1. **Training-time evidence.** Every logged step recorded ``l0``, computed
       from ``feature_acts > 0``. That metric was always correct, bug or no bug.
       If ``min(l0) == k`` across every logged step, no zero-valued selection
       occurred in any logged batch.

    2. **Corpus-wide evidence.** Re-encode all stored activations with the
       stored checkpoint and count zero-valued Top-K selections exactly, plus
       the minimum number of positive pre-activations over any single token.

    Read-only: nothing is written and no weights are updated.
    """
    from pathlib import Path

    import torch

    from brainpatch.ml.activation_store import ActivationSubset
    from brainpatch.ml.sae import TopKSAE
    from brainpatch.paths import VolumePaths

    paths = VolumePaths(VOL_MOUNT)

    # -- 1. training-time evidence from the metrics log -------------------------
    metrics_path = Path(paths.sae_metrics(experiment))
    logged_l0: list[float] = []
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "l0" in row:
                    logged_l0.append(float(row["l0"]))

    # -- 2. corpus-wide re-encode ----------------------------------------------
    checkpoint_path = Path(paths.sae_checkpoint(experiment))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sae = TopKSAE.from_checkpoint(checkpoint, device="cpu")
    k = sae.k

    input_scale = sae.config.input_scale
    if input_scale is None:
        raise ValueError(f"checkpoint {checkpoint_path} has no recorded input_scale")

    subset = ActivationSubset.load(paths, experiment, dtype=torch.float32)
    total_rows = len(subset)

    zero_selections = 0
    min_positive_pre = k
    rows_below_k = 0
    naive_fire_total = 0
    correct_fire_total = 0

    with torch.no_grad():
        for start in range(0, total_rows, 4096):
            batch = subset.activations[start : start + 4096].to(torch.float32) * float(input_scale)
            out = sae(batch)
            positive = out.topk_values > 0

            zero_selections += int((~positive).sum().item())
            naive_fire_total += int(out.topk_indices.numel())
            correct_fire_total += int(positive.sum().item())

            per_row = positive.sum(dim=-1)
            min_positive_pre = min(min_positive_pre, int(per_row.min().item()))
            rows_below_k += int((per_row < k).sum().item())

    # The checkpoint's fire_count was accumulated by the *buggy* code, which
    # counted every selection. Compare its total against k * tokens_seen: under
    # the bug they are equal by construction, so a match is expected and is NOT
    # itself evidence of corruption. It is reported for completeness.
    stored_fire_total = int(sae.fire_count.sum().item())
    stored_tokens_seen = int(sae.tokens_seen.item())

    affected = zero_selections > 0 or (bool(logged_l0) and min(logged_l0) < k)

    result = {
        "ok": True,
        "experiment": experiment,
        "k": k,
        "d_sae": sae.d_sae,
        "corpus_rows": total_rows,
        "training_evidence": {
            "logged_steps": len(logged_l0),
            "min_logged_l0": min(logged_l0) if logged_l0 else None,
            "max_logged_l0": max(logged_l0) if logged_l0 else None,
            "all_logged_l0_equal_k": bool(logged_l0) and min(logged_l0) == k == max(logged_l0),
        },
        "corpus_evidence": {
            "zero_valued_topk_selections": zero_selections,
            "total_topk_selections": naive_fire_total,
            "strictly_positive_selections": correct_fire_total,
            "rows_with_fewer_than_k_positive": rows_below_k,
            "min_positive_selections_in_any_row": min_positive_pre,
        },
        "checkpoint_state": {
            "stored_fire_count_total": stored_fire_total,
            "stored_tokens_seen": stored_tokens_seen,
            "k_times_tokens_seen": k * stored_tokens_seen,
            "note": (
                "The stored fire_count was written by the pre-fix code, which counted "
                "every Top-K index. Equality with k * tokens_seen is expected under "
                "the old code regardless of whether any zero-valued selection occurred, "
                "so it is not evidence either way. The corpus and training evidence above "
                "are what settle the question."
            ),
        },
        "smoke_v0_affected_by_liveness_bug": affected,
        "verdict": (
            "AFFECTED: zero-valued Top-K selections occurred, so persisted "
            "fire_count / dead-feature numbers are inflated and must be recomputed."
            if affected
            else "NOT AFFECTED: every Top-K selection had a strictly positive value, so "
            "the buggy and fixed accounting produce identical numbers for this run. "
            "Persisted smoke_v0 metrics remain valid as published."
        ),
    }
    print(json.dumps(result, indent=2))
    return result
