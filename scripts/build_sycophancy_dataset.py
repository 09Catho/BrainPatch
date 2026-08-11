"""Regenerate ``examples/contrast/antisycophancy_v1.json`` from the claim pool.

The JSON fixture is what Modal actually loads, and it is committed so a run is
reproducible from the repository alone. This script is the only thing that
writes it -- edit :mod:`brainpatch.research.sycophancy_data` and re-run.

    python scripts/build_sycophancy_dataset.py

Deterministic: no seed, no shuffling, no sampling. Running it twice on an
unchanged claim pool produces a byte-identical file.
"""

from __future__ import annotations

import json
from pathlib import Path

from brainpatch.research.sycophancy_data import (
    FALSE_CLAIMS,
    TRUE_CLAIMS,
    build_examples,
    split_counts,
)

OUT = Path(__file__).resolve().parent.parent / "examples" / "contrast" / "antisycophancy_v1.json"

DESCRIPTION = (
    "SYNTHETIC behavioural dataset for the independent-criticism / anti-sycophancy "
    "experiment. Each item is a user assertion presented with social pressure to agree. "
    "TWO POLARITIES: 'false_claim' items assert something false, and the positive "
    "response corrects it; 'true_claim' items assert something true, and the positive "
    "response agrees while the negative response manufactures a disagreement. The true "
    "items are the control that distinguishes selective independence from plain "
    "contrarianism -- a direction that only increases disagreement gains on one polarity "
    "and loses on the other. Every item is a distinct proposition: no claim is restated "
    "in a second wording and none is reused across splits, so the item count is a count "
    "of independent observations. Splits are stratified within category so every "
    "category and both polarities appear in all three. Invitation phrasings are drawn "
    "from pools that are disjoint across splits, so the 'do you agree?' wrapper cannot "
    "itself carry the signal. Responses are matched for length and register within a "
    "pair. HAND-WRITTEN, not validated against human annotation, NOT a benchmark."
)


def main() -> None:
    examples = build_examples()
    payload = {
        "name": "antisycophancy_v1",
        "description": DESCRIPTION,
        "synthetic": True,
        "version": "1.0",
        "examples": examples,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    counts = split_counts(examples)
    print(f"wrote {OUT} ({len(examples)} examples)")
    print(f"claim pool: {len(FALSE_CLAIMS)} false, {len(TRUE_CLAIMS)} true")
    for split in ("train", "validation", "test"):
        row = counts[split]
        print(
            f"  {split:<11} total={row['total']:>3}  "
            f"false={row['false_claim']:>3}  true={row['true_claim']:>3}"
        )


if __name__ == "__main__":
    main()
