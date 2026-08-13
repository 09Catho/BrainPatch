"""Regenerate the `anti_sycophancy_v2` dataset and its manifest.

    python scripts/build_sycophancy_v2.py

Deterministic: no seed, no sampling, no shuffling. The length audit runs as a
**gate** -- if the pool ever drifts back into a length confound this script
exits non-zero and writes nothing, because a confounded dataset makes every
downstream number uninterpretable and that is exactly how v1 was lost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from brainpatch.research.sycophancy_data_v2 import CLAIMS
from brainpatch.research.sycophancy_v2_build import (
    audit_lengths,
    audit_per_split,
    build_examples,
    dataset_hashes,
    near_duplicate_pairs,
    split_counts,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "contrast" / "antisycophancy_v2.json"
MANIFEST = ROOT / "experiments" / "anti_sycophancy_v2" / "dataset_manifest.json"

DESCRIPTION = (
    "SYNTHETIC behavioural dataset for anti_sycophancy_v2. Fresh pool: no proposition "
    "here appears in anti_sycophancy_v1, whose test split is opened and frozen. Each item "
    "is a user assertion carrying social pressure to agree. TWO POLARITIES: 'false_claim' "
    "items assert something false and the preferred response corrects it; 'true_claim' "
    "items assert something true and the preferred response agrees while the undesired one "
    "manufactures a disagreement. The true items are the control that separates selective "
    "independence from plain contrarianism. LENGTH BALANCE IS THE HEADLINE FIX: v1's "
    "preferred response was longer in ~96% of pairs and its result was disqualified as "
    "indistinguishable from a preference for longer text. Here each pair is authored to a "
    "declared length polarity and then trimmed to near-parity, giving a mean length gap "
    "near zero, a median near zero, and almost no correlation between behavioural class "
    "and length. Splits are stratified within (category, polarity, length polarity); "
    "agreement invitations come from pools disjoint across splits. HAND-WRITTEN, not "
    "validated against human annotation, NOT a benchmark."
)


def main() -> int:
    examples = build_examples()

    audit = audit_lengths(examples)
    per_split = audit_per_split(examples)
    duplicates = near_duplicate_pairs(examples)

    print("=== length audit (characters) ===")
    print(audit.render())
    print()
    for split, result in per_split.items():
        flag = "OK  " if result.ok else "FAIL"
        print(f"  {split:<11} {flag} " + " ".join(result.failures))

    failed = not audit.ok or any(not r.ok for r in per_split.values())
    if duplicates:
        print(f"\n{len(duplicates)} near-duplicate assertion pairs (>=0.75 Jaccard):")
        for left, right, score in duplicates[:10]:
            print(f"  {left} ~ {right}  {score:.2f}")
        failed = True

    if failed:
        print(
            "\nDATASET AUDIT FAILED -- nothing written. Fix the pool before running "
            "any activation experiment; a length-confounded dataset cannot produce an "
            "interpretable result.",
            file=sys.stderr,
        )
        return 1

    payload = {
        "name": "antisycophancy_v2",
        "description": DESCRIPTION,
        "synthetic": True,
        "version": "2.0",
        "examples": examples,
    }
    FIXTURE.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    counts = split_counts(examples)
    hashes = dataset_hashes(examples)
    categories = sorted({e["category"] for e in examples})
    total = len(examples)
    true_total = sum(1 for e in examples if e["metadata"]["polarity"] == "true_claim")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "name": "antisycophancy_v2",
                "version": "2.0",
                "generated_by": "scripts/build_sycophancy_v2.py",
                "source_pool": "brainpatch/research/sycophancy_data_v2.py",
                "n_propositions": total,
                "n_categories": len(categories),
                "categories": categories,
                "true_control_fraction": true_total / total,
                "split_counts": counts,
                "sha256": hashes,
                "length_audit_characters": {
                    "global": audit.stats,
                    "per_split": {k: v.stats for k, v in per_split.items()},
                },
                "near_duplicate_pairs": len(duplicates),
                "relationship_to_v1": {
                    "shared_propositions": 0,
                    "shared_topics": 0,
                    "note": (
                        "anti_sycophancy_v1 is frozen as a negative result. Its opened "
                        "test data is not reused here in any split."
                    ),
                },
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nwrote {FIXTURE}  ({total} examples)")
    print(f"wrote {MANIFEST}")
    print(f"pool: {len(CLAIMS)} claims, {len(categories)} categories, "
          f"{true_total / total:.1%} true controls")
    for split in ("train", "validation", "test"):
        row = counts[split]
        print(f"  {split:<11} total={row['total']:>3} false={row['false_claim']:>3} "
              f"true={row['true_claim']:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
