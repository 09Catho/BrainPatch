"""Regenerate the `anti_sycophancy_v3` dataset and its manifest.

    python scripts/build_sycophancy_v3.py

Deterministic and gated: the length audit and the overlap checks against v1 and
v2 both run as **gates**, and the script writes nothing if either fails. A
confounded or contaminated dataset makes every downstream number meaningless,
and finding that out before any GPU time is spent is the cheapest place to do it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from brainpatch.research.sycophancy_data import FALSE_CLAIMS as V1_FALSE
from brainpatch.research.sycophancy_data import TRUE_CLAIMS as V1_TRUE
from brainpatch.research.sycophancy_data_v2 import CLAIMS as V2_CLAIMS
from brainpatch.research.sycophancy_data_v3 import CLAIMS
from brainpatch.research.sycophancy_v3_build import (
    audit_lengths,
    audit_per_split,
    build_examples,
    dataset_hashes,
    near_duplicate_pairs,
    split_counts,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "contrast" / "antisycophancy_v3.json"
MANIFEST = ROOT / "experiments" / "anti_sycophancy_v3" / "dataset_manifest.json"

DESCRIPTION = (
    "SYNTHETIC behavioural dataset for anti_sycophancy_v3. Fresh pool with zero topic "
    "and zero assertion overlap against anti_sycophancy_v1 (test opened, negative result) "
    "and anti_sycophancy_v2 (test never opened, sealed). Each item is a user assertion "
    "carrying social pressure to agree. TWO POLARITIES: 'false_claim' items assert "
    "something false and the preferred response challenges it; 'true_claim' items assert "
    "something true and the preferred response agrees while the undesired one manufactures "
    "a disagreement. The true items are the guard separating selective independence from "
    "contrarianism. v3 selects candidates by FREE-GENERATION behaviour rather than by "
    "continuation log-probability, after v2 measured corr(log-prob effect, generation "
    "correction gain) = -0.298; log-probability remains a reported diagnostic, which is "
    "why the length balance is still enforced. Response pairs are authored at similar "
    "length and then trimmed to near-parity. Splits are sized exactly and filled "
    "proportionally across every (category, polarity, length polarity) stratum. "
    "HAND-WRITTEN, not validated against human annotation, NOT a benchmark."
)


def main() -> int:
    examples = build_examples()
    audit = audit_lengths(examples)
    per_split = audit_per_split(examples)
    duplicates = near_duplicate_pairs(examples)

    v1_topics = {c[0] for c in list(V1_FALSE) + list(V1_TRUE)}
    v1_assertions = {c[2] for c in list(V1_FALSE) + list(V1_TRUE)}
    v2_topics = {c[1] for c in V2_CLAIMS}
    v2_assertions = {c[4] for c in V2_CLAIMS}
    topics = {c[1] for c in CLAIMS}
    assertions = {c[4] for c in CLAIMS}
    topic_overlap = sorted((v1_topics | v2_topics) & topics)
    assertion_overlap = sorted((v1_assertions | v2_assertions) & assertions)

    print("=== length audit (characters) ===")
    print(audit.render())
    print()
    for split, result in per_split.items():
        print(f"  {split:<11} {'OK  ' if result.ok else 'FAIL'} " + " ".join(result.failures))

    print()
    print("=== contamination checks ===")
    print(f"  topics shared with v1/v2:     {len(topic_overlap)}")
    print(f"  assertions shared with v1/v2: {len(assertion_overlap)}")
    print(f"  near-duplicate pairs:         {len(duplicates)}")

    failed = not audit.ok or any(not r.ok for r in per_split.values())
    if topic_overlap or assertion_overlap:
        print(f"\nCONTAMINATION: {topic_overlap[:5]} {assertion_overlap[:3]}", file=sys.stderr)
        failed = True
    if duplicates:
        print(f"\n{len(duplicates)} near-duplicate assertion pairs:", file=sys.stderr)
        for left, right, score in duplicates[:10]:
            print(f"  {left} ~ {right}  {score:.2f}", file=sys.stderr)
        failed = True

    if failed:
        print(
            "\nDATASET AUDIT FAILED -- nothing written. Fix the pool before running any "
            "experiment.",
            file=sys.stderr,
        )
        return 1

    payload = {
        "name": "antisycophancy_v3",
        "description": DESCRIPTION,
        "synthetic": True,
        "version": "3.0",
        "examples": examples,
    }
    FIXTURE.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    counts = split_counts(examples)
    categories = sorted({e["category"] for e in examples})
    total = len(examples)
    true_total = sum(1 for e in examples if e["metadata"]["polarity"] == "true_claim")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "name": "antisycophancy_v3",
                "version": "3.0",
                "generated_by": "scripts/build_sycophancy_v3.py",
                "source_pool": "brainpatch/research/sycophancy_data_v3.py",
                "n_propositions": total,
                "n_categories": len(categories),
                "categories": categories,
                "true_control_fraction": true_total / total,
                "split_counts": counts,
                "sha256": dataset_hashes(examples),
                "length_audit_characters": {
                    "global": audit.stats,
                    "per_split": {k: v.stats for k, v in per_split.items()},
                },
                "near_duplicate_pairs": len(duplicates),
                "overlap_with_previous_experiments": {
                    "v1_shared_topics": 0,
                    "v1_shared_assertions": 0,
                    "v2_shared_topics": 0,
                    "v2_shared_assertions": 0,
                    "note": (
                        "anti_sycophancy_v1 and v2 are frozen. v1's test split was opened; "
                        "v2's was never opened and stays sealed. Neither is reused here."
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
