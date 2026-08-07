#!/usr/bin/env python
"""Local pre-flight check. Runs nothing remote and costs nothing.

Verifies the four things that must hold on a development machine:

1. the pure-Python package imports with the ML stack blocked
2. the local test suite passes
3. every shipped patch validates
4. no shipped patch claims causal evidence it has not recorded

Usage::

    python scripts/verify_local.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FORBIDDEN = {"torch", "transformers", "datasets", "accelerate", "safetensors"}


class _Blocker:
    def find_module(self, fullname: str, path=None):
        return self if fullname.split(".")[0] in FORBIDDEN else None

    def load_module(self, fullname: str):
        raise AssertionError(f"local import boundary violated: {fullname}")


def check_import_boundary() -> bool:
    sys.meta_path.insert(0, _Blocker())
    try:
        import brainpatch
        from brainpatch.datasets import CONTRAST_SET_NAMES, load_contrast_set

        for name in CONTRAST_SET_NAMES:
            load_contrast_set(name).validate()
        print(f"  ok: brainpatch {brainpatch.__version__} imports with ML stack blocked")
        return True
    except Exception as exc:  # noqa: BLE001 - report and continue to other checks
        print(f"  FAIL: {exc}")
        return False
    finally:
        sys.meta_path.pop(0)


def check_patches() -> bool:
    from brainpatch.patches.io import load_patch_dir

    specs, failures = load_patch_dir(REPO_ROOT / "patches", strict=False)
    for path, message in failures:
        print(f"  FAIL: {path.name}: {message}")
    if failures:
        return False

    overclaims = [s.name for s in specs if s.is_validated and not s.evaluation]
    for name in overclaims:
        print(f"  FAIL: {name} claims causal evidence with no recorded evaluation")
    if overclaims:
        return False

    for spec in specs:
        print(f"  ok: {spec.summary()}")
    return True


def check_tests() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=REPO_ROOT, check=False
    )
    return result.returncode == 0


def main() -> int:
    checks = [
        ("import boundary", check_import_boundary),
        ("shipped patches", check_patches),
        ("local test suite", check_tests),
    ]
    failed: list[str] = []
    for label, fn in checks:
        print(f"\n== {label} ==")
        if not fn():
            failed.append(label)

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("all local checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
