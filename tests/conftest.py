"""Test configuration.

These tests are **pure Python**. They must pass on a machine with no torch, no
transformers, no CUDA and no network, and they must never trigger a paid Modal
job. The guard below enforces the first half of that: if a test (or something a
test imports) reaches for the ML stack, it fails loudly rather than silently
passing on a developer machine that happens to have torch installed.

``tests/remote/`` holds pytest files that legitimately need torch (SAE maths).
They are excluded from local collection by ``collect_ignore_glob`` below and are
executed inside a Modal CPU container via
``modal run modal_app/app.py::sae_unit_tests``. They need no GPU and no model,
so running them costs cents, but they must never run locally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Set to "1" only by the Modal runner in ``modal_app/sae_audit.py``. Opt-in
#: rather than auto-detected: a local machine that happens to have torch
#: installed must still get the local-only behaviour, and this whole file exists
#: to stop that machine from silently passing tests it should not be running.
REMOTE_MODE = os.environ.get("BRAINPATCH_REMOTE_TESTS") == "1"

#: ``tests/remote/`` needs torch, so it is invisible to local collection.
collect_ignore_glob = [] if REMOTE_MODE else ["remote/*"]

#: Importing any of these from a local test is a bug, not a convenience.
FORBIDDEN_TOP_LEVEL = frozenset(
    {"torch", "transformers", "datasets", "accelerate", "safetensors", "bitsandbytes"}
)


class _HeavyImportBlocker:
    """Meta-path hook that turns a heavy import into an explicit failure."""

    def find_module(self, fullname: str, path=None):  # noqa: D102 - legacy finder API
        return self if fullname.split(".")[0] in FORBIDDEN_TOP_LEVEL else None

    def load_module(self, fullname: str):  # noqa: D102
        raise AssertionError(
            f"local test tried to import {fullname!r}. The local test suite must not "
            "require the ML stack -- move this test to a Modal entry point."
        )


@pytest.fixture(scope="session", autouse=True)
def _block_heavy_imports():
    """Install the blocker for the whole session, except in remote mode.

    In remote mode the suite is running inside a Modal container where torch is
    both present and required, so blocking it would be nonsense.
    """
    if REMOTE_MODE:
        yield
        return
    blocker = _HeavyImportBlocker()
    sys.meta_path.insert(0, blocker)
    yield
    sys.meta_path.remove(blocker)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def valid_patch_dict() -> dict:
    """A minimal well-formed patch, used as the base for mutation tests."""
    return {
        "format_version": "0.1",
        "name": "test-patch",
        "description": "fixture",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "sae": {
            "reference": "smoke_v0",
            "layer": 18,
            "hook": "residual_post",
            "d_in": 1536,
            "d_sae": 2048,
            "input_scale": 0.5610531069008018,
        },
        "features": [{"feature_id": 727, "strength": 16.0}],
        "evidence_level": "none",
        "license": "Apache-2.0",
        "authors": [],
    }
