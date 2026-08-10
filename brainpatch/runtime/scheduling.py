"""Token-indexed strength schedules.

Re-exported from :mod:`brainpatch.steering.schedule`, which already implements
and tests this. It lives under ``runtime`` too because scheduling is a *runtime*
concern, not a research one, and a user reading the runtime package should not
have to know that the code happens to predate the split.
"""

from __future__ import annotations

from brainpatch.steering.schedule import StrengthSchedule

__all__ = ["StrengthSchedule"]
