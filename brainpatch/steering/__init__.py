"""Steering control logic.

Pure-Python scheduling and intervention *planning* live here. The torch hooks
that actually mutate activations live in :mod:`brainpatch.research.ml.intervention`.

Keeping the two apart means the trickiest part to get right -- when a strength
changes and by how much -- is unit-testable on a laptop with no GPU.
"""

from brainpatch.steering.plan import InterventionPlan, PlannedEdit
from brainpatch.steering.schedule import StrengthSchedule

__all__ = ["InterventionPlan", "PlannedEdit", "StrengthSchedule"]
