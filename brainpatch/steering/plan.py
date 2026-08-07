"""Resolving installed patches into a concrete per-token intervention.

The runtime keeps a set of installed :class:`~brainpatch.schemas.patch.BrainPatchSpec`
objects, each with a user-controllable strength multiplier and enabled flag. On
every forward pass the hook needs one question answered: *at generated-token
index n, which feature directions do I add, and with what coefficients?*

:class:`InterventionPlan` answers that with no torch involved, which is what
makes the semantics testable locally. In particular the guarantee that

    total strength == 0  =>  no tensor is modified at all

is enforced here by returning an empty edit list, so "strength 0" is baseline by
construction rather than by floating-point luck.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from brainpatch.schemas.patch import BrainPatchSpec
from brainpatch.steering.schedule import StrengthSchedule

#: Coefficients whose absolute value is below this are treated as exactly zero.
#: Chosen well below any strength a user would set, but above float noise.
STRENGTH_EPSILON = 1e-12


@dataclass
class PlannedEdit:
    """One feature direction to apply, with its resolved coefficient."""

    feature_id: int
    coefficient: float
    mode: str = "add"
    source_patch: str = ""


@dataclass
class InstalledPatch:
    """A patch registered in the runtime, plus its live control state."""

    spec: BrainPatchSpec
    strength: float = 1.0
    enabled: bool = True
    schedule: StrengthSchedule | None = None

    def __post_init__(self) -> None:
        if self.schedule is None and self.spec.schedule is not None:
            self.schedule = StrengthSchedule.from_dict(self.spec.schedule)

    @property
    def name(self) -> str:
        return self.spec.name

    def multiplier_at(self, token_index: int) -> float:
        """Global multiplier for this patch at a generated-token index."""
        if not self.enabled:
            return 0.0
        if self.schedule is None:
            return self.strength
        return self.strength * self.schedule.strength_at(token_index)


@dataclass
class InterventionPlan:
    """The set of installed patches and the layers they touch.

    A plan is intentionally cheap to build and query: the hot path
    (:meth:`edits_at`) runs once per forward pass.
    """

    patches: dict[str, InstalledPatch] = field(default_factory=dict)
    #: Restrict interventions to ``[start, end)`` in generated-token index.
    #: ``None`` means "every generated token".
    token_range: tuple[int, int] | None = None

    # -- registry --------------------------------------------------------------

    def install(
        self,
        spec: BrainPatchSpec,
        *,
        strength: float = 1.0,
        enabled: bool = True,
    ) -> None:
        """Register a patch. Re-installing the same name replaces it."""
        self.patches[spec.name] = InstalledPatch(spec=spec, strength=strength, enabled=enabled)

    def uninstall(self, name: str) -> None:
        if name not in self.patches:
            raise KeyError(f"no patch named {name!r} is installed")
        del self.patches[name]

    def set_strength(self, name: str, strength: float) -> None:
        if name not in self.patches:
            raise KeyError(f"no patch named {name!r} is installed")
        self.patches[name].strength = float(strength)

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name not in self.patches:
            raise KeyError(f"no patch named {name!r} is installed")
        self.patches[name].enabled = bool(enabled)

    def set_schedule(self, name: str, schedule: StrengthSchedule | None) -> None:
        if name not in self.patches:
            raise KeyError(f"no patch named {name!r} is installed")
        self.patches[name].schedule = schedule

    # -- queries ---------------------------------------------------------------

    def layers(self) -> list[int]:
        """Every layer index that at least one installed patch hooks."""
        return sorted({p.spec.sae.layer for p in self.patches.values()})

    def is_active(self, token_index: int) -> bool:
        """True if any edit would be applied at this generated-token index."""
        return bool(self.edits_at(token_index))

    def in_token_range(self, token_index: int) -> bool:
        if self.token_range is None:
            return True
        start, end = self.token_range
        return start <= token_index < end

    def edits_at(self, token_index: int, *, layer: int | None = None) -> list[PlannedEdit]:
        """Resolve the edits to apply at ``token_index``.

        Parameters
        ----------
        token_index:
            Index of the token being generated, 0-based, prompt excluded.
        layer:
            When given, only edits for patches hooking this layer are returned.

        Returns
        -------
        list[PlannedEdit]
            Possibly empty. An empty list is the signal to leave the residual
            stream untouched, which is what makes ``strength=0`` bit-identical
            to baseline rather than merely close to it.
        """
        if not self.in_token_range(token_index):
            return []

        edits: list[PlannedEdit] = []
        for patch in self.patches.values():
            if layer is not None and patch.spec.sae.layer != layer:
                continue
            multiplier = patch.multiplier_at(token_index)
            if abs(multiplier) < STRENGTH_EPSILON:
                continue
            for edit in patch.spec.features:
                coefficient = multiplier * edit.strength
                if abs(coefficient) < STRENGTH_EPSILON:
                    continue
                edits.append(
                    PlannedEdit(
                        feature_id=edit.feature_id,
                        coefficient=coefficient,
                        mode=edit.mode,
                        source_patch=patch.name,
                    )
                )
        return edits

    def describe(self) -> list[str]:
        """One human-readable line per installed patch."""
        lines = []
        for patch in self.patches.values():
            state = "on" if patch.enabled else "off"
            sched = "scheduled" if patch.schedule is not None else "constant"
            lines.append(
                f"{patch.name}: strength={patch.strength:+.3f} [{state}, {sched}] "
                f"-> {patch.spec.summary()}"
            )
        return lines
