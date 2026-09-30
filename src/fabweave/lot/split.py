from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from fabweave.core.decision import Decision, allow, deny
from fabweave.core.ids import deterministic_id
from fabweave.core.quantities import require_conservation
from fabweave.lot.merge import Lot


@dataclass(frozen=True, slots=True)
class SubLot:
    sub_lot_id: str
    parent_lot_id: str
    machine_id: str
    quantity: int
    sequence: int


@dataclass(frozen=True, slots=True)
class SplitResult:
    request_id: str
    split_id: str
    parent_lot_id: str
    reticle_id: str
    quantity: int
    sub_lots: tuple[SubLot, ...]

    @property
    def parent_of(self) -> Mapping[str, str]:
        return MappingProxyType({sub.sub_lot_id: sub.parent_lot_id for sub in self.sub_lots})

    @property
    def machine_loads(self) -> Mapping[str, int]:
        return MappingProxyType({sub.machine_id: sub.quantity for sub in self.sub_lots})


def _usable_machines(
    reticle_id: str,
    available: Mapping[str, int],
    compatible: frozenset[tuple[str, str]],
    min_lot_size: int,
) -> list[tuple[str, int]]:
    """Return compatible machines with capacity >= min_lot_size, sorted deterministically.

    Sorting by (-capacity, machine_id) makes the assignment reproducible: the
    largest compatible machine is filled first, and ties break by machine id.
    """
    usable = [
        (machine_id, capacity)
        for machine_id, capacity in available.items()
        if capacity >= min_lot_size and (reticle_id, machine_id) in compatible
    ]
    return sorted(usable, key=lambda item: (-item[1], item[0]))


def _select_count(quantity: int, min_lot_size: int, machines: list[tuple[str, int]]) -> int:
    """Return the fewest machines whose total capacity can absorb the lot.

    Raises ValueError when even the full compatible fleet cannot cover the
    quantity, so the caller can reject atomically rather than leave a partial
    split.
    """
    cumulative = 0
    for count, (_, capacity) in enumerate(machines, start=1):
        cumulative += capacity
        if cumulative >= quantity:
            return count
    raise ValueError("insufficient compatible capacity to split the lot")


def split_decision(
    parent: Lot,
    reticle_id: str,
    available: Mapping[str, int],
    compatible: frozenset[tuple[str, str]],
    min_lot_size: int,
) -> Decision:
    """Evaluate whether the parent lot can be split across the available machines.

    A split is feasible only when every constraint holds at once: the parent is
    not held, the quantity and lower limit are valid, and the compatible fleet
    can absorb the whole lot without any sub-lot falling below min_lot_size.
    """
    if parent.active_holds:
        return deny("parent_lot_held")
    if parent.quantity <= 0:
        return deny("invalid_parent_quantity")
    if min_lot_size <= 0:
        return deny("invalid_min_lot_size")
    if not reticle_id:
        return deny("missing_reticle_id")

    usable = _usable_machines(reticle_id, available, compatible, min_lot_size)
    if not usable:
        return deny("no_compatible_machine_with_capacity")

    quantity = parent.quantity
    try:
        required_count = _select_count(quantity, min_lot_size, usable)
    except ValueError:
        return deny("insufficient_compatible_capacity")
    if required_count * min_lot_size > quantity:
        return deny(
            "min_lot_size_exceeds_partition",
            min_lot_size=min_lot_size,
            quantity=quantity,
        )

    total_capacity = sum(capacity for _, capacity in usable[:required_count])
    return allow(
        sub_lot_count=required_count,
        total_capacity=total_capacity,
        min_lot_size=min_lot_size,
    )


def split_lots(
    request_id: str,
    parent: Lot,
    reticle_id: str,
    available: Mapping[str, int],
    compatible: frozenset[tuple[str, str]],
    min_lot_size: int,
) -> SplitResult:
    """Split the parent lot into sub-lots across the available machines.

    The split is all-or-nothing: every constraint is validated before any
    sub-lot is materialized, and quantity conservation is asserted before the
    result is returned.  Sub-lot ids, quantities and the parent-child mapping
    are deterministic for a given request, so the same request replays to the
    same split.
    """
    decision = split_decision(parent, reticle_id, available, compatible, min_lot_size)
    if not decision.allowed:
        raise ValueError(f"split rejected: {', '.join(decision.reasons)}")

    usable = _usable_machines(reticle_id, available, compatible, min_lot_size)
    quantity = parent.quantity
    required_count = _select_count(quantity, min_lot_size, usable)
    selected = usable[:required_count]

    assignments: list[tuple[str, int]] = []
    remaining = quantity - required_count * min_lot_size
    for machine_id, capacity in selected:
        extra = min(remaining, capacity - min_lot_size)
        assignments.append((machine_id, min_lot_size + extra))
        remaining -= extra
    if remaining != 0:
        raise ValueError("split assignment failed to conserve quantity")

    split_id = deterministic_id("SPLIT", request_id, parent.lot_id)
    sub_lots: list[SubLot] = []
    for sequence, (machine_id, sub_quantity) in enumerate(assignments, start=1):
        sub_lot_id = deterministic_id("LOT", request_id, parent.lot_id, machine_id, sequence)
        sub_lots.append(
            SubLot(
                sub_lot_id=sub_lot_id,
                parent_lot_id=parent.lot_id,
                machine_id=machine_id,
                quantity=sub_quantity,
                sequence=sequence,
            )
        )

    before = {parent.lot_id: quantity}
    after = {sub.sub_lot_id: sub.quantity for sub in sub_lots}
    require_conservation(before, after)

    return SplitResult(
        request_id=request_id,
        split_id=split_id,
        parent_lot_id=parent.lot_id,
        reticle_id=reticle_id,
        quantity=quantity,
        sub_lots=tuple(sub_lots),
    )
