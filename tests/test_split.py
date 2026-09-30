from __future__ import annotations

import pytest

from fabweave.lot.merge import Lot
from fabweave.lot.split import SplitResult, SubLot, split_decision, split_lots
from fabweave.workflows.control import split_for_processing


RETICLE = "RET-1"
COMPATIBLE = frozenset({(RETICLE, "M1"), (RETICLE, "M2"), (RETICLE, "M3")})


def _parent(quantity: int = 100, holds: frozenset[str] = frozenset()) -> Lot:
    return Lot("LOT-1", "P", "R", "S", 1, quantity, holds)


def test_split_basic_divides_across_parallel_machines():
    result = split_lots("REQ-1", _parent(), RETICLE, {"M1": 60, "M2": 60}, COMPATIBLE, 30)
    assert [sub.machine_id for sub in result.sub_lots] == ["M1", "M2"]
    assert [sub.quantity for sub in result.sub_lots] == [60, 40]


def test_split_uses_actual_available_capacity():
    result = split_lots("REQ-2", _parent(), RETICLE, {"M1": 60, "M2": 50}, COMPATIBLE, 30)
    assert [sub.quantity for sub in result.sub_lots] == [60, 40]


def test_split_single_sub_lot_when_one_machine_covers_lot():
    result = split_lots("REQ-3", _parent(), RETICLE, {"M1": 100}, COMPATIBLE, 30)
    assert len(result.sub_lots) == 1
    assert result.sub_lots[0].machine_id == "M1" and result.sub_lots[0].quantity == 100


def test_split_spills_across_machines_when_no_single_machine_covers_lot():
    result = split_lots("REQ-4", _parent(), RETICLE, {"M1": 40, "M2": 40, "M3": 40}, COMPATIBLE, 30)
    assert [sub.quantity for sub in result.sub_lots] == [40, 30, 30]


def test_split_never_produces_sub_lot_below_min_lot_size():
    result = split_lots("REQ-5", _parent(), RETICLE, {"M1": 80, "M2": 30}, COMPATIBLE, 30)
    assert all(sub.quantity >= 30 for sub in result.sub_lots)
    assert [sub.quantity for sub in result.sub_lots] == [70, 30]


def test_split_conserves_parent_quantity():
    result = split_lots("REQ-6", _parent(), RETICLE, {"M1": 40, "M2": 40, "M3": 40}, COMPATIBLE, 30)
    assert sum(sub.quantity for sub in result.sub_lots) == result.quantity == 100


def test_split_is_deterministic_and_replays_identically():
    first = split_lots("REQ-7", _parent(), RETICLE, {"M1": 60, "M2": 50}, COMPATIBLE, 30)
    second = split_lots("REQ-7", _parent(), RETICLE, {"M1": 60, "M2": 50}, COMPATIBLE, 30)
    assert first == second
    assert [sub.sub_lot_id for sub in first.sub_lots] == [sub.sub_lot_id for sub in second.sub_lots]
    assert first.split_id == second.split_id


def test_split_parent_child_relationship_is_recorded():
    result = split_lots("REQ-8", _parent(), RETICLE, {"M1": 60, "M2": 50}, COMPATIBLE, 30)
    assert result.parent_of == {sub.sub_lot_id: "LOT-1" for sub in result.sub_lots}
    assert all(sub.parent_lot_id == "LOT-1" for sub in result.sub_lots)


def test_split_respects_reticle_machine_matching():
    # M2 has plenty of capacity but is not qualified for this reticle, so it
    # must not receive a sub-lot even when it is the only way to absorb the lot.
    result = split_lots("REQ-9", _parent(), RETICLE, {"M1": 100, "M2": 100}, frozenset({(RETICLE, "M1")}), 30)
    assert [sub.machine_id for sub in result.sub_lots] == ["M1"]
    assert "M2" not in result.machine_loads


def test_split_rejects_insufficient_compatible_capacity_atomically():
    decision = split_decision(_parent(), RETICLE, {"M1": 40, "M2": 40}, COMPATIBLE, 30)
    assert not decision.allowed and decision.reasons == ("insufficient_compatible_capacity",)
    with pytest.raises(ValueError):
        split_lots("REQ-10", _parent(), RETICLE, {"M1": 40, "M2": 40}, COMPATIBLE, 30)


def test_split_rejects_min_lot_size_that_cannot_partition_lot():
    decision = split_decision(_parent(), RETICLE, {"M1": 80, "M2": 80}, COMPATIBLE, 60)
    assert not decision.allowed and decision.reasons == ("min_lot_size_exceeds_partition",)
    with pytest.raises(ValueError):
        split_lots("REQ-11", _parent(), RETICLE, {"M1": 80, "M2": 80}, COMPATIBLE, 60)


def test_split_rejects_when_no_compatible_machine_has_capacity():
    decision = split_decision(_parent(), RETICLE, {"M1": 60}, frozenset({("RET-OTHER", "M1")}), 30)
    assert not decision.allowed and decision.reasons == ("no_compatible_machine_with_capacity",)
    with pytest.raises(ValueError):
        split_lots("REQ-12", _parent(), RETICLE, {"M1": 60}, frozenset({("RET-OTHER", "M1")}), 30)


def test_split_rejects_held_parent_lot():
    decision = split_decision(_parent(holds=frozenset({"QA"})), RETICLE, {"M1": 60}, COMPATIBLE, 30)
    assert not decision.allowed and decision.reasons == ("parent_lot_held",)
    with pytest.raises(ValueError):
        split_lots("REQ-13", _parent(holds=frozenset({"QA"})), RETICLE, {"M1": 60}, COMPATIBLE, 30)


def test_split_rejects_invalid_parent_quantity():
    decision = split_decision(_parent(quantity=0), RETICLE, {"M1": 60}, COMPATIBLE, 30)
    assert not decision.allowed and decision.reasons == ("invalid_parent_quantity",)


def test_split_rejects_invalid_min_lot_size():
    decision = split_decision(_parent(), RETICLE, {"M1": 60}, COMPATIBLE, 0)
    assert not decision.allowed and decision.reasons == ("invalid_min_lot_size",)


def test_split_rejects_missing_reticle_id():
    decision = split_decision(_parent(), "", {"M1": 60}, COMPATIBLE, 30)
    assert not decision.allowed and decision.reasons == ("missing_reticle_id",)


def test_split_decision_allowed_carries_evidence():
    decision = split_decision(_parent(), RETICLE, {"M1": 60, "M2": 50}, COMPATIBLE, 30)
    assert decision.allowed
    assert decision.evidence["sub_lot_count"] == 2
    assert decision.evidence["min_lot_size"] == 30


def test_split_min_lot_size_equal_to_quantity_is_a_single_sub_lot():
    result = split_lots("REQ-14", _parent(quantity=100), RETICLE, {"M1": 100}, COMPATIBLE, 100)
    assert len(result.sub_lots) == 1 and result.sub_lots[0].quantity == 100


def test_split_machine_loads_reflect_assignment():
    result = split_lots("REQ-15", _parent(), RETICLE, {"M1": 60, "M2": 50}, COMPATIBLE, 30)
    assert dict(result.machine_loads) == {"M1": 60, "M2": 40}


def test_split_for_processing_builds_genealogy_graph():
    result, graph = split_for_processing(
        "REQ-16", _parent(), RETICLE, {"M1": 60, "M2": 50}, COMPATIBLE, 30
    )
    assert isinstance(result, SplitResult)
    assert graph.related("LOT-1", "downstream", 1) == tuple(
            sorted(sub.sub_lot_id for sub in result.sub_lots)
        )
    for sub in result.sub_lots:
        assert graph.related(sub.sub_lot_id, "upstream", 1) == ("LOT-1",)


def test_split_replay_with_different_request_id_produces_different_ids():
    first = split_lots("REQ-17", _parent(), RETICLE, {"M1": 100}, COMPATIBLE, 30)
    second = split_lots("REQ-18", _parent(), RETICLE, {"M1": 100}, COMPATIBLE, 30)
    assert first.sub_lots[0].sub_lot_id != second.sub_lots[0].sub_lot_id
    assert first.sub_lots[0].quantity == second.sub_lots[0].quantity


def test_split_sub_lot_is_a_sub_lot_instance():
    result = split_lots("REQ-19", _parent(), RETICLE, {"M1": 100}, COMPATIBLE, 30)
    assert isinstance(result.sub_lots[0], SubLot)
