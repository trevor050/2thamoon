"""Paid promotion consumes real cash and is included in the attacker's return."""

import pytest

from moonsim import from_flat
from moonsim.engine import Simulation


@pytest.mark.parametrize(
    "cash,price,hype,expected", [(0.0, 1000.0, 6.0, 0.0), (1500.0, 1000.0, 6.0, 1.5), (10000.0, 1000.0, 6.0, 6.0)]
)
def test_promotion_is_limited_to_affordable_hype(cash, price, hype, expected):
    cfg = from_flat(
        {
            "attacker.style": "pump_and_dump",
            "attacker.capital_usd": cash,
            "attacker.accumulate_days": 0,
            "attacker.promo_hype": hype,
            "attacker.promo_usd_per_hype": price,
        }
    )
    sim = Simulation(cfg, strict=True)
    sim._record(0)
    before = sim._usd_total()
    sim._start_pump(1)
    sim._record(1)
    cost = expected * price
    assert sim.hype == pytest.approx(expected)
    assert sim.attacker.usd == pytest.approx(cash - cost)
    assert sim.external_spend == pytest.approx(cost)
    assert sim._usd_total() == pytest.approx(before - cost)
    assert sim.attacker_cost == pytest.approx(cost)
    assert sim.series["external_spend_usd"][1] == pytest.approx(cost)
    assert sim.series["attacker_pnl"][1] == pytest.approx(-cost)
    sim.check_conservation()
    event = next(e for e in sim.events if e.kind == "attack_pump")
    assert "spends" in event.detail.lower()
