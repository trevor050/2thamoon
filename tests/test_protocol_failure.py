"""Governance collapse is an observable, permanent protocol failure, not a crash."""

import numpy as np
import pytest

from moonsim import from_flat, run, story
from moonsim.engine import Simulation


@pytest.mark.parametrize("cr,fee,days", [(0.1795372879407694, 0.0812446960837556, 110), (0.05, 0.1, 120)])
def test_governance_collapse_keeps_running(cr, fee, days):
    res = run(
        from_flat(
            {"run.mode": "stablecoin", "run.days": days, "stable.collateral_ratio": cr, "stable.redeem_fee": fee}
        ),
        strict=True,
    )
    failed = res.series["protocol_failed"]
    assert failed.dtype == np.bool_
    assert failed[-1]
    day = res.protocol_failure_day
    assert day is not None and 0 < day <= 120
    assert not failed[:day].any() and failed[day:].all()
    events = [e for e in res.events if e.kind == "protocol_failed"]
    assert len(events) == 1 and events[0].day == day
    assert events[0].title and events[0].detail.endswith(".")
    assert np.isfinite(res.series["gov_price"]).all()
    assert np.all(np.diff(res.series["gov_supply"][day:]) == 0)
    assert res.summary()["protocol_failure_day"] == day
    assert any("failed" in s.note.lower() for s in story.key_stats(res))
    assert res.max_conservation_error < 1e-9


def test_failed_redemption_pays_only_pro_rata_collateral():
    sim = Simulation(
        from_flat({"run.mode": "stablecoin", "stable.collateral_ratio": 0.05, "stable.redeem_fee": 0.1}), strict=True
    )
    # Drive a real redemption sequence to failure; retain the same object to probe the aftermath.
    for day in range(1, 121):
        sim.step(day)
        if sim.protocol_failed:
            break
    assert sim.protocol_failed
    sim.redeem_cap_left = sim.token0
    outstanding = sim.token0 + sim.minted - sim.burned
    tokens = min(1000.0, outstanding)
    accepted, cash, gov = sim._quote_redeem(tokens)
    assert accepted == tokens and gov == 0.0
    owed = tokens * sim.peg * (1 - sim.cfg.stable.redeem_fee) * sim.cfg.stable.collateral_ratio
    assert cash == pytest.approx(min(owed, sim.reserve.usd * tokens / outstanding))
    # A rich stablecoin pool cannot restart governance-dependent minting after failure.
    minted, burned, supply = sim.minted, sim.gov_burned, sim.gov_minted
    sim.buy(sim.holders, sim.holders.usd, "holders")
    sim._arbitrage(121)
    assert (sim.minted, sim.gov_burned, sim.gov_minted) == (minted, burned, supply)
    sim.check_conservation()


@pytest.mark.parametrize(
    "threshold,value", [("GOV_MIN_PRICE", 20.0), ("GOV_MIN_RESERVE", 1e9), ("GOV_SUPPLY_CAP", 1.0)]
)
def test_every_failure_trigger_is_permanent_and_preserves_balances(monkeypatch, threshold, value):
    sim = Simulation(from_flat({"run.mode": "stablecoin"}), strict=True)
    gp = sim.gov_pool
    before = (gp.token, gp.usd, sim.gov_minted, sim.reserve.usd)
    with monkeypatch.context() as patch:
        patch.setitem(sim._check_governance.__func__.__globals__, threshold, value)
        sim._check_governance(1)
    assert sim.protocol_failed
    sim._check_governance(2)
    sim._act_gov_holders(2)
    assert before == (gp.token, gp.usd, sim.gov_minted, sim.reserve.usd)
    assert len(sim.events) == 1 and sim.events[0].day == 1
    sim.check_conservation()


def test_all_26_reported_governance_crashes_are_regressions():
    import json
    from pathlib import Path

    failures = json.loads((Path(__file__).parent / "fixtures/fuzz_failures.json").read_text())
    cases = [f for f in failures if f["type"] == "ZeroDivisionError"]
    assert len(cases) == 26
    for case in cases:
        res = run(from_flat(case["flat"]), strict=True)
        assert res.protocol_failure_day is not None, case["label"]
        assert res.max_conservation_error < 1e-9, case["label"]


def test_collateral_only_arbitrage_settles_after_failure(monkeypatch):
    sim = Simulation(
        from_flat(
            {
                "run.mode": "stablecoin",
                "stable.collateral_ratio": 0.3,
                "stable.reserve_ratio": 0.001,
                "stable.max_redeem_per_day": 0.5,
            }
        ),
        strict=True,
    )
    with monkeypatch.context() as patch:
        patch.setitem(sim._check_governance.__func__.__globals__, "GOV_SUPPLY_CAP", 1.0)
        sim._check_governance(1)
    for wallet in sim.wallets:
        sim.sell(wallet, wallet.token, wallet.name if wallet.name in sim.flows else "whales")
    sim.redeem_cap_left = 0.5 * sim.token0
    cash, reserve = sim.arbs.usd, sim.reserve.usd
    sim._arbitrage(2)
    assert sim.redeemed_total > 0
    assert sim.arbs.usd > cash
    assert sim.reserve.usd < reserve
    assert sim.gov_minted == 0.0 and sim.arbs.gov == 0.0
    assert sim.arbs.token < 1e-9
    sim.check_conservation()
