"""Turnover must not determine the precision of a conservation check."""

import json
from pathlib import Path

import pytest

from moonsim import from_flat, run
from moonsim.engine import ConservationError, Simulation

FIXTURES = Path(__file__).parent / "fixtures"


def test_minimized_two_trillion_dollar_churn_conserves():
    cases = json.loads((FIXTURES / "minimal_failures.json").read_text())
    flat = next(c["flat"] for c in cases if c["signature"][0] == "ConservationError")
    res = run(from_flat(flat), strict=True)
    assert res.max_conservation_error < 1e-9


def test_all_five_original_dollar_failures_conserve():
    cases = json.loads((FIXTURES / "fuzz_failures.json").read_text())
    cases = [c for c in cases if c["type"] == "ConservationError"]
    assert len(cases) == 5
    for case in cases:
        res = run(from_flat(case["flat"]), strict=True)
        assert res.max_conservation_error < 1e-9, case["label"]


def test_strict_mode_still_detects_a_real_missing_dollar():
    sim = Simulation(from_flat({}), strict=True)
    sim.holders.usd -= 1.0
    with pytest.raises(ConservationError, match="ledger drift"):
        sim.check_conservation()


def test_compensated_cash_keeps_change_and_cannot_be_overdrawn():
    from moonsim.engine import Wallet

    wallet = Wallet("large cohort", usd=1e16)
    wallet.add_usd(1.0)
    wallet.add_usd(-1e16)
    assert wallet.usd == 1.0
    wallet.add_usd(-wallet.usd)
    assert wallet.usd == 0.0
    wallet.usd = 1e16
    wallet.add_usd(-1.0)
    wallet.add_usd(-wallet.usd)
    assert wallet.usd >= 0.0


def test_compensated_issue_and_burn_counters_keep_small_entries():
    from moonsim.engine import _RunningSum

    total = _RunningSum()
    for amount in (1e16, 1.0, 1.0, -1e16):
        total.add(amount)
    assert total.value == 2.0
