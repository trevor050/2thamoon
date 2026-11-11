"""Property tests: whatever the knobs say, the ledgers balance and the run is reproducible."""

import math
import sys

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from moonsim import FLOW_GROUPS, SimConfig, from_flat, run
from moonsim.config import KNOBS
from moonsim.presets import PRESETS

SCALAR_KNOBS = [k for k in KNOBS if k.path not in ("run.days", "run.mode")]


@st.composite
def knob_values(draw):
    """A random subset of knobs set to random in-range values (log-uniform where the knob is logarithmic)."""
    chosen = draw(st.lists(st.sampled_from(SCALAR_KNOBS), min_size=1, max_size=25, unique_by=lambda k: k.path))
    flat = {}
    for k in chosen:
        if k.kind == "bool":
            flat[k.path] = draw(st.booleans())
        elif k.kind == "choice":
            flat[k.path] = draw(st.sampled_from(k.choices))
        elif k.log and k.lo > 0:
            flat[k.path] = math.exp(draw(st.floats(math.log(k.lo), math.log(k.hi))))
        else:
            flat[k.path] = draw(st.floats(float(k.lo), float(k.hi)))
    flat["run.mode"] = draw(st.sampled_from(("token", "stablecoin")))
    flat["run.days"] = draw(st.integers(30, 60))
    return flat


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(knob_values())
def test_ledgers_balance_for_random_configs(flat):
    # strict=True raises ConservationError the moment any ledger drifts or any balance goes negative.
    res = run(from_flat(flat), strict=True)
    s = res.series
    assert res.max_conservation_error < 1e-9
    assert np.all(np.isfinite(s["price"])) and np.all(s["price"] > 0)
    assert np.all(s["pool_usd"] > 0) and np.all(s["pool_token"] > 0)
    assert np.all(s["burned"] >= 0) and np.all(np.diff(s["burned"]) >= -1e-6)
    assert np.all(s["users"] >= 0)
    # Supply only changes through explicit mints and burns.
    expected = s["total_supply"][0] + s["minted"] - s["minted"][0] - (s["burned"] - s["burned"][0])
    assert np.allclose(s["total_supply"], expected, rtol=1e-9)


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(knob_values(), st.integers(0, 2**31 - 1))
def test_same_seed_same_run(flat, seed):
    flat = {**flat, "run.seed": seed}
    a = run(from_flat(flat))
    b = run(from_flat(flat))
    for name in a.series:
        assert np.array_equal(a.series[name], b.series[name], equal_nan=True), name
    assert a.events == b.events


def test_different_seeds_differ():
    a = run(from_flat({"run.seed": 1, "run.days": 60}))
    b = run(from_flat({"run.seed": 2, "run.days": 60}))
    assert not np.array_equal(a.price, b.price)


def test_flows_explain_pool_dollars():
    """Every dollar that entered or left the pool through a trade is attributed to some group."""
    res = run(SimConfig(), strict=True)
    s = res.series
    net_flow = sum(s[f"flow_{g}"][1:] for g in FLOW_GROUPS)
    change = np.diff(s["pool_usd"])
    # Differences come only from LP adds/removals, which move both reserves at the same price.
    lp_moves = change - net_flow
    assert np.all(
        np.abs(lp_moves[s["lp_external_share"][1:] == s["lp_external_share"][:-1]]) < 1e-6 * s["pool_usd"].max()
    )


def test_presets_conserve():
    for preset in PRESETS:
        res = run(preset.config, strict=True)
        assert res.max_conservation_error < 1e-9, preset.key


def test_engine_has_no_ui_dependencies():
    for mod in [m for m in list(sys.modules) if m.split(".")[0] in ("moonsim", "streamlit", "plotly")]:
        del sys.modules[mod]
    import moonsim  # noqa: F401
    import moonsim.presets  # noqa: F401
    import moonsim.story  # noqa: F401

    assert "streamlit" not in sys.modules
    assert "plotly" not in sys.modules


def test_no_whales_keeps_the_full_launch_supply():
    cfg = from_flat({"whales.count": 0})
    res = run(cfg, strict=True)
    assert res.series["total_supply"][0] == cfg.supply.total_supply


def test_attacker_exit_value_includes_the_transfer_tax():
    from moonsim.engine import Simulation

    cfg = from_flat({"supply.transfer_tax": 0.25, "attacker.token_share": 0.1})
    sim = Simulation(cfg, strict=True)
    actual_exit = sim.attacker.usd + sim.sell(sim.attacker, sim.attacker.token, "attacker")
    assert np.isclose(actual_exit, sim.attacker_start_value, rtol=1e-12)


def test_strict_ledger_rejects_nonfinite_wallets():
    from moonsim.engine import ConservationError, Simulation

    for value in (float("nan"), float("inf"), -float("inf")):
        sim = Simulation(SimConfig(), strict=True)
        sim.holders.usd = value
        try:
            sim.check_conservation()
        except ConservationError:
            pass
        else:
            raise AssertionError(f"strict ledger accepted {value}")


def test_treasury_daily_cap_covers_buybacks_and_floor_defence_together():
    from moonsim.engine import Simulation

    for defend in (False, True):
        for limit in (0.0, 0.001, 0.2):
            cfg = from_flat(
                {
                    "treasury.band_defense": defend,
                    "treasury.floor_x_launch": 100.0,
                    "treasury.buyback_per_year": 10.0,
                    "treasury.max_daily_spend": limit,
                }
            )
            sim = Simulation(cfg, strict=True)
            before = sim.treasury.usd
            sim._act_treasury(1)
            assert before - sim.treasury.usd <= before * limit + 1e-9
            sim.check_conservation()
