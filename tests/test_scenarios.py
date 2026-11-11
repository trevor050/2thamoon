"""Scenario tests: each preset tells its story, on its own seed and on others.

The assertions are qualitative on purpose (they describe the shape of the story, not exact
numbers), and each one is checked across several seeds so a preset cannot pass by luck.
"""

import numpy as np
import pytest

from moonsim import run, story
from moonsim.presets import PRESETS, get

SEEDS = range(10)


def runs(key):
    preset = get(key)
    return [run(preset.with_seed(seed), strict=True) for seed in SEEDS]


def kinds(res):
    return {e.kind for e in res.events}


def test_every_preset_has_a_two_sentence_blurb():
    for p in PRESETS:
        assert p.blurb.count(". ") + p.blurb.endswith(".") == 2, p.key
        assert "Watch" in p.blurb, p.key


@pytest.mark.parametrize("res", runs("pump_and_dump"), ids=lambda r: f"seed{r.config.run.seed}")
def test_pump_and_dump(res):
    s = res.series
    price = s["price"]
    start_day = res.config.attacker.start_day
    peak_day = int(np.argmax(price))
    assert price[peak_day] > 3 * price[0], "the pump should at least triple the price"
    assert start_day <= peak_day <= start_day + 30, "the top should come from the attack"
    assert price[-1] < 0.2 * price[peak_day], "late buyers should be left holding the bag"
    assert s["attacker_pnl"][-1] > 0, "the attacker should walk away richer"
    assert {"attack_pump", "attack_dump"} <= kinds(res)


@pytest.mark.parametrize("res", runs("whale_exit"), ids=lambda r: f"seed{r.config.run.seed}")
def test_whale_exit_into_thin_liquidity(res):
    s = res.series
    d = res.config.whales.exit_day
    before = s["price"][d - 1]
    assert s["price"][d + res.config.whales.exit_days] < 0.3 * before, "the exit should crater the price"
    got = s["whale_usd"][-1] - s["whale_usd"][d - 1]
    paper = s["whale_token"][d - 1] * before
    assert 0 < got < 0.3 * paper, "the whale realizes a small fraction of the paper value"
    assert s["lp_external_share"][d + 10] < s["lp_external_share"][d - 1], "LPs should flee"
    assert "whale_exit" in kinds(res)


@pytest.mark.parametrize("res", runs("death_spiral"), ids=lambda r: f"seed{r.config.run.seed}")
def test_algorithmic_stablecoin_death_spiral(res):
    s = res.series
    rel = s["price"] / s["peg"]
    assert rel[-1] < 0.2, "the peg should be gone"
    assert s["protocol_failed"][-1], "the showcase should reach terminal governance failure"
    assert res.protocol_failure_day is not None
    assert np.all(s["protocol_failed"][res.protocol_failure_day :])
    assert sum(e.kind == "protocol_failed" for e in res.events) == 1
    assert s["gov_supply"][-1] > 10 * s["gov_supply"][0], "redemptions should hyperinflate the governance token"
    assert s["gov_price"][-1] < 0.01 * s["gov_price"][0]
    assert s["treasury_usd"][-1] < 0.05 * s["treasury_usd"][0], "the defence fund should be spent"
    assert {"depeg", "gov_inflation", "treasury_drained"} <= kinds(res)
    assert "peg broke" in story.headline(res)


@pytest.mark.parametrize("res", runs("peg_defense"), ids=lambda r: f"seed{r.config.run.seed}")
def test_successful_peg_defense(res):
    s = res.series
    rel = s["price"] / s["peg"]
    attack = res.config.attacker.start_day
    assert rel[attack:].min() < 0.99, "the attack should visibly hit the peg"
    assert abs(rel[-1] - 1.0) < 0.02, "the peg should be restored"
    assert np.all(np.abs(rel[attack + 15 :] - 1.0) < 0.02), "and stay restored"
    assert s["reserve_usd"][-1] > 0.5 * s["reserve_usd"][0], "the reserve should survive"
    assert not s["protocol_failed"].any(), "the fully backed companion must not fail"
    assert s["gov_supply"][-1] == pytest.approx(s["gov_supply"][0]), "no governance tokens minted"


def test_same_attack_two_endings():
    """The two stablecoin presets differ only in backing; that difference alone decides the outcome."""
    spiral, defense = get("death_spiral").overrides, get("peg_defense").overrides
    differing = {k for k in set(spiral) | set(defense) if spiral.get(k) != defense.get(k)}
    assert differing == {"stable.reserve_ratio", "stable.collateral_ratio"}


@pytest.mark.parametrize("res", runs("gold_peg"), ids=lambda r: f"seed{r.config.run.seed}")
def test_gold_peg_tracks_gold_and_backing_moves_inversely(res):
    s = res.series
    rel = s["price"] / s["peg"]
    assert np.all(np.abs(rel[5:] - 1) < 0.05), "a fully backed coin should track gold closely"
    # Dollar reserve vs gold-denominated liabilities: collateral moves against gold.
    gold_up = s["peg"][-1] > s["peg"][0]
    cr_down = s["collateral_ratio"][-1] < s["collateral_ratio"][0]
    assert gold_up == cr_down


def test_gold_peg_default_seed_shows_the_problem():
    res = run(get("gold_peg").config)
    assert res.series["peg"][-1] > 1.2
    assert res.series["collateral_ratio"][-1] < 0.85


@pytest.mark.parametrize("res", runs("unlock_cliff"), ids=lambda r: f"seed{r.config.run.seed}")
def test_unlock_cliff(res):
    s = res.series
    cliff = res.config.supply.vesting_cliff_days
    assert s["price"][cliff - 1] > 1.3 * s["price"][0], "the hype phase should run up first"
    assert s["price"][-1] < 0.8 * s["price"][cliff - 1], "insider selling should bleed it after the cliff"
    assert s["locked"][cliff] < s["locked"][cliff - 1]
    assert "unlock" in kinds(res)


def test_sandbox_is_a_mixed_bag():
    """The default world is not rigged: across seeds some runs end up and some end down."""
    ends = [run(get("sandbox").with_seed(seed)).series["price"][-1] / 0.1 for seed in range(12)]
    assert min(ends) < 0.8 and max(ends) > 1.2


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.key)
def test_all_presets_are_deterministic_and_conserve_over_ten_seeds(preset):
    for seed in SEEDS:
        a = run(preset.with_seed(seed), strict=True)
        b = run(preset.with_seed(seed), strict=True)
        assert a.max_conservation_error < 1e-9
        assert a.inapplicable_series == b.inapplicable_series
        for key in a.applicable_series:
            assert np.array_equal(a.series[key], b.series[key]), (seed, key)
        for key in a.inapplicable_series:
            assert np.isnan(a.series[key]).all() and np.isnan(b.series[key]).all()
        assert a.events == b.events
