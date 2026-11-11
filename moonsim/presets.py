"""Named scenarios a visitor can load with one click.

Each preset is a plain set of knob overrides on top of the defaults, so anything a preset
does can be reproduced by hand in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple

from .config import SimConfig, from_flat


@dataclass(frozen=True)
class Preset:
    key: str
    title: str
    blurb: str  # two sentences: what is going on and what to watch
    overrides: Dict[str, Any]

    @property
    def config(self) -> SimConfig:
        return from_flat(self.overrides)

    def with_seed(self, seed: int) -> SimConfig:
        cfg = self.config
        return replace(cfg, run=replace(cfg.run, seed=int(seed)))


_QUIET_WORLD = {
    # Shared by scenarios that want one mechanism on stage without random macro noise.
    "market.regimes": False,
    "market.shock_per_year": 0.0,
    "market.hype_per_year": 1.0,
}

_STABLE_BASE = {
    "run.mode": "stablecoin",
    "supply.total_supply": 100_000_000.0,
    "supply.holders_share": 0.82,
    "supply.whales_share": 0.08,
    "supply.insiders_share": 0.0,
    "supply.emission_apy": 0.195,
    "liquidity.pool_usd": 10_000_000.0,
    "liquidity.protocol_owned_share": 0.2,
    "crowd.initial_users": 50_000.0,
    "crowd.market_potential": 400_000.0,
    "crowd.ticket_usd": 2_000.0,
    "crowd.conviction_start": 0.8,
    "whales.count": 5,
    "whales.take_profit_multiple": 1000.0,
    "traders.momentum_capital": 200_000.0,
    "traders.noise_capital": 300_000.0,
    "traders.noise_intensity": 0.05,
    "stable.max_redeem_per_day": 0.03,
    "stable.arb_capital": 2_000_000.0,
    "stable.gov_price": 10.0,
    "stable.gov_supply": 50_000_000.0,
    "stable.gov_pool_usd": 2_000_000.0,
    "attacker.style": "bear_raid",
    "attacker.capital_usd": 0.0,
    "attacker.token_share": 0.1,
    "attacker.start_day": 15,
    "attacker.raid_days": 2,
    "attacker.raid_rebuy_drawdown": 0.9,
    "market.regimes": False,
    "market.shock_per_year": 0.0,
    "market.hype_per_year": 0.0,
    "run.days": 120,
}

PRESETS: Tuple[Preset, ...] = (
    Preset(
        key="sandbox",
        title="Fair launch sandbox",
        blurb=(
            "A small token launches into a $250k pool with real holders, whales, insiders on a vesting schedule, "
            "and a market that drifts between bull and bear. Watch who is buying and selling each day, then change "
            "the seed: the same rules can moon or die on luck alone."
        ),
        overrides={"run.seed": 7},
    ),
    Preset(
        key="pump_and_dump",
        title="Pump and dump",
        blurb=(
            "An attacker quietly buys into a thin pool, then buys hard and pays for a shill campaign so momentum "
            "traders and fresh newcomers pile in. Watch the attacker sell into that FOMO and leave the late "
            "buyers holding the bag."
        ),
        overrides={
            **_QUIET_WORLD,
            "run.days": 150,
            "run.seed": 3,
            "liquidity.pool_usd": 120_000.0,
            "supply.insiders_share": 0.0,
            "supply.whales_share": 0.0,
            "whales.count": 0,
            "traders.momentum_capital": 300_000.0,
            "traders.momentum_sensitivity": 12.0,
            "attacker.style": "pump_and_dump",
            "attacker.capital_usd": 100_000.0,
            "attacker.token_share": 0.0,
            "attacker.start_day": 10,
            "attacker.accumulate_days": 10,
            "attacker.accumulate_share": 0.8,
            "attacker.pump_days": 6,
            "attacker.promo_hype": 6.0,
            "attacker.dump_multiple": 1.6,
            "attacker.dump_days": 6,
        },
    ),
    Preset(
        key="whale_exit",
        title="Whale exit into thin liquidity",
        blurb=(
            "One whale owns 8% of the supply and decides to cash out over five days into a pool with only "
            "$150k of dollars in it. Watch how little the whale actually gets compared to the paper value of the "
            "bag, and how the crash scares the LPs and the crowd into making it worse."
        ),
        overrides={
            **_QUIET_WORLD,
            "run.days": 120,
            "run.seed": 11,
            "liquidity.pool_usd": 150_000.0,
            "supply.holders_share": 0.25,
            "supply.whales_share": 0.08,
            "supply.insiders_share": 0.0,
            "whales.count": 1,
            "whales.take_profit_multiple": 1000.0,
            "whales.exit_day": 40,
            "whales.exit_fraction": 1.0,
            "whales.exit_days": 5,
        },
    ),
    Preset(
        key="death_spiral",
        title="Algorithmic stablecoin death spiral",
        blurb=(
            "A stablecoin backed only by its own governance token (Terra style) pays 19.5% yield, and "
            "a two-day raid starts on day 15 with 9% of launch supply. Watch arbitrageurs redeem coins for freshly "
            "minted governance tokens and "
            "dump them, which crashes the backing, which makes holders run, which mints even more."
        ),
        overrides={
            **_STABLE_BASE,
            "run.seed": 5,
            "stable.reserve_ratio": 0.0,
            "stable.collateral_ratio": 0.0,
            "treasury.usd": 3_000_000.0,
            "treasury.max_daily_spend": 0.35,
        },
    ),
    Preset(
        key="peg_defense",
        title="Peg defence that holds",
        blurb=(
            "Same coin, same yield, same day-15 attack, but every coin is backed by a full dollar in reserve and "
            "redemptions pay out in those dollars. Watch the price dip, arbitrageurs redeem against the reserve, "
            "and the peg snap back while the governance token barely notices."
        ),
        overrides={
            **_STABLE_BASE,
            "run.seed": 5,
            "stable.reserve_ratio": 1.0,
            "stable.collateral_ratio": 1.0,
            "treasury.usd": 3_000_000.0,
            "treasury.max_daily_spend": 0.35,
        },
    ),
    Preset(
        key="gold_peg",
        title="The original: a gold-pegged coin",
        blurb=(
            "This is where 2thamoon started: a coin that tracks the price of gold, fully backed, but backed in "
            "dollars. Watch the collateral ratio: when gold rallies the coin's liabilities grow and the dollar "
            "reserve does not, so a fully backed coin quietly becomes an under-backed one."
        ),
        overrides={
            **_STABLE_BASE,
            "run.seed": 4,
            "run.days": 365,
            "treasury.usd": 0.0,
            "attacker.style": "off",
            "attacker.token_share": 0.0,
            "supply.emission_apy": 0.03,
            "stable.peg_target": "gold",
            "stable.gold_drift": 0.08,
            "stable.gold_vol": 0.18,
            "stable.reserve_ratio": 1.0,
            "stable.collateral_ratio": 1.0,
        },
    ),
    Preset(
        key="unlock_cliff",
        title="Vesting cliff dump",
        blurb=(
            "A hyped token rides a bull market while 30% of supply sits locked for the team and early investors. "
            "Watch the day-120 cliff hand insiders a quarter of their bag at once, and how steady insider selling "
            "turns a chart that only went up into a slow bleed."
        ),
        overrides={
            "run.days": 240,
            "run.seed": 2,
            "market.start_regime": "bull",
            "market.bull_days": 600.0,
            "market.shock_per_year": 0.0,
            "supply.insiders_share": 0.3,
            "supply.holders_share": 0.25,
            "supply.vesting_cliff_days": 120,
            "supply.vesting_days": 480,
            "supply.insider_sell_rate": 0.03,
        },
    ),
)

PRESET_BY_KEY: Dict[str, Preset] = {p.key: p for p in PRESETS}


def get(key: str) -> Preset:
    return PRESET_BY_KEY[key]
