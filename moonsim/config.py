"""Typed, validated configuration for the simulation.

Every tunable number lives here as a frozen dataclass field built with :func:`knob`.
The metadata on each knob (label, range, help text, whether it is an "advanced" knob)
is what the Streamlit front end uses to render controls, so adding a knob here is
enough to expose it in the UI.

Units: prices and balances are in USD and whole tokens. One simulation step is one day.
Rates that say "per year" are converted to per-day inside the engine.
"""

from __future__ import annotations

import math
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any, Dict, Iterator, List, Tuple

DAYS_PER_YEAR = 365.0


def knob(
    default: Any,
    lo: float | None = None,
    hi: float | None = None,
    *,
    label: str,
    help: str = "",
    step: float | None = None,
    advanced: bool = False,
    fmt: str | None = None,
    choices: Tuple[str, ...] | None = None,
    log: bool = False,
) -> Any:
    """Declare a config field with UI/validation metadata."""
    meta = {
        "lo": lo,
        "hi": hi,
        "label": label,
        "help": help,
        "step": step,
        "advanced": advanced,
        "fmt": fmt,
        "choices": choices,
        "log": log,
    }
    return field(default=default, metadata=meta)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    mode: str = knob(
        "token",
        label="Mode",
        choices=("token", "stablecoin"),
        help="token: a free-floating token. stablecoin: an algorithmic stablecoin with a "
        "mint/redeem peg mechanism and a sister governance token that absorbs losses.",
    )
    days: int = knob(365, 30, 1095, label="Days to simulate", step=5)
    seed: int = knob(
        7, 0, 1_000_000, label="Random seed", step=1, help="Same seed + same knobs = the exact same run, every time."
    )


@dataclass(frozen=True)
class SupplyConfig:
    total_supply: float = knob(
        100_000_000.0,
        1_000_000.0,
        10_000_000_000.0,
        label="Total supply at launch",
        step=1_000_000.0,
        fmt="%.0f",
        log=True,
    )
    launch_price: float = knob(
        0.10,
        0.0001,
        1000.0,
        label="Launch price (USD)",
        step=0.01,
        fmt="%.4f",
        log=True,
        help="Opening pool price. Ignored in stablecoin mode, where the coin launches at its peg.",
    )
    holders_share: float = knob(
        0.25,
        0.0,
        0.9,
        label="Share held by retail holders",
        step=0.01,
        help="Fraction of supply that the starting user base already holds.",
    )
    whales_share: float = knob(0.15, 0.0, 0.9, label="Share held by whales", step=0.01)
    insiders_share: float = knob(
        0.15,
        0.0,
        0.9,
        label="Share locked for team/investors",
        step=0.01,
        help="Locked at launch, then vests to insiders who tend to sell.",
    )
    vesting_cliff_days: int = knob(
        180,
        0,
        1095,
        label="Vesting cliff (days)",
        step=5,
        advanced=True,
        help="Nothing unlocks before the cliff. At the cliff, the elapsed share unlocks at once.",
    )
    vesting_days: int = knob(720, 1, 2000, label="Vesting duration (days)", step=10, advanced=True)
    insider_sell_rate: float = knob(
        0.01,
        0.0,
        0.5,
        label="Insider daily sell rate",
        step=0.005,
        advanced=True,
        help="Fraction of their unlocked tokens insiders sell each day.",
    )
    emission_apy: float = knob(
        0.0,
        0.0,
        2.0,
        label="Emission / yield rate (per year)",
        step=0.01,
        help="New tokens minted to holders as staking rewards (token mode) or savings "
        "yield (stablecoin mode). Minted, so it dilutes everyone.",
    )
    emission_halving_days: int = knob(
        0, 0, 1095, label="Emission halving interval (days, 0 = never)", step=15, advanced=True
    )
    max_supply: float = knob(
        0.0,
        0.0,
        100_000_000_000.0,
        label="Hard cap on supply (0 = none)",
        step=1_000_000.0,
        fmt="%.0f",
        advanced=True,
        help="Emissions stop once total minted supply reaches this.",
    )
    transfer_tax: float = knob(
        0.0,
        0.0,
        0.25,
        label="Transfer tax on trades",
        step=0.0025,
        fmt="%.4f",
        help="Taken in tokens from every buy and sell on the pool.",
    )
    tax_burn_share: float = knob(
        0.5,
        0.0,
        1.0,
        label="Share of tax burned",
        step=0.05,
        advanced=True,
        help="The rest of the tax goes to the treasury.",
    )


@dataclass(frozen=True)
class LiquidityConfig:
    pool_usd: float = knob(
        250_000.0,
        1_000.0,
        1_000_000_000.0,
        label="Pool depth at launch (USD side)",
        step=10_000.0,
        fmt="%.0f",
        log=True,
        help="USD seeded into the constant-product pool. The token side is pool_usd / price, "
        "so the pool holds about 2x this in total value.",
    )
    lp_fee: float = knob(
        0.003,
        0.0,
        0.05,
        label="Pool swap fee",
        step=0.0005,
        fmt="%.4f",
        advanced=True,
        help="Paid by every trade and left in the pool for liquidity providers.",
    )
    protocol_owned_share: float = knob(
        0.3,
        0.0,
        1.0,
        label="Protocol-owned share of liquidity",
        step=0.05,
        help="LP shares owned by the treasury. Those never flee.",
    )
    lp_flight_vol: float = knob(
        0.15,
        0.005,
        1.0,
        label="LP flight trigger: daily volatility",
        step=0.005,
        advanced=True,
        help="When the 7-day realised volatility of daily returns exceeds this, outside LPs start pulling liquidity.",
    )
    lp_flight_drawdown: float = knob(0.6, 0.02, 1.0, label="LP flight trigger: drawdown", step=0.02, advanced=True)
    lp_flight_rate: float = knob(0.08, 0.0, 1.0, label="LP withdrawal per stressed day", step=0.01, advanced=True)
    lp_return_rate: float = knob(
        0.02,
        0.0,
        1.0,
        label="LP re-entry per calm day",
        step=0.01,
        advanced=True,
        help="Fraction of idle LP capital re-added when things are calm.",
    )
    lp_dump_share: float = knob(0.3, 0.0, 1.0, label="Share of withdrawn tokens LPs sell", step=0.05, advanced=True)


@dataclass(frozen=True)
class CrowdConfig:
    initial_users: float = knob(2_000.0, 1.0, 10_000_000.0, label="Holders at launch", step=100.0, fmt="%.0f", log=True)
    market_potential: float = knob(
        60_000.0,
        10.0,
        100_000_000.0,
        label="Potential users (market size)",
        step=1_000.0,
        fmt="%.0f",
        log=True,
        help="The ceiling of the Bass adoption curve: how many people could ever care.",
    )
    bass_p: float = knob(
        0.05,
        0.0,
        5.0,
        label="Adoption: outside discovery (per year)",
        step=0.01,
        help="Bass model 'innovation' coefficient: people who find it on their own.",
    )
    bass_q: float = knob(
        1.2,
        0.0,
        20.0,
        label="Adoption: word of mouth (per year)",
        step=0.05,
        help="Bass model 'imitation' coefficient: people who join because others did.",
    )
    fomo: float = knob(
        1.5,
        0.0,
        10.0,
        label="Price FOMO on adoption",
        step=0.1,
        advanced=True,
        help="How much a rising 30-day price pulls in extra newcomers (and a falling one repels them).",
    )
    ticket_usd: float = knob(
        600.0, 1.0, 1_000_000.0, label="Average newcomer budget (USD)", step=50.0, fmt="%.0f", log=True
    )
    newcomer_speed: float = knob(
        0.5,
        0.0,
        1.0,
        label="Newcomer buying speed",
        step=0.05,
        advanced=True,
        help="Share of their budget newcomers try to put to work each day. "
        "Whatever they do not buy within a few days joins the crowd's idle cash.",
    )
    ticket_noise: float = knob(0.3, 0.0, 2.0, label="Newcomer budget noise", step=0.05, advanced=True)
    churn_per_year: float = knob(
        0.6,
        0.0,
        20.0,
        label="Baseline churn (per year)",
        step=0.05,
        advanced=True,
        help="Share of holders who drift away each year when nothing is wrong.",
    )
    churn_fear_mult: float = knob(
        2.0,
        0.0,
        50.0,
        label="Fear multiplier on churn",
        step=0.5,
        advanced=True,
        help="Churn rate is multiplied by (1 + this x fear).",
    )
    max_alloc: float = knob(
        0.9,
        0.05,
        1.0,
        label="Max share of wallet in the token",
        step=0.05,
        advanced=True,
        help="At full conviction, holders keep this share of their wallet in the token.",
    )
    conviction_start: float = knob(0.75, 0.01, 0.99, label="Starting conviction", step=0.01)
    rebalance_speed: float = knob(
        0.08,
        0.001,
        1.0,
        label="Holder trading speed",
        step=0.01,
        advanced=True,
        help="Fraction of the gap to their target position holders close each day.",
    )
    slippage: float = knob(
        0.03,
        0.001,
        1.0,
        label="Holder slippage tolerance",
        step=0.005,
        fmt="%.3f",
        advanced=True,
        help="Calm holders stop an order once it would move the price this much. "
        "Fear widens it, up to the panic tolerance.",
    )
    panic_slippage: float = knob(
        0.2,
        0.001,
        1.0,
        label="Panic slippage tolerance",
        step=0.01,
        advanced=True,
        help="How much price impact scared sellers will eat in one order.",
    )
    greed_speed: float = knob(0.05, 0.0, 1.0, label="Conviction gain speed", step=0.01, advanced=True)
    fear_speed: float = knob(
        0.2,
        0.0,
        1.0,
        label="Conviction loss speed",
        step=0.01,
        advanced=True,
        help="Fear moves faster than greed. This is how fast conviction drops toward its target.",
    )
    return_sensitivity: float = knob(2.0, 0.0, 50.0, label="Sensitivity to recent gains", step=0.5, advanced=True)
    fear_threshold: float = knob(
        0.3,
        0.0,
        1.0,
        label="Drawdown people shrug off",
        step=0.01,
        help="Token mode: drawdown from the recent high before holders start to worry.",
    )
    memory_days: float = knob(
        40.0,
        1.0,
        365.0,
        label="Price memory (days)",
        step=1.0,
        advanced=True,
        help="Drawdowns are measured from a remembered high that fades with this time "
        "constant. Short memory means people re-anchor quickly after a crash.",
    )
    diamond_hands: float = knob(
        0.15,
        0.0,
        0.99,
        label="Diamond hands",
        step=0.01,
        advanced=True,
        help="Floor on conviction: the share of the crowd's appetite that no amount of fear removes.",
    )
    dip_buying: float = knob(
        1.0,
        0.0,
        20.0,
        label="Buy-the-dip instinct",
        step=0.1,
        advanced=True,
        help="Token mode: small drawdowns (up to the shrug-off level) raise conviction. "
        "Bigger ones are handled by fear.",
    )
    fear_sensitivity: float = knob(3.0, 0.0, 50.0, label="Fear sensitivity", step=0.5, advanced=True)
    panic_threshold: float = knob(
        0.6, 0.0, 1.0, label="Panic drawdown", step=0.01, help="Beyond this drawdown holders panic and fear jumps."
    )
    panic_intensity: float = knob(3.0, 0.0, 30.0, label="Panic intensity", step=0.5, advanced=True)
    depeg_tolerance: float = knob(
        0.01,
        0.0,
        0.5,
        label="Depeg people shrug off",
        step=0.005,
        fmt="%.3f",
        help="Stablecoin mode: how far below peg before holders start to worry.",
    )
    depeg_panic: float = knob(
        0.05,
        0.0,
        0.9,
        label="Depeg that triggers panic",
        step=0.005,
        fmt="%.3f",
        advanced=True,
        help="Stablecoin mode: below peg by this much, holders and LPs panic.",
    )
    depeg_sensitivity: float = knob(
        60.0,
        0.0,
        500.0,
        label="Depeg fear sensitivity",
        step=5.0,
        advanced=True,
        help="Fear per unit of depeg beyond tolerance (0.05 x 60 = 3).",
    )
    backing_sensitivity: float = knob(
        3.0,
        0.0,
        50.0,
        label="Fear of thin backing",
        step=0.5,
        advanced=True,
        help="Stablecoin mode: fear per unit of backing shortfall, counted only once the coin is visibly off peg.",
    )
    dilution_fear: float = knob(
        3.0,
        0.0,
        50.0,
        label="Governance holders' fear of dilution",
        step=0.5,
        advanced=True,
        help="Stablecoin mode: fear per unit of log growth in governance supply from redemptions.",
    )
    yield_sensitivity: float = knob(
        4.0,
        0.0,
        50.0,
        label="Love of yield",
        step=0.5,
        advanced=True,
        help="Stablecoin mode: conviction boost per unit of yield APY.",
    )


@dataclass(frozen=True)
class TraderConfig:
    momentum_capital: float = knob(
        60_000.0, 0.0, 1_000_000_000.0, label="Momentum trader capital (USD)", step=10_000.0, fmt="%.0f", log=True
    )
    momentum_fast_days: float = knob(3.0, 1.0, 60.0, label="Momentum fast average (days)", step=1.0, advanced=True)
    momentum_slow_days: float = knob(12.0, 2.0, 200.0, label="Momentum slow average (days)", step=1.0, advanced=True)
    momentum_sensitivity: float = knob(8.0, 0.0, 100.0, label="Momentum sensitivity", step=0.5, advanced=True)
    momentum_speed: float = knob(0.35, 0.0, 1.0, label="Momentum trading speed", step=0.05, advanced=True)
    momentum_max_alloc: float = knob(0.9, 0.0, 1.0, label="Momentum max allocation", step=0.05, advanced=True)
    momentum_slippage: float = knob(
        0.1, 0.001, 1.0, label="Momentum slippage tolerance", step=0.005, fmt="%.3f", advanced=True
    )
    noise_capital: float = knob(
        80_000.0,
        0.0,
        1_000_000_000.0,
        label="Noise trader capital (USD)",
        step=10_000.0,
        fmt="%.0f",
        log=True,
        help="Random two-sided flow. Gives the chart texture and the pool volume.",
    )
    noise_intensity: float = knob(0.08, 0.0, 1.0, label="Noise trade size (share of capital)", step=0.01, advanced=True)
    noise_slippage: float = knob(
        0.02, 0.001, 1.0, label="Noise slippage tolerance", step=0.005, fmt="%.3f", advanced=True
    )


@dataclass(frozen=True)
class WhaleConfig:
    count: int = knob(4, 0, 50, label="Number of whales", step=1)
    basis_multiple: float = knob(
        0.25,
        0.01,
        10.0,
        label="Whale entry price (x launch)",
        step=0.05,
        advanced=True,
        help="Early whales usually got in cheaper than the public launch price.",
    )
    take_profit_multiple: float = knob(
        8.0,
        1.0,
        1000.0,
        label="Whale take-profit multiple",
        step=0.5,
        help="Whales start selling once price is this many times their entry.",
    )
    take_profit_rate: float = knob(0.02, 0.0, 1.0, label="Whale take-profit rate (per day)", step=0.005, advanced=True)
    slippage: float = knob(
        0.05,
        0.001,
        1.0,
        label="Whale/insider slippage tolerance",
        step=0.005,
        fmt="%.3f",
        advanced=True,
        help="Patient sellers split orders. A scripted exit ignores this.",
    )
    panic_drawdown: float = knob(0.7, 0.0, 1.0, label="Whale panic drawdown", step=0.05, advanced=True)
    panic_rate: float = knob(0.03, 0.0, 1.0, label="Whale panic sell rate (per day)", step=0.01, advanced=True)
    exit_day: int = knob(
        -1,
        -1,
        1095,
        label="Scripted whale exit day (-1 = off)",
        step=1,
        help="One whale dumps on purpose starting this day.",
    )
    exit_fraction: float = knob(1.0, 0.0, 1.0, label="Share of that whale's bag sold", step=0.05)
    exit_days: int = knob(3, 1, 120, label="Days the exit is spread over", step=1)


@dataclass(frozen=True)
class ManualOrder:
    day: int = 0
    days: int = 1
    side: str = "sell"  # "buy" or "sell"
    usd: float = 0.0  # total USD to buy, or USD-equivalent (at order time) to sell


@dataclass(frozen=True)
class AttackerConfig:
    style: str = knob(
        "off",
        label="Attacker strategy",
        choices=("off", "pump_and_dump", "bear_raid", "manual"),
        help="pump_and_dump: quietly accumulate, buy hard plus paid hype, sell into the FOMO. "
        "bear_raid: dump a big bag to trigger panic, buy back lower. manual: your own order list.",
    )
    capital_usd: float = knob(
        400_000.0, 0.0, 10_000_000_000.0, label="Attacker cash (USD)", step=10_000.0, fmt="%.0f", log=True
    )
    token_share: float = knob(0.0, 0.0, 0.9, label="Attacker starting bag (share of supply)", step=0.01)
    start_day: int = knob(30, 0, 1095, label="Attack starts on day", step=1)
    accumulate_days: int = knob(20, 0, 365, label="Quiet accumulation (days)", step=1, advanced=True)
    accumulate_share: float = knob(0.35, 0.0, 1.0, label="Share of cash spent accumulating", step=0.05, advanced=True)
    pump_days: int = knob(6, 1, 120, label="Pump phase (days)", step=1, advanced=True)
    promo_hype: float = knob(
        2.5,
        0.0,
        20.0,
        label="Paid hype at pump start",
        step=0.25,
        help="Requested shill campaign. Hype is scaled down to what the attacker's remaining cash can buy.",
    )
    promo_usd_per_hype: float = knob(
        1_000.0,
        1.0,
        1_000_000.0,
        label="Promotion cost (USD per hype unit)",
        step=100.0,
        fmt="%.0f",
        log=True,
        advanced=True,
        help="Paid once at pump start from attacker cash. Dollars leave the economy as external promotion spend "
        "and reduce attacker profit; zero cash buys zero hype.",
    )
    dump_multiple: float = knob(2.5, 1.0, 100.0, label="Dump once price is this x their cost", step=0.25, advanced=True)
    dump_days: int = knob(
        4,
        1,
        60,
        label="Dump phase (days)",
        step=1,
        advanced=True,
        help="Target pace. A careful dumper also respects the slippage limit, so it can take longer.",
    )
    dump_slippage: float = knob(
        0.1,
        0.001,
        1.0,
        label="Dump slippage limit",
        step=0.01,
        advanced=True,
        help="Max price impact per sell. Smart dumpers sell into the buyers instead of nuking their own exit.",
    )
    raid_days: int = knob(2, 1, 60, label="Bear raid: days to dump the bag", step=1, advanced=True)
    raid_rebuy_drawdown: float = knob(
        0.35, 0.0, 0.99, label="Bear raid: rebuy after this drop", step=0.05, advanced=True
    )
    manual_orders: Tuple[ManualOrder, ...] = field(default=(), metadata={"label": "Manual orders", "advanced": True})


@dataclass(frozen=True)
class TreasuryConfig:
    usd: float = knob(
        100_000.0,
        0.0,
        10_000_000_000.0,
        label="Treasury cash (USD)",
        step=10_000.0,
        fmt="%.0f",
        log=True,
        help="In stablecoin mode this is the peg defence fund, separate from the reserve.",
    )
    band_defense: bool = knob(
        False,
        label="Defend a price band",
        help="Treasury buys when price is under the floor and sells its tokens above the "
        "ceiling, trading on the same pool as everyone else.",
    )
    floor_x_launch: float = knob(0.5, 0.0, 100.0, label="Floor (x launch price)", step=0.05)
    ceiling_x_launch: float = knob(
        0.0, 0.0, 1000.0, label="Ceiling (x launch price, 0 = none)", step=0.5, advanced=True
    )
    max_daily_spend: float = knob(0.2, 0.0, 1.0, label="Max share of cash spent per day", step=0.05, advanced=True)
    buyback_per_year: float = knob(
        0.0,
        0.0,
        10.0,
        label="Buyback rate (share of cash per year)",
        step=0.1,
        help="Steady buybacks regardless of price.",
    )
    burn_buybacks: bool = knob(True, label="Burn bought-back tokens", advanced=True)
    circuit_breaker: bool = knob(
        False,
        label="Circuit breaker",
        help="Pause all trading for a few days after a big one-day move. Real AMMs "
        "can only do this if the token contract is pausable.",
    )
    breaker_move: float = knob(0.3, 0.01, 0.99, label="Breaker trip size (one-day move)", step=0.01, advanced=True)
    breaker_days: int = knob(2, 1, 30, label="Breaker pause (days)", step=1, advanced=True)


@dataclass(frozen=True)
class StableConfig:
    peg_target: str = knob(
        "usd",
        label="Peg target",
        choices=("usd", "gold"),
        help="usd: a fixed 1.00 dollar. gold: tracks a simulated gold price (the original "
        "2thamoon idea), while reserves are still held in dollars.",
    )
    gold_drift: float = knob(0.03, -0.5, 0.5, label="Gold drift (per year)", step=0.01, advanced=True)
    gold_vol: float = knob(0.15, 0.0, 1.0, label="Gold volatility (per year)", step=0.01, advanced=True)
    reserve_ratio: float = knob(
        1.0,
        0.0,
        2.0,
        label="Starting reserve (x supply)",
        step=0.05,
        help="Dollar reserve at launch as a fraction of the coin's market value.",
    )
    collateral_ratio: float = knob(
        1.0,
        0.0,
        1.0,
        label="Collateral ratio paid on redemption",
        step=0.05,
        help="Redeeming 1 coin pays this share in reserve dollars and the rest in freshly "
        "minted governance tokens (FRAX v1 style). 0 = fully algorithmic (Terra style).",
    )
    fallback_to_gov: bool = knob(
        True,
        label="Pay in governance tokens when the reserve is empty",
        help="If off, redemptions stop when the reserve runs dry.",
    )
    redeem_fee: float = knob(0.003, 0.0, 0.1, label="Redeem fee", step=0.001, fmt="%.4f", advanced=True)
    mint_fee: float = knob(0.003, 0.0, 0.1, label="Mint fee", step=0.001, fmt="%.4f", advanced=True)
    max_redeem_per_day: float = knob(
        0.1,
        0.0,
        1.0,
        label="Redemption cap (share of supply per day)",
        step=0.01,
        help="Protocol limit on daily redemptions. Terra's swap pool had one too.",
    )
    arb_capital: float = knob(
        2_000_000.0, 0.0, 10_000_000_000.0, label="Arbitrageur capital (USD)", step=100_000.0, fmt="%.0f", log=True
    )
    arb_margin: float = knob(0.002, 0.0, 0.1, label="Arbitrageur minimum edge", step=0.001, fmt="%.4f", advanced=True)
    gov_price: float = knob(10.0, 0.0001, 10_000.0, label="Governance token price", step=0.5, log=True, advanced=True)
    gov_supply: float = knob(
        50_000_000.0,
        1_000.0,
        100_000_000_000.0,
        label="Governance token supply",
        step=1_000_000.0,
        fmt="%.0f",
        log=True,
        advanced=True,
    )
    gov_pool_usd: float = knob(
        5_000_000.0,
        1_000.0,
        10_000_000_000.0,
        label="Governance pool depth (USD)",
        step=100_000.0,
        fmt="%.0f",
        log=True,
        help="Thin governance liquidity is what turns a depeg into a death spiral.",
    )
    gov_backing_weight: float = knob(
        1.0, 0.0, 1.0, label="How much holders count gov tokens as backing", step=0.05, advanced=True
    )
    gov_holder_usd: float = knob(
        2_000_000.0,
        0.0,
        10_000_000_000.0,
        label="Governance holders' spare cash",
        step=100_000.0,
        fmt="%.0f",
        log=True,
        advanced=True,
    )
    defense_band: float = knob(
        0.02,
        0.0,
        0.5,
        label="Defence fund buys below peg minus",
        step=0.005,
        fmt="%.3f",
        help="The treasury (defence fund) buys the coin on the pool when it trades this far below peg.",
    )


@dataclass(frozen=True)
class MarketConfig:
    regimes: bool = knob(True, label="Bull / bear regime shifts")
    start_regime: str = knob("neutral", label="Starting regime", choices=("bull", "neutral", "bear"))
    bull_days: float = knob(120.0, 1.0, 2000.0, label="Average bull run (days)", step=5.0, advanced=True)
    neutral_days: float = knob(90.0, 1.0, 2000.0, label="Average sideways stretch (days)", step=5.0, advanced=True)
    bear_days: float = knob(100.0, 1.0, 2000.0, label="Average bear market (days)", step=5.0, advanced=True)
    bull_bias: float = knob(0.5, 0.0, 1.0, label="Odds the next regime is bull", step=0.05, advanced=True)
    bull_adoption: float = knob(1.6, 0.0, 10.0, label="Bull: adoption multiplier", step=0.1, advanced=True)
    bear_adoption: float = knob(0.4, 0.0, 10.0, label="Bear: adoption multiplier", step=0.1, advanced=True)
    bull_ticket: float = knob(1.4, 0.0, 10.0, label="Bull: newcomer budget multiplier", step=0.1, advanced=True)
    bear_ticket: float = knob(0.6, 0.0, 10.0, label="Bear: newcomer budget multiplier", step=0.1, advanced=True)
    bull_mood: float = knob(0.4, -5.0, 5.0, label="Bull: conviction boost", step=0.1, advanced=True)
    bear_mood: float = knob(-0.3, -5.0, 5.0, label="Bear: conviction hit", step=0.1, advanced=True)
    bear_churn: float = knob(2.0, 0.0, 20.0, label="Bear: churn multiplier", step=0.1, advanced=True)
    bear_lp_stress: float = knob(1.5, 0.1, 10.0, label="Bear: LP nervousness multiplier", step=0.1, advanced=True)
    hype_per_year: float = knob(
        4.0,
        0.0,
        100.0,
        label="Hype waves per year",
        step=0.5,
        help="Random news / influencer moments. Each bumps the hype index, which decays.",
    )
    hype_size: float = knob(1.0, 0.0, 20.0, label="Typical hype wave size", step=0.1)
    hype_half_life: float = knob(5.0, 0.5, 120.0, label="Hype half-life (days)", step=0.5, advanced=True)
    viral_chance: float = knob(0.1, 0.0, 1.0, label="Chance a wave goes viral", step=0.05, advanced=True)
    viral_mult: float = knob(4.0, 1.0, 50.0, label="Viral wave multiplier", step=0.5, advanced=True)
    hype_adoption: float = knob(1.0, 0.0, 20.0, label="Hype effect on adoption", step=0.1, advanced=True)
    hype_mood: float = knob(0.5, 0.0, 10.0, label="Hype effect on conviction", step=0.1, advanced=True)
    shock_per_year: float = knob(
        1.0,
        0.0,
        50.0,
        label="Macro shocks per year",
        step=0.25,
        help="Exchange collapses, hacks, rate hikes. A sudden hit to everyone's conviction.",
    )
    shock_size: float = knob(1.0, 0.0, 10.0, label="Macro shock size", step=0.1, advanced=True)


@dataclass(frozen=True)
class SimConfig:
    run: RunConfig = field(default_factory=RunConfig)
    supply: SupplyConfig = field(default_factory=SupplyConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    crowd: CrowdConfig = field(default_factory=CrowdConfig)
    traders: TraderConfig = field(default_factory=TraderConfig)
    whales: WhaleConfig = field(default_factory=WhaleConfig)
    attacker: AttackerConfig = field(default_factory=AttackerConfig)
    treasury: TreasuryConfig = field(default_factory=TreasuryConfig)
    stable: StableConfig = field(default_factory=StableConfig)
    market: MarketConfig = field(default_factory=MarketConfig)


SECTION_TITLES: Dict[str, str] = {
    "run": "Run",
    "supply": "Supply and token design",
    "liquidity": "Liquidity pool",
    "crowd": "Holders and adoption",
    "traders": "Momentum and noise traders",
    "whales": "Whales",
    "attacker": "Attacker",
    "treasury": "Treasury policy",
    "stable": "Stablecoin mechanics",
    "market": "Regimes, hype and shocks",
}


# ---------------------------------------------------------------------------
# Introspection helpers (used by the UI, presets and tests)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnobSpec:
    section: str
    name: str
    default: Any
    kind: str  # "float", "int", "bool", "choice"
    label: str
    help: str
    lo: float | None
    hi: float | None
    step: float | None
    advanced: bool
    fmt: str | None
    choices: Tuple[str, ...] | None
    log: bool

    @property
    def path(self) -> str:
        return f"{self.section}.{self.name}"


def _field_default(f) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    raise ValueError(f"field {f.name} has no default")


def iter_knobs() -> Iterator[KnobSpec]:
    """Yield every scalar knob in declaration order."""
    for section_field in fields(SimConfig):
        section_cls = section_field.default_factory  # type: ignore[assignment]
        for f in fields(section_cls):
            md = f.metadata
            if "label" not in md or md.get("choices") is None and "lo" not in md:
                continue
            default = _field_default(f)
            if md.get("choices"):
                kind = "choice"
            elif isinstance(default, bool):
                kind = "bool"
            elif isinstance(default, int):
                kind = "int"
            else:
                kind = "float"
            yield KnobSpec(
                section=section_field.name,
                name=f.name,
                default=default,
                kind=kind,
                label=md["label"],
                help=md.get("help", ""),
                lo=md.get("lo"),
                hi=md.get("hi"),
                step=md.get("step"),
                advanced=bool(md.get("advanced")),
                fmt=md.get("fmt"),
                choices=md.get("choices"),
                log=bool(md.get("log")),
            )


KNOBS: Tuple[KnobSpec, ...] = tuple(iter_knobs())
KNOB_BY_PATH: Dict[str, KnobSpec] = {k.path: k for k in KNOBS}


def coerce_value(spec: KnobSpec, value: Any) -> Any:
    """Cast and clamp a raw value to what the knob accepts."""
    if spec.kind == "bool":
        return bool(value)
    if spec.kind == "choice":
        value = str(value)
        if spec.choices and value not in spec.choices:
            raise ValueError(f"{spec.path}: {value!r} is not one of {spec.choices}")
        return value
    num = float(value)
    if num != num:  # NaN
        raise ValueError(f"{spec.path}: NaN is not allowed")
    if spec.lo is not None:
        num = max(num, float(spec.lo))
    if spec.hi is not None:
        num = min(num, float(spec.hi))
    return int(round(num)) if spec.kind == "int" else num


def to_flat(cfg: SimConfig) -> Dict[str, Any]:
    """Flatten a config to {"section.knob": value} for scalar knobs."""
    out: Dict[str, Any] = {}
    for spec in KNOBS:
        out[spec.path] = getattr(getattr(cfg, spec.section), spec.name)
    return out


def from_flat(flat: Dict[str, Any], base: SimConfig | None = None) -> SimConfig:
    """Build a config from {"section.knob": value}; unknown keys raise, missing keys keep ``base``."""
    base = base or SimConfig()
    per_section: Dict[str, Dict[str, Any]] = {}
    for path, value in flat.items():
        spec = KNOB_BY_PATH.get(path)
        if spec is None:
            raise KeyError(f"unknown knob {path!r}")
        per_section.setdefault(spec.section, {})[spec.name] = coerce_value(spec, value)
    updates = {name: replace(getattr(base, name), **vals) for name, vals in per_section.items()}
    return replace(base, **updates)


def override(cfg: SimConfig, **paths: Any) -> SimConfig:
    """``override(cfg, run__days=90)`` style helper; double underscore separates section and knob."""
    return from_flat({k.replace("__", "."): v for k, v in paths.items()}, base=cfg)


def to_dict(cfg: SimConfig) -> Dict[str, Any]:
    """JSON-friendly nested dict (manual orders included)."""
    return asdict(cfg)


def from_dict(data: Dict[str, Any]) -> SimConfig:
    """Inverse of :func:`to_dict`. Unknown keys are ignored so old exports keep loading."""
    if not isinstance(data, dict):
        raise ValueError("knob file must contain a JSON object")
    flat: Dict[str, Any] = {}
    orders: Tuple[ManualOrder, ...] = ()
    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            path = f"{section}.{name}"
            if path == "attacker.manual_orders":
                if not isinstance(value, list):
                    raise ValueError("manual orders must be a list")
                parsed = []
                for raw in value:
                    try:
                        if not isinstance(raw, dict):
                            raise ValueError("expected an object")
                        order = ManualOrder(**{k: v for k, v in raw.items() if k in ManualOrder.__dataclass_fields__})
                        day, days, usd = float(order.day), float(order.days), float(order.usd)
                        if not all(math.isfinite(v) for v in (day, days, usd)):
                            raise ValueError("values must be finite")
                        if day < 0 or not day.is_integer() or days < 1 or not days.is_integer() or usd < 0:
                            raise ValueError("day, duration or amount is out of range")
                        if order.side not in ("buy", "sell"):
                            raise ValueError("side must be buy or sell")
                        parsed.append(ManualOrder(day=int(day), days=int(days), side=order.side, usd=usd))
                    except (ValueError, TypeError, OverflowError) as exc:
                        raise ValueError(f"invalid manual order: {exc}") from exc
                orders = tuple(parsed)
            elif path in KNOB_BY_PATH:
                flat[path] = value
    cfg = from_flat(flat)
    if orders:
        cfg = replace(cfg, attacker=replace(cfg.attacker, manual_orders=orders))
    return cfg


def validate(cfg: SimConfig) -> List[str]:
    """Return human-readable warnings about inconsistent settings (the engine still runs)."""
    warnings: List[str] = []
    s = cfg.supply
    if cfg.run.mode == "token":
        allocated = s.holders_share + s.whales_share + s.insiders_share + cfg.attacker.token_share
        pool_tokens = cfg.liquidity.pool_usd / max(s.launch_price, 1e-12)
        if pool_tokens > 0.9 * s.total_supply:
            warnings.append(
                "The pool needs more tokens than 90% of supply at this launch price; pool depth was capped."
            )
        if allocated > 1.0:
            warnings.append(
                f"Holder, whale, insider and attacker shares add up to {allocated:.0%}; they were scaled down to fit."
            )
    if cfg.traders.momentum_fast_days >= cfg.traders.momentum_slow_days:
        warnings.append("Momentum fast average is not faster than the slow one, so momentum traders stay idle.")
    if cfg.treasury.ceiling_x_launch and cfg.treasury.ceiling_x_launch <= cfg.treasury.floor_x_launch:
        warnings.append("Treasury ceiling is at or below the floor; the ceiling is ignored.")
    return warnings


def is_config(obj: Any) -> bool:
    return is_dataclass(obj) and isinstance(obj, SimConfig)
