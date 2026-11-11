"""The simulation engine: a daily agent-based market around a constant-product pool.

Design rules this file sticks to:

* Price is never set directly. It is whatever the pool reserves say after the day's trades.
* Every trade goes through :class:`~moonsim.amm.Pool`, is capped by the trader's actual
  balances, and moves tokens and dollars between named wallets. Nothing appears from
  nowhere: new tokens only come from explicit mints (emissions, stablecoin mints,
  governance-token mints on redemption) and dollars only enter or leave through
  newcomers, churners and explicit external promotion spend. :meth:`Simulation.check_conservation` proves it every step.
* All randomness comes from one seeded ``numpy.random.Generator``, so a seed fully
  determines a run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .amm import Pool
from .config import DAYS_PER_YEAR, SimConfig, validate

FLOW_GROUPS: Tuple[str, ...] = (
    "holders",
    "momentum",
    "noise",
    "whales",
    "insiders",
    "attacker",
    "treasury",
    "arbitrageurs",
    "lps",
)

REGIMES = ("bear", "neutral", "bull")
DUST = 1e-9
CONSERVATION_TOL = 1e-9
# Economic terminal state, well before float underflow/overflow. These are triggers,
# never price/reserve floors: the last traded governance reserves remain untouched.
GOV_MIN_PRICE = 1e-12  # USD per governance token
GOV_MIN_RESERVE = 1e-9  # either whole governance tokens or USD
GOV_SUPPLY_CAP = 1e24  # reject a redemption that would issue beyond this supply


class ConservationError(AssertionError):
    pass


@dataclass
class _RunningSum:
    """Neumaier sum: retain the low bits lost when cumulative turnover grows.

    Conservation uses both components in one fsum, never the rounded total of
    two large counters subtracted from each other. Storage is constant size.
    """

    high: float = 0.0
    low: float = 0.0

    def add(self, amount: float) -> None:
        total = self.high + amount
        if abs(self.high) >= abs(amount):
            self.low += (self.high - total) + amount
        else:
            self.low += (amount - total) + self.high
        self.high = total

    @property
    def value(self) -> float:
        return math.fsum((self.high, self.low))


class Wallet:
    """Cash retains low bits while a large cohort wallet drains through churn."""

    def __init__(self, name: str, token: float = 0.0, usd: float = 0.0, gov: float = 0.0, lp: float = 0.0):
        self.name, self.token, self.gov, self.lp = name, token, gov, lp
        self._usd = _RunningSum(usd)

    @property
    def usd(self) -> float:
        value = self._usd.value
        # A spendable float must not exceed the represented cash balance. Keep
        # sub-ULP change in the wallet instead of overdrawing it on an all-in buy.
        if math.isfinite(value) and value > 0 and math.fsum((self._usd.high, self._usd.low, -value)) < 0:
            return math.nextafter(value, 0.0)
        return value

    @usd.setter
    def usd(self, value: float) -> None:
        self._usd = _RunningSum(value)

    def add_usd(self, amount: float) -> None:
        self._usd.add(amount)


@dataclass(frozen=True)
class Event:
    day: int
    kind: str
    title: str
    detail: str = ""
    severity: int = 1  # 1 minor, 2 notable, 3 major


@dataclass
class SimResult:
    config: SimConfig
    series: Dict[str, np.ndarray]
    events: List[Event]
    warnings: List[str]
    max_conservation_error: float

    @property
    def days(self) -> np.ndarray:
        return self.series["day"]

    @property
    def price(self) -> np.ndarray:
        return self.series["price"]

    @property
    def protocol_failure_day(self) -> int | None:
        days = np.flatnonzero(self.series["protocol_failed"])
        return int(days[0]) if days.size else None

    def frame(self):
        import pandas as pd  # local import: the engine itself only needs numpy

        return pd.DataFrame(self.series)

    def summary(self) -> Dict[str, float | int | None]:
        s = self.series
        price = s["price"]
        start = float(price[0])
        peak_idx = int(np.argmax(price))
        trough_after_peak = float(np.min(price[peak_idx:]))
        running_max = np.maximum.accumulate(price)
        max_dd = float(np.max(1.0 - price / running_max))
        return {
            "start_price": start,
            "end_price": float(price[-1]),
            "peak_price": float(price[peak_idx]),
            "peak_day": int(s["day"][peak_idx]),
            "min_price": float(np.min(price)),
            "min_day": int(s["day"][int(np.argmin(price))]),
            "trough_after_peak": trough_after_peak,
            "max_drawdown": max_dd,
            "end_users": float(s["users"][-1]),
            "end_pool_usd": float(s["pool_usd"][-1]),
            "start_pool_usd": float(s["pool_usd"][0]),
            "end_market_cap": float(s["market_cap"][-1]),
            "end_exit_value": float(s["exit_value"][-1]),
            "attacker_pnl": float(s["attacker_pnl"][-1]),
            "volume_usd": float(np.sum(s["volume_usd"])),
            "protocol_failure_day": self.protocol_failure_day,
            "external_spend_usd": float(s["external_spend_usd"][-1]),
        }


SERIES_NAMES: Tuple[str, ...] = (
    "day",
    "price",
    "peg",
    "pool_usd",
    "pool_token",
    "market_cap",
    "fdv",
    "exit_value",
    "circulating",
    "total_supply",
    "burned",
    "minted",
    "locked",
    "users",
    "conviction",
    "fear",
    "hype",
    "regime",
    "treasury_usd",
    "treasury_token",
    "reserve_usd",
    "collateral_ratio",
    "backing_ratio",
    "gov_price",
    "gov_supply",
    "protocol_failed",
    "redeemed",
    "volume_usd",
    "attacker_value",
    "attacker_pnl",
    "external_spend_usd",
    "attacker_token",
    "whale_token",
    "whale_usd",
    "holder_token",
    "holder_usd",
    "lp_external_share",
    "paused",
    "drawdown",
    "vol7",
    "token_error",
    "usd_error",
) + tuple(f"flow_{g}" for g in FLOW_GROUPS)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _fmt_usd(x: float) -> str:
    ax = abs(x)
    if ax >= 1e9:
        return f"${x / 1e9:,.2f}B"
    if ax >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if ax >= 1e3:
        return f"${x / 1e3:,.1f}k"
    if ax >= 1:
        return f"${x:,.2f}"
    if ax == 0:
        return "$0"
    if ax >= 0.001:
        return f"${x:.3g}"
    return f"${x:.2e}"


def _fmt_num(x: float) -> str:
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:,.2f}B"
    if ax >= 1e6:
        return f"{x / 1e6:,.2f}M"
    if ax >= 1e3:
        return f"{x / 1e3:,.1f}k"
    return f"{x:,.0f}"


class Simulation:
    """One run. Build it, call :meth:`run`, read the :class:`SimResult`."""

    def __init__(self, cfg: SimConfig, strict: bool = False) -> None:
        self.cfg = cfg
        self.strict = strict
        self.warnings: List[str] = validate(cfg)
        self.rng = np.random.default_rng(int(cfg.run.seed))
        self.days = int(cfg.run.days)
        self.stable = cfg.run.mode == "stablecoin"
        self.events: List[Event] = []
        self._setup_world()
        self._setup_series()

    # ------------------------------------------------------------------ setup

    def _setup_world(self) -> None:
        c = self.cfg
        s, liq = c.supply, c.liquidity

        self.peg = 1.0
        self.gold = 1.0
        launch_price = self.peg if self.stable else float(s.launch_price)
        self.launch_price = launch_price

        total = float(s.total_supply)
        pool_usd = float(liq.pool_usd)
        pool_tokens = pool_usd / launch_price
        if pool_tokens > 0.9 * total:
            pool_tokens = 0.9 * total
            pool_usd = pool_tokens * launch_price
        self.pool = Pool(token=pool_tokens, usd=pool_usd, fee=float(liq.lp_fee))

        shares = [s.holders_share, s.whales_share, s.insiders_share, c.attacker.token_share]
        room = (total - pool_tokens) / total
        if sum(shares) > room:
            scale = room / sum(shares)
            shares = [x * scale for x in shares]
        h_tok, w_tok, i_tok, a_tok = (x * total for x in shares)

        W = Wallet
        self.holders = W("holders")
        self.momentum = W("momentum", usd=float(c.traders.momentum_capital))
        self.noise = W("noise", usd=float(c.traders.noise_capital))
        self.insiders = W("insiders")
        self.attacker = W("attacker", token=a_tok, usd=float(c.attacker.capital_usd))
        self.treasury = W("treasury", usd=float(c.treasury.usd))
        self.arbs = W("arbitrageurs", usd=float(c.stable.arb_capital) if self.stable else 0.0)
        self.lps = W("lps")
        self.reserve = W("reserve")
        self.gov_holders = W("gov_holders")

        # Noise traders start half in tokens, taken from the holders' allocation.
        noise_tok = min(self.noise.usd * 0.5 / launch_price, h_tok * 0.5)
        self.noise.token = noise_tok
        self.noise.add_usd(-(noise_tok * launch_price))
        h_tok -= noise_tok
        self.holders.token = h_tok

        self.conviction = float(c.crowd.conviction_start)
        self.base_logit = _logit(self.conviction)
        w0 = c.crowd.max_alloc * self.conviction
        holder_value = h_tok * launch_price
        if holder_value > 0:
            self.holders.usd = holder_value * (1.0 - w0) / max(w0, 1e-6)
        else:
            self.holders.usd = float(c.crowd.initial_users) * float(c.crowd.ticket_usd)
        self.users = float(c.crowd.initial_users)
        self.newcomer_cash = 0.0  # part of holders.usd that newcomers came to spend

        n_whales = int(c.whales.count)
        if n_whales == 0:
            w_tok = 0.0  # Unassigned whale allocation remains in the treasury.
        self.whales: List[Wallet] = []
        self.whale_basis: List[float] = []
        self.whale_initial: List[float] = []
        for i in range(n_whales):
            bag = w_tok / n_whales
            self.whales.append(W(f"whale_{i + 1}", token=bag))
            self.whale_basis.append(launch_price * float(c.whales.basis_multiple))
            self.whale_initial.append(bag)

        self.locked = i_tok
        self.locked_initial = i_tok
        self.unlocked = 0.0
        self._burned = _RunningSum()
        self._minted = _RunningSum()

        allocated = pool_tokens + h_tok + noise_tok + w_tok + i_tok + a_tok
        self.treasury.token = max(0.0, total - allocated)

        self.treasury.lp = self.pool.shares * float(liq.protocol_owned_share)
        self.lps.lp = self.pool.shares - self.treasury.lp

        # Stablecoin machinery: reserve, governance token and its pool.
        self.gov_pool: Pool | None = None
        self.protocol_failed = False
        self._gov_minted = _RunningSum()
        self._gov_burned = _RunningSum()
        self.gov_total0 = 0.0
        self.redeemed_total = 0.0
        if self.stable:
            st = c.stable
            self.reserve.usd = float(st.reserve_ratio) * total * self.peg
            g_total = float(st.gov_supply)
            g_pool_tokens = min(float(st.gov_pool_usd) / float(st.gov_price), 0.9 * g_total)
            self.gov_pool = Pool(token=g_pool_tokens, usd=g_pool_tokens * float(st.gov_price), fee=float(liq.lp_fee))
            self.gov_holders.gov = g_total - g_pool_tokens
            self.gov_holders.usd = float(st.gov_holder_usd)
            self.gov_total0 = g_total
            self.gov_conviction = 0.6
            g_value = self.gov_holders.gov * self.gov_pool.price
            # Governance holders start in equilibrium: their target allocation at day-0 mood is what they hold.
            self.gov_base_alloc = g_value / max(g_value + self.gov_holders.usd, 1e-9)
            self.gov_high = self.gov_pool.price
            self.gov_launch = self.gov_pool.price

        self.wallets: List[Wallet] = [
            self.holders,
            self.momentum,
            self.noise,
            self.insiders,
            self.attacker,
            self.treasury,
            self.arbs,
            self.lps,
            self.reserve,
            self.gov_holders,
            *self.whales,
        ]
        self.token0 = self._token_total()
        self._initial_usd = self._cash_components()
        self.usd0 = self._usd_total()
        self.gov0 = self._gov_total()
        self._ext_in = _RunningSum()
        self._ext_out = _RunningSum()
        self._external_spend = _RunningSum()

        # Baseline for attacker PnL: what they could have walked away with on day 0.
        self.attacker_start_value = self.attacker.usd + self.pool.quote_sell(self.attacker.token * (1.0 - self.tax))
        self.attacker_phase = "waiting"
        self.attacker_phase_day = 0
        self.attacker_cost = 0.0
        self.attacker_bought = 0.0
        self.raid_price = 0.0

        # Market state.
        self.regime = c.market.start_regime if c.market.regimes else "neutral"
        self.hype = 0.0
        self.shock_fear = 0.0
        self.fear = 0.0
        self.pause_left = 0
        self.prices: List[float] = [self.pool.price]
        self.ema_fast = self.pool.price
        self.ema_slow = self.pool.price
        self.high = self.pool.price
        self.high_decay = math.exp(-1.0 / max(float(c.crowd.memory_days), 1.0))
        self.ath = self.pool.price
        self.vol7 = 0.0
        self.flows: Dict[str, float] = {g: 0.0 for g in FLOW_GROUPS}
        self.volume = 0.0
        self.redeem_cap_left = 0.0
        self.redeemed_today = 0.0

        # Event bookkeeping.
        self._last_event_day: Dict[str, int] = {}
        self._dd_marks_hit: set = set()
        self._depegged = False
        self._reserve_empty = False
        self._gov_marks_hit: set = set()
        self._cr_marks_hit: set = set()
        self._lp_fleeing = False
        self._treasury_drained = False
        self._ath_event_price = self.pool.price
        self._whale_tp_announced = False
        self._whale_panic_announced = False

    def _setup_series(self) -> None:
        n = self.days + 1
        self.series: Dict[str, np.ndarray] = {name: np.zeros(n) for name in SERIES_NAMES}
        self.series["protocol_failed"] = np.zeros(n, dtype=bool)
        self.max_error = 0.0

    # ------------------------------------------------------------ accounting

    @property
    def minted(self) -> float:
        return self._minted.value

    @property
    def burned(self) -> float:
        return self._burned.value

    @property
    def gov_minted(self) -> float:
        return self._gov_minted.value

    @property
    def gov_burned(self) -> float:
        return self._gov_burned.value

    @property
    def ext_in(self) -> float:
        return self._ext_in.value

    @property
    def ext_out(self) -> float:
        return self._ext_out.value

    @property
    def external_spend(self) -> float:
        return self._external_spend.value

    def _token_total(self) -> float:
        return math.fsum([w.token for w in self.wallets] + [self.pool.token, self.locked, self.burned])

    def _usd_total(self) -> float:
        return math.fsum(self._cash_components())

    def _cash_components(self) -> List[float]:
        return [v for w in self.wallets for v in (w._usd.high, w._usd.low)] + [
            self.pool.usd,
            self.gov_pool.usd if self.gov_pool else 0.0,
        ]

    def _gov_total(self) -> float:
        if self.gov_pool is None:
            return 0.0
        return math.fsum([w.gov for w in self.wallets] + [self.gov_pool.token, self.gov_burned])

    def check_conservation(self) -> Tuple[float, float, float]:
        """Relative residuals of all four ledgers, summed before rounding/cancellation."""
        if self.strict:
            balances = [v for w in self.wallets for v in (w.token, w.usd, w.gov, w.lp)]
            balances += [
                self.pool.token,
                self.pool.usd,
                self.pool.shares,
                self.locked,
                self.burned,
                self.minted,
                self.gov_minted,
                self.gov_burned,
                self.ext_in,
                self.ext_out,
                self.external_spend,
            ]
            if self.gov_pool is not None:
                balances += [self.gov_pool.token, self.gov_pool.usd, self.gov_pool.shares]
            if not all(math.isfinite(v) for v in balances):
                raise ConservationError("nonfinite balance in ledger")
        tok_expected = [self.token0, self._minted.high, self._minted.low]
        usd_expected = self._initial_usd + [
            self._ext_in.high,
            self._ext_in.low,
            -self._ext_out.high,
            -self._ext_out.low,
            -self._external_spend.high,
            -self._external_spend.low,
        ]
        gov_expected = [self.gov0, self._gov_minted.high, self._gov_minted.low]
        token = [w.token for w in self.wallets] + [self.pool.token, self.locked, self._burned.high, self._burned.low]
        usd = self._cash_components()
        gov = [w.gov for w in self.wallets] + [
            self.gov_pool.token if self.gov_pool else 0.0,
            self._gov_burned.high,
            self._gov_burned.low,
        ]
        tok_err = abs(math.fsum(token + [-v for v in tok_expected])) / max(math.fsum(tok_expected), 1.0)
        usd_err = abs(math.fsum(usd + [-v for v in usd_expected])) / max(math.fsum(usd_expected), 1.0)
        gov_err = abs(math.fsum(gov + [-v for v in gov_expected])) / max(math.fsum(gov_expected), 1.0)
        lp_err = abs(math.fsum((self.treasury.lp, self.lps.lp, -self.pool.shares))) / max(self.pool.shares, 1.0)
        worst = max(tok_err, usd_err, gov_err, lp_err)
        if self.strict:
            if worst > CONSERVATION_TOL:
                raise ConservationError(
                    f"ledger drift: token {tok_err:.2e}, usd {usd_err:.2e}, gov {gov_err:.2e}, lp {lp_err:.2e}"
                )
            for w in self.wallets:
                if min(w.token, w.usd, w.gov, w.lp) < -1e-9:
                    raise ConservationError(f"negative balance in {w.name}: {w}")
            if self.pool.token <= 0 or self.pool.usd <= 0:
                raise ConservationError("pool reserve went non-positive")
        return tok_err, usd_err, max(gov_err, lp_err)

    # ---------------------------------------------------------------- trading

    @property
    def tax(self) -> float:
        return float(self.cfg.supply.transfer_tax)

    def _take_tax(self, amount: float) -> None:
        burn = amount * float(self.cfg.supply.tax_burn_share)
        self._burned.add(burn)
        self.treasury.token += amount - burn

    def buy(self, w: Wallet, usd: float, group: str, taxed: bool = True) -> float:
        usd = min(usd, w.usd)
        if usd <= DUST:
            return 0.0
        out = self.pool.buy(usd)
        w.add_usd(-usd)
        tax = out * self.tax if taxed else 0.0
        w.token += out - tax
        if tax > 0:
            self._take_tax(tax)
        self.flows[group] += usd
        self.volume += usd
        return out - tax

    def sell(self, w: Wallet, tokens: float, group: str, taxed: bool = True) -> float:
        tokens = min(tokens, w.token)
        if tokens <= DUST:
            return 0.0
        tax = tokens * self.tax if taxed else 0.0
        w.token -= tokens
        if tax > 0:
            self._take_tax(tax)
        usd = self.pool.sell(tokens - tax)
        w.add_usd(usd)
        self.flows[group] -= usd
        self.volume += usd
        return usd

    def buy_capped(self, w: Wallet, usd: float, group: str, max_slip: float, taxed: bool = True) -> float:
        """Buy, but stop once the price has moved ``max_slip`` (traders set slippage limits)."""
        if max_slip < 1.0:
            usd = min(usd, self.pool.usd_to_reach_price(self.pool.price * (1.0 + max_slip)))
        return self.buy(w, usd, group, taxed)

    def sell_capped(self, w: Wallet, tokens: float, group: str, max_slip: float, taxed: bool = True) -> float:
        if max_slip < 1.0:
            limit = self.pool.tokens_to_reach_price(self.pool.price * (1.0 - max_slip))
            if taxed and self.tax < 1.0:
                limit /= 1.0 - self.tax
            tokens = min(tokens, limit)
        return self.sell(w, tokens, group, taxed)

    def rebalance(self, w: Wallet, target_alloc: float, speed: float, group: str, max_slip: float = 1.0) -> None:
        """Move a fraction ``speed`` of the way toward holding ``target_alloc`` of wealth in tokens."""
        p = self.pool.price
        value = w.token * p
        wealth = value + w.usd
        if wealth <= DUST:
            return
        gap = (target_alloc * wealth - value) * min(max(speed, 0.0), 1.0)
        if gap > 0:
            self.buy_capped(w, gap, group, max_slip)
        elif gap < 0:
            self.sell_capped(w, -gap / p, group, max_slip)

    # --------------------------------------------------------------- events

    def _event(self, day: int, kind: str, title: str, detail: str = "", severity: int = 1, cooldown: int = 0) -> None:
        last = self._last_event_day.get(kind)
        if cooldown and last is not None and day - last < cooldown:
            return
        self._last_event_day[kind] = day
        self.events.append(Event(day, kind, title, detail, severity))

    # ------------------------------------------------------------ world step

    def _update_regime(self, day: int) -> None:
        m = self.cfg.market
        if not m.regimes:
            return
        mean = {"bull": m.bull_days, "neutral": m.neutral_days, "bear": m.bear_days}[self.regime]
        if self.rng.random() < 1.0 - math.exp(-1.0 / max(mean, 1.0)):
            weights = {"bull": m.bull_bias, "bear": 1.0 - m.bull_bias, "neutral": 0.5}
            options = [r for r in REGIMES if r != self.regime]
            ws = np.array([max(weights[r], 1e-9) for r in options])
            new = options[int(self.rng.choice(len(options), p=ws / ws.sum()))]
            self.regime = new
            label = {"bull": "Bull market begins", "bear": "Bear market begins", "neutral": "Market goes sideways"}[new]
            self._event(day, "regime", label, "Macro mood shifted: adoption, newcomer budgets and LP nerves follow.", 2)

    def _update_hype_and_shocks(self, day: int) -> None:
        m = self.cfg.market
        self.hype *= 0.5 ** (1.0 / max(m.hype_half_life, 0.5))
        self.shock_fear *= 0.5 ** (1.0 / 7.0)
        if m.hype_per_year > 0 and self.rng.random() < 1.0 - math.exp(-m.hype_per_year / DAYS_PER_YEAR):
            size = m.hype_size * float(self.rng.lognormal(0.0, 0.5))
            viral = self.rng.random() < m.viral_chance
            if viral:
                size *= m.viral_mult
            self.hype += size
            if viral:
                self._event(day, "hype", "Goes viral", f"A hype wave of size {size:.1f} hits the timeline.", 2)
            elif size >= 1.5 * max(m.hype_size, 1e-9):
                self._event(day, "hype", "Hype wave", f"News and influencer chatter, size {size:.1f}.", 1, cooldown=10)
        if m.shock_per_year > 0 and self.rng.random() < 1.0 - math.exp(-m.shock_per_year / DAYS_PER_YEAR):
            hit = m.shock_size * float(self.rng.uniform(0.7, 1.3))
            self.shock_fear += hit
            self._event(day, "shock", "Macro shock", "Something blew up elsewhere. Everyone gets more nervous.", 2)

    def _update_peg(self) -> None:
        st = self.cfg.stable
        if st.peg_target == "gold":
            dt = 1.0 / DAYS_PER_YEAR
            z = float(self.rng.standard_normal())
            self.gold *= math.exp((st.gold_drift - 0.5 * st.gold_vol**2) * dt + st.gold_vol * math.sqrt(dt) * z)
            self.peg = self.gold
        else:
            self.peg = 1.0

    def _vest(self, day: int) -> None:
        s = self.cfg.supply
        if self.locked <= DUST or day < s.vesting_cliff_days:
            return
        target = self.locked_initial * min(1.0, day / max(s.vesting_days, 1))
        amount = min(max(target - self.unlocked, 0.0), self.locked)
        if amount <= DUST:
            return
        self.locked -= amount
        self.unlocked += amount
        self.insiders.token += amount
        if s.vesting_cliff_days > 0 and day == s.vesting_cliff_days:
            self._event(day, "unlock", "Vesting cliff", f"{_fmt_num(amount)} insider tokens unlock at once.", 3)

    def _mint_tokens(self, w: Wallet, amount: float) -> float:
        cap = float(self.cfg.supply.max_supply)
        if cap > 0 and not self.stable:
            amount = min(amount, max(0.0, cap - (self.token0 + self.minted)))
        if amount <= 0:
            return 0.0
        w.token += amount
        self._minted.add(amount)
        return amount

    def _emit(self, day: int) -> None:
        s = self.cfg.supply
        if s.emission_apy <= 0 or self.holders.token <= DUST:
            return
        rate = s.emission_apy / DAYS_PER_YEAR
        if s.emission_halving_days > 0:
            rate *= 0.5 ** (day // s.emission_halving_days)
        self._mint_tokens(self.holders, self.holders.token * rate)

    # ----------------------------------------------------------- the crowd

    def _signals(self) -> Tuple[float, float]:
        """Return (mood, fear) for retail holders. Mood is a shift in logit space."""
        c, m = self.cfg.crowd, self.cfg.market
        p = self.pool.price
        regime_mood = {"bull": m.bull_mood, "bear": m.bear_mood, "neutral": 0.0}[self.regime]
        if self.stable:
            depeg = max(0.0, 1.0 - p / self.peg)
            fear = c.depeg_sensitivity * max(0.0, depeg - c.depeg_tolerance)
            if depeg > c.depeg_tolerance:
                fear += c.backing_sensitivity * max(0.0, 1.0 - self._backing_ratio())
            fear += c.panic_intensity * _sigmoid((depeg - c.depeg_panic) * 200.0)
            fear += self.shock_fear
            mood = regime_mood * 0.25 + c.yield_sensitivity * self.cfg.supply.emission_apy - fear
        else:
            dd = 1.0 - p / self.high if self.high > 0 else 0.0
            look = self.prices[-8] if len(self.prices) >= 8 else self.prices[0]
            r7 = math.log(max(p, 1e-300) / max(look, 1e-300))
            fear = c.fear_sensitivity * max(0.0, dd - c.fear_threshold)
            fear += c.panic_intensity * _sigmoid((dd - c.panic_threshold) * 20.0)
            fear += self.shock_fear
            dip = c.dip_buying * min(max(dd, 0.0), c.fear_threshold)
            mood = regime_mood + m.hype_mood * self.hype + c.return_sensitivity * max(-1.0, min(1.0, r7)) + dip - fear
        return mood, fear

    def _users_and_holders_prep(self, day: int, paused: bool) -> None:
        c, m = self.cfg.crowd, self.cfg.market
        mood, fear = self._signals()
        self.fear = fear
        target = c.diamond_hands + (1.0 - c.diamond_hands) * _sigmoid(self.base_logit + mood)
        speed = c.greed_speed if target > self.conviction else c.fear_speed
        self.conviction += speed * (target - self.conviction)

        # Adoption (Bass diffusion) and newcomer cash.
        U, M = self.users, float(c.market_potential)
        regime_adopt = {"bull": m.bull_adoption, "bear": m.bear_adoption, "neutral": 1.0}[self.regime]
        regime_ticket = {"bull": m.bull_ticket, "bear": m.bear_ticket, "neutral": 1.0}[self.regime]
        if self.stable:
            pull = math.exp(-min(fear, 20.0))
        else:
            look = self.prices[-31] if len(self.prices) >= 31 else self.prices[0]
            r30 = math.log(max(self.pool.price, 1e-300) / max(look, 1e-300))
            pull = math.exp(c.fomo * max(-1.0, min(1.0, r30)) - 0.5 * min(fear, 20.0))
        rate = (c.bass_p + c.bass_q * U / max(M, 1.0)) * max(M - U, 0.0) / DAYS_PER_YEAR
        rate *= regime_adopt * (1.0 + m.hype_adoption * self.hype) * pull
        rate = max(rate, 0.0)
        new = float(self.rng.poisson(rate)) if rate < 1e7 else rate
        new = min(new, max(M - U, 0.0))
        if new > 0:
            sigma = c.ticket_noise / math.sqrt(max(new, 1.0))
            ticket = c.ticket_usd * regime_ticket * math.exp(float(self.rng.normal(-0.5 * sigma**2, sigma)))
            cash = new * ticket
            self.holders.add_usd(cash)
            self._ext_in.add(cash)
            self.users += new
            self.newcomer_cash += cash

        # Churn: leavers sell their pro-rata bag and take their cash out of the system.
        if paused or self.users <= 1.0:
            return
        churn = c.churn_per_year / DAYS_PER_YEAR
        if self.regime == "bear":
            churn *= m.bear_churn
        churn *= 1.0 + c.churn_fear_mult * fear
        frac = min(1.0 - math.exp(-churn), 0.5)
        if frac <= 0:
            return
        self.sell_capped(self.holders, self.holders.token * frac, "holders", self.cfg.crowd.panic_slippage)
        cash_out = self.holders.usd * frac
        self.holders.add_usd(-cash_out)
        self._ext_out.add(cash_out)
        self.users -= self.users * frac

    def _act_holders(self, day: int) -> None:
        c = self.cfg.crowd
        # Newcomers came to buy: they work their cash in over a few days, then join the crowd's rebalancing.
        self.newcomer_cash = min(self.newcomer_cash, self.holders.usd)
        if self.newcomer_cash > DUST:
            want = self.newcomer_cash * c.newcomer_speed * self.conviction
            before = self.holders.usd
            self.buy_capped(self.holders, want, "holders", min(c.slippage * 2.0 * (1.0 + self.hype), c.panic_slippage))
            spent = before - self.holders.usd
            self.newcomer_cash = max(0.0, self.newcomer_cash - max(spent, want * 0.5))
        speed = min(1.0, c.rebalance_speed * (1.0 + self.fear))
        slip = min(c.slippage * (1.0 + self.fear), c.panic_slippage) if self.fear > 0 else c.slippage
        self.rebalance(self.holders, c.max_alloc * self.conviction, speed, "holders", slip)

    def _act_momentum(self, day: int) -> None:
        t = self.cfg.traders
        if self.momentum.usd + self.momentum.token <= DUST:
            return
        trend = self.ema_fast / max(self.ema_slow, 1e-300) - 1.0
        signal = math.tanh(t.momentum_sensitivity * trend * (1.0 + 0.5 * self.hype))
        target = t.momentum_max_alloc * max(0.0, signal)
        slip = min(t.momentum_slippage * (1.0 + 0.5 * self.hype), 0.5)
        self.rebalance(self.momentum, target, t.momentum_speed, "momentum", slip)

    def _act_noise(self, day: int) -> None:
        t = self.cfg.traders
        w = self.noise
        p = self.pool.price
        wealth = w.usd + w.token * p
        if wealth <= DUST:
            return
        gap = 0.5 * wealth - w.token * p
        order = 0.1 * gap + float(self.rng.standard_normal()) * t.noise_intensity * wealth * (1.0 + 0.5 * self.hype)
        if order > 0:
            self.buy_capped(w, order, "noise", t.noise_slippage)
        elif order < 0:
            self.sell_capped(w, -order / p, "noise", t.noise_slippage)

    def _act_whales(self, day: int) -> None:
        wc = self.cfg.whales
        p = self.pool.price
        dd = 1.0 - p / self.high if self.high > 0 else 0.0
        if self.stable:
            # Big depositors run when the crowd is running, not on every intraday wobble.
            stressed = max(0.0, 1.0 - p / self.peg) > self.cfg.crowd.depeg_panic and self.fear > 1.0
        else:
            stressed = dd > wc.panic_drawdown
        for i, w in enumerate(self.whales):
            if w.token <= DUST:
                continue
            if i == 0 and wc.exit_day >= 0 and wc.exit_day <= day < wc.exit_day + wc.exit_days:
                if day == wc.exit_day:
                    bag = self.whale_initial[0] * wc.exit_fraction
                    self._event(
                        day,
                        "whale_exit",
                        "Whale starts exiting",
                        f"Selling {_fmt_num(bag)} tokens (~{_fmt_usd(bag * p)} at today's price) into a pool "
                        f"with {_fmt_usd(self.pool.usd)} of dollars.",
                        3,
                    )
                self.sell(w, self.whale_initial[0] * wc.exit_fraction / wc.exit_days, "whales")
                continue
            if stressed and wc.panic_rate > 0:
                if not self._whale_panic_announced:
                    self._whale_panic_announced = True
                    self._event(day, "whale_panic", "Whales panic", "Big holders start dumping into the drawdown.", 2)
                self.sell_capped(w, w.token * wc.panic_rate, "whales", self.cfg.crowd.panic_slippage)
            elif p > self.whale_basis[i] * wc.take_profit_multiple and wc.take_profit_rate > 0:
                if not self._whale_tp_announced:
                    self._whale_tp_announced = True
                    self._event(
                        day,
                        "whale_tp",
                        "Whales take profit",
                        f"Price is {p / self.whale_basis[i]:.0f}x the whales' entry, so they start selling.",
                        2,
                    )
                self.sell_capped(w, w.token * wc.take_profit_rate, "whales", wc.slippage)

    def _act_insiders(self, day: int) -> None:
        rate = self.cfg.supply.insider_sell_rate
        if rate > 0 and self.insiders.token > DUST:
            self.sell_capped(self.insiders, self.insiders.token * rate, "insiders", self.cfg.whales.slippage)

    # ------------------------------------------------------------ attacker

    def _set_phase(self, day: int, phase: str, title: str, detail: str, severity: int = 2) -> None:
        self.attacker_phase = phase
        self.attacker_phase_day = day
        self._event(day, f"attack_{phase}", title, detail, severity)

    def _act_attacker(self, day: int) -> None:
        a = self.cfg.attacker
        w = self.attacker
        p = self.pool.price
        if a.style == "manual":
            for order in a.manual_orders:
                if order.day <= day < order.day + max(order.days, 1):
                    if day == order.day:
                        self._event(
                            day,
                            "manual",
                            f"Your scripted {order.side}",
                            f"{_fmt_usd(order.usd)} over {max(order.days, 1)} day(s).",
                            2,
                        )
                    per_day = order.usd / max(order.days, 1)
                    if order.side == "buy":
                        self._attacker_buy(per_day)
                    else:
                        self.sell(w, per_day / p, "attacker")
            return
        if a.style == "off" or day < a.start_day:
            return

        if a.style == "pump_and_dump":
            if self.attacker_phase == "waiting":
                if a.accumulate_days > 0:
                    self._set_phase(
                        day,
                        "accumulate",
                        "Attacker accumulates quietly",
                        f"Spreading {_fmt_usd(a.capital_usd * a.accumulate_share)} of buys over "
                        f"{a.accumulate_days} days.",
                        1,
                    )
                else:
                    self._start_pump(day)
            if self.attacker_phase == "accumulate":
                if day - self.attacker_phase_day < a.accumulate_days:
                    self._attacker_buy(a.capital_usd * a.accumulate_share / max(a.accumulate_days, 1))
                    return
                self._start_pump(day)
            if self.attacker_phase == "pump":
                elapsed = day - self.attacker_phase_day
                cost = self.attacker_cost / self.attacker_bought if self.attacker_bought > DUST else p
                if (self.attacker_bought > DUST and p >= cost * a.dump_multiple) or elapsed >= a.pump_days:
                    self._set_phase(
                        day,
                        "dump",
                        "Attacker dumps",
                        f"Price is {p / max(cost, 1e-300):.1f}x their average cost. Selling "
                        f"{_fmt_num(w.token)} tokens into the crowd.",
                        3,
                    )
                else:
                    self._attacker_buy(w.usd / max(a.pump_days - elapsed, 1))
                    return
            if self.attacker_phase == "dump":
                left = max(a.dump_days - (day - self.attacker_phase_day), 1)
                self.sell_capped(w, w.token / left, "attacker", a.dump_slippage)
                if w.token <= DUST:
                    self.attacker_phase = "done"
            return

        if a.style == "bear_raid":
            if self.attacker_phase == "waiting":
                self.raid_price = p
                self._set_phase(
                    day,
                    "raid",
                    "Bear raid",
                    f"Attacker dumps {_fmt_num(w.token)} tokens as fast as the "
                    f"pool will take them, hoping to trigger panic.",
                    3,
                )
            if self.attacker_phase == "raid":
                left = max(a.raid_days - (day - self.attacker_phase_day), 1)
                self.sell(w, w.token / left, "attacker")
                if w.token <= DUST or left <= 1:
                    self.attacker_phase = "wait_rebuy"
                return
            if self.attacker_phase == "wait_rebuy" and p <= self.raid_price * (1.0 - a.raid_rebuy_drawdown):
                self._set_phase(
                    day,
                    "rebuy",
                    "Raider buys back",
                    f"Price is down {1 - p / self.raid_price:.0%} from the raid start. Buying the bag back cheap.",
                    2,
                )
            if self.attacker_phase == "rebuy":
                left = max(a.raid_days - (day - self.attacker_phase_day), 1)
                self._attacker_buy(w.usd / left)
                if left <= 1:
                    self.attacker_phase = "done"

    def _start_pump(self, day: int) -> None:
        a = self.cfg.attacker
        self.attacker_phase = "pump"
        self.attacker_phase_day = day
        spend = min(a.promo_hype * a.promo_usd_per_hype, self.attacker.usd)
        hype = spend / a.promo_usd_per_hype
        self.attacker.add_usd(-spend)
        self._external_spend.add(spend)
        self.attacker_cost += spend
        self.hype += hype
        self._event(
            day,
            "attack_pump",
            "Pump starts",
            f"Attacker spends {_fmt_usd(spend)} on promotion (+{hype:.1f} hype) and buys into the pump.",
            3,
        )

    def _attacker_buy(self, usd: float) -> None:
        before = self.attacker.usd
        got = self.buy(self.attacker, usd, "attacker")
        if got > 0:
            self.attacker_cost += before - self.attacker.usd
            self.attacker_bought += got

    # ------------------------------------------------------------ treasury

    def _act_treasury(self, day: int) -> None:
        tc = self.cfg.treasury
        tr = self.treasury
        if self.stable:
            st = self.cfg.stable
            floor = self.peg * (1.0 - st.defense_band)
            if self.pool.price < floor and tr.usd > DUST:
                need = self.pool.usd_to_reach_price(self.peg * (1.0 - 0.5 * st.defense_band))
                spend = min(need, tr.usd * tc.max_daily_spend)
                if spend > DUST:
                    self._event(
                        day,
                        "defense",
                        "Defence fund steps in",
                        f"Treasury spends {_fmt_usd(spend)} buying the coin under {_fmt_usd(floor)}.",
                        1,
                        cooldown=30,
                    )
                    self.buy(tr, spend, "treasury", taxed=False)
            self._check_treasury(day)
            return

        launch = self.launch_price
        budget = tr.usd * tc.max_daily_spend
        if tc.band_defense:
            floor = tc.floor_x_launch * launch
            ceiling = tc.ceiling_x_launch * launch if tc.ceiling_x_launch > tc.floor_x_launch else 0.0
            if floor > 0 and self.pool.price < floor and tr.usd > DUST:
                spend = min(self.pool.usd_to_reach_price(floor), budget)
                if spend > DUST:
                    self._event(
                        day,
                        "floor",
                        "Treasury defends the floor",
                        f"Buying with treasury cash below {_fmt_usd(floor)}.",
                        1,
                        cooldown=20,
                    )
                    self.buy(tr, spend, "treasury", taxed=False)
                    budget -= spend
            elif ceiling > 0 and self.pool.price > ceiling and tr.token > DUST:
                qty = min(self.pool.tokens_to_reach_price(ceiling), tr.token * tc.max_daily_spend)
                if qty > DUST:
                    self._event(
                        day,
                        "ceiling",
                        "Treasury sells into strength",
                        f"Selling treasury tokens above {_fmt_usd(ceiling)}.",
                        1,
                        cooldown=20,
                    )
                    self.sell(tr, qty, "treasury", taxed=False)
        if tc.buyback_per_year > 0 and tr.usd > DUST:
            spend = min(budget, tr.usd * (1.0 - math.exp(-tc.buyback_per_year / DAYS_PER_YEAR)))
            got = self.buy(tr, spend, "treasury", taxed=False)
            if tc.burn_buybacks and got > 0:
                tr.token -= got
                self._burned.add(got)
        self._check_treasury(day)

    def _check_treasury(self, day: int) -> None:
        start = float(self.cfg.treasury.usd)
        if start > 0 and not self._treasury_drained and self.treasury.usd < 0.02 * start:
            self._treasury_drained = True
            self._event(day, "treasury_drained", "Treasury is out of cash", "Nothing left to defend with.", 3)

    # ------------------------------------------------------------ stablecoin

    def _check_governance(self, day: int, issuing: float = 0.0) -> None:
        """Permanently close governance-dependent operations before numbers lose meaning."""
        gp = self.gov_pool
        if gp is None or self.protocol_failed:
            return
        supply = self.gov0 + self.gov_minted - self.gov_burned + issuing
        if gp.price < GOV_MIN_PRICE or min(gp.token, gp.usd) < GOV_MIN_RESERVE or supply > GOV_SUPPLY_CAP:
            self.protocol_failed = True
            self._event(
                day,
                "protocol_failed",
                "Governance token failed",
                "Governance payouts and dependent minting have stopped permanently; redemptions pay only "
                "their pro-rata available collateral and the stablecoin keeps trading.",
                3,
            )

    def _backing_ratio(self) -> float:
        out = self.token0 + self.minted - self.burned
        if out <= DUST:
            return 10.0
        gov_cap = 0.0
        if self.gov_pool is not None and not self.protocol_failed:
            gov_circ = self.gov0 + self.gov_minted - self.gov_burned - self.gov_pool.token
            gov_cap = max(gov_circ, 0.0) * self.gov_pool.price
        return (self.reserve.usd + self.cfg.stable.gov_backing_weight * gov_cap) / (out * self.peg)

    def _collateral_ratio(self) -> float:
        out = self.token0 + self.minted - self.burned
        return self.reserve.usd / (out * self.peg) if out > DUST else 10.0

    def _quote_redeem(self, tokens: float) -> Tuple[float, float, float]:
        """(tokens_redeemable, usd_paid, gov_tokens_minted) for redeeming ``tokens`` right now."""
        st = self.cfg.stable
        tokens = min(tokens, self.redeem_cap_left)
        if tokens <= DUST:
            return 0.0, 0.0, 0.0
        value = tokens * self.peg * (1.0 - st.redeem_fee)
        usd_part = value * st.collateral_ratio
        if self.protocol_failed:
            outstanding = self.token0 + self.minted - self.burned
            # Equal claim per outstanding coin, so early redeemers cannot drain
            # collateral owed to the remaining holders. The shortfall is theirs.
            available = self.reserve.usd * min(tokens / outstanding, 1.0) if outstanding > DUST else 0.0
            return tokens, min(usd_part, available), 0.0
        if usd_part > self.reserve.usd and not st.fallback_to_gov:
            scale = self.reserve.usd / usd_part if usd_part > 0 else 0.0
            tokens *= scale
            value *= scale
            usd_part = self.reserve.usd
        usd_paid = min(usd_part, self.reserve.usd)
        gov_value = value - usd_paid if st.fallback_to_gov else value * (1.0 - st.collateral_ratio)
        assert self.gov_pool is not None
        gov_amt = gov_value / self.gov_pool.price if gov_value > 0 else 0.0
        return tokens, usd_paid, gov_amt

    def _arbitrage(self, day: int) -> None:
        if not self.stable or self.arbs.usd <= DUST:
            return
        self._check_governance(day)
        st = self.cfg.stable
        pool, gp = self.pool, self.gov_pool
        assert gp is not None
        p = pool.price
        lo = self.peg * (1.0 - st.redeem_fee) * (1.0 - st.arb_margin)
        hi = self.peg * (1.0 + st.mint_fee) * (1.0 + st.arb_margin)
        keep = 1.0 - self.tax
        if p < lo and self.redeem_cap_left > DUST:
            full = min(pool.usd_to_reach_price(lo), self.arbs.usd)
            best: Tuple[float, float] | None = None
            for frac in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                usd = full * frac
                tok = pool.quote_buy(usd) * keep
                if tok > self.redeem_cap_left:
                    tok = self.redeem_cap_left
                    usd = pool.usd_in_for_tokens_out(tok / keep)
                r_tok, r_usd, r_gov = self._quote_redeem(tok)
                if r_tok < tok * 0.999:
                    continue  # the reserve could not take it all, so the leftover would be stuck
                profit = r_usd + gp.quote_sell(r_gov) - usd
                if profit > 0 and (best is None or profit > best[0]):
                    best = (profit, usd)
            if best is None:
                return
            _, _, proposed = self._quote_redeem(pool.quote_buy(best[1]) * keep)
            self._check_governance(day, issuing=proposed)
            if self.protocol_failed and proposed > 0:
                self._arbitrage(day)  # re-quote under collateral-only settlement
                return
            got = self.buy(self.arbs, best[1], "arbitrageurs")
            r_tok, r_usd, r_gov = self._quote_redeem(got)
            self.arbs.token -= r_tok
            self._burned.add(r_tok)
            self.redeemed_total += r_tok
            self.redeemed_today += r_tok
            self.redeem_cap_left -= r_tok
            self.reserve.add_usd(-r_usd)
            self.arbs.add_usd(r_usd)
            if r_gov > 0:
                self._gov_minted.add(r_gov)
                self.arbs.gov += r_gov
                self.arbs.add_usd(gp.sell(r_gov))
                self.arbs.gov = 0.0
                self._check_governance(day)
            # Any leftover coins (tax rounding, cap) go back to the pool.
            if self.arbs.token > DUST:
                self.sell(self.arbs, self.arbs.token, "arbitrageurs")
        elif p > hi:
            # Minting (FRAX style): the collateral-ratio share is paid in dollars into the reserve, the rest by
            # buying governance tokens on their pool and burning them. Pure algorithmic coins (CR = 0) are
            # minted entirely by burning governance tokens, which is what pumps them in good times.
            cr = st.collateral_ratio
            if self.protocol_failed and cr < 1.0:
                return
            value_per = self.peg * (1.0 + st.mint_fee)
            full_tok = min(pool.tokens_to_reach_price(hi) / keep, self.arbs.usd / value_per)
            best: Tuple[float, float, float, float] | None = None
            for frac in (1.0, 0.5, 0.25, 0.125, 0.0625):
                tok = full_tok * frac
                usd_part = tok * value_per * cr
                gov_needed = tok * value_per * (1.0 - cr) / gp.price
                gov_cost = gp.usd_in_for_tokens_out(gov_needed) if gov_needed > 0 else 0.0
                cost = usd_part + gov_cost
                if cost > self.arbs.usd:
                    continue
                profit = pool.quote_sell(tok * keep) - cost
                if profit > 0 and (best is None or profit > best[0]):
                    best = (profit, tok, usd_part, gov_cost)
            if best is None:
                return
            _, tok, usd_part, gov_cost = best
            self.arbs.add_usd(-usd_part)
            self.reserve.add_usd(usd_part)
            if gov_cost > 0:
                self.arbs.add_usd(-gov_cost)
                bought = gp.buy(gov_cost)
                self._gov_burned.add(bought)
                self._check_governance(day)
            self._mint_tokens(self.arbs, tok)
            self.sell(self.arbs, tok, "arbitrageurs")

    def _act_gov_holders(self, day: int) -> None:
        gp = self.gov_pool
        self._check_governance(day)
        if gp is None or self.protocol_failed:
            return
        c, m = self.cfg.crowd, self.cfg.market
        g = gp.price
        self.gov_high = max(self.gov_high * self.high_decay, g)
        dd = 1.0 - g / self.gov_high
        regime_mood = {"bull": m.bull_mood, "bear": m.bear_mood, "neutral": 0.0}[self.regime]
        fear = c.fear_sensitivity * max(0.0, dd - c.fear_threshold) + self.shock_fear
        fear += c.panic_intensity * _sigmoid((dd - c.panic_threshold) * 20.0)
        # Governance holders fear dilution: every redemption paid in governance tokens is new supply.
        fear += c.dilution_fear * math.log1p(self.gov_minted / max(self.gov_total0, 1.0))
        dip = c.dip_buying * min(max(dd, 0.0), c.fear_threshold)
        target = _sigmoid(_logit(0.6) + regime_mood + m.hype_mood * self.hype + dip - fear)
        speed = c.greed_speed if target > self.gov_conviction else c.fear_speed
        self.gov_conviction += speed * (target - self.gov_conviction)
        w = self.gov_holders
        value = w.gov * g
        wealth = value + w.usd
        if wealth <= DUST:
            return
        target_alloc = min(1.0, self.gov_base_alloc * self.gov_conviction / 0.6)
        gap = (target_alloc * wealth - value) * min(1.0, c.rebalance_speed * (1.0 + fear))
        slip = min(c.slippage * (1.0 + fear), c.panic_slippage)
        if gap > 0:
            spend = min(gap, w.usd, gp.usd_to_reach_price(g * (1.0 + slip)))
            if spend > DUST:
                w.add_usd(-spend)
                w.gov += gp.buy(spend)
        elif gap < 0:
            qty = min(-gap / g, w.gov, gp.tokens_to_reach_price(g * (1.0 - slip)))
            if qty > DUST:
                w.gov -= qty
                w.add_usd(gp.sell(qty))
        self._check_governance(day)

    # ------------------------------------------------------------- LPs

    def _act_lps(self, day: int) -> None:
        lq, m = self.cfg.liquidity, self.cfg.market
        nerves = m.bear_lp_stress if self.regime == "bear" else 1.0
        p = self.pool.price
        if self.stable:
            dd = max(0.0, 1.0 - p / self.peg)
            dd_trigger = self.cfg.crowd.depeg_panic / nerves
        else:
            dd = 1.0 - p / self.high if self.high > 0 else 0.0
            dd_trigger = lq.lp_flight_drawdown / nerves
        vol_trigger = lq.lp_flight_vol / nerves
        stressed = self.vol7 > vol_trigger or dd > dd_trigger or self.shock_fear > 1.5
        calm = self.vol7 < 0.5 * vol_trigger and dd < 0.5 * dd_trigger and self.shock_fear < 0.5
        w = self.lps
        if stressed and w.lp > DUST and lq.lp_flight_rate > 0:
            before = self.pool.usd
            t_out, u_out, burned_shares = self.pool.remove_liquidity(w.lp * lq.lp_flight_rate)
            if t_out > 0:
                w.lp -= burned_shares
                w.token += t_out
                w.add_usd(u_out)
                if not self._lp_fleeing:
                    self._lp_fleeing = True
                    self._event(
                        day,
                        "lp_flight",
                        "Liquidity providers pull out",
                        f"Outside LPs withdraw {_fmt_usd(before - self.pool.usd)} of the dollar side today. "
                        "Thinner pool, bigger price moves.",
                        2,
                    )
                if lq.lp_dump_share > 0:
                    self.sell_capped(w, t_out * lq.lp_dump_share, "lps", self.cfg.crowd.panic_slippage)
        elif calm:
            self._lp_fleeing = False
            if lq.lp_return_rate > 0 and w.token > DUST and w.usd > DUST:
                t_used, u_used, minted = self.pool.add_liquidity(w.token * lq.lp_return_rate, w.usd * lq.lp_return_rate)
                w.token -= t_used
                w.add_usd(-u_used)
                w.lp += minted

    # ------------------------------------------------------------- main loop

    def _record(self, i: int) -> None:
        s = self.series
        p = self.pool.price
        total = self.token0 + self.minted - self.burned
        circulating = max(total - self.locked - self.treasury.token, 0.0)
        outside_pool = max(circulating - self.pool.token, 0.0)
        s["day"][i] = i
        s["price"][i] = p
        s["peg"][i] = self.peg if self.stable else np.nan
        s["pool_usd"][i] = self.pool.usd
        s["pool_token"][i] = self.pool.token
        s["market_cap"][i] = p * circulating
        s["fdv"][i] = p * total
        s["exit_value"][i] = self.pool.quote_sell(outside_pool * (1.0 - self.tax))
        s["circulating"][i] = circulating
        s["total_supply"][i] = total
        s["burned"][i] = self.burned
        s["minted"][i] = self.minted
        s["locked"][i] = self.locked
        s["users"][i] = self.users
        s["conviction"][i] = self.conviction
        s["fear"][i] = self.fear
        s["hype"][i] = self.hype
        s["regime"][i] = REGIMES.index(self.regime)
        s["treasury_usd"][i] = self.treasury.usd
        s["treasury_token"][i] = self.treasury.token
        s["reserve_usd"][i] = self.reserve.usd
        s["protocol_failed"][i] = self.protocol_failed
        if self.stable and self.gov_pool is not None:
            s["collateral_ratio"][i] = self._collateral_ratio()
            s["backing_ratio"][i] = self._backing_ratio()
            s["gov_price"][i] = self.gov_pool.price
            s["gov_supply"][i] = self.gov0 + self.gov_minted - self.gov_burned
        else:
            for k in ("collateral_ratio", "backing_ratio", "gov_price", "gov_supply"):
                s[k][i] = np.nan
        s["redeemed"][i] = self.redeemed_today
        s["volume_usd"][i] = self.volume
        # What the attacker could walk away with right now: cash plus what the bag sells for (not paper value),
        # compared with what they could have walked away with on day 0.
        value = self.attacker.usd + self.pool.quote_sell(self.attacker.token * (1.0 - self.tax))
        s["attacker_value"][i] = value
        s["attacker_pnl"][i] = value - self.attacker_start_value
        s["external_spend_usd"][i] = self.external_spend
        s["attacker_token"][i] = self.attacker.token
        s["whale_token"][i] = sum(w.token for w in self.whales)
        s["whale_usd"][i] = sum(w.usd for w in self.whales)
        s["holder_token"][i] = self.holders.token
        s["holder_usd"][i] = self.holders.usd
        s["lp_external_share"][i] = self.lps.lp / self.pool.shares if self.pool.shares > 0 else 0.0
        s["paused"][i] = 1.0 if self.pause_left > 0 else 0.0
        s["drawdown"][i] = 1.0 - p / self.ath if self.ath > 0 else 0.0
        s["vol7"][i] = self.vol7
        tok_err, usd_err, other_err = self.check_conservation()
        s["token_error"][i] = max(tok_err, other_err)
        s["usd_error"][i] = usd_err
        self.max_error = max(self.max_error, tok_err, usd_err, other_err)
        for g in FLOW_GROUPS:
            s[f"flow_{g}"][i] = self.flows[g]

    def _update_trackers(self) -> None:
        t = self.cfg.traders
        p = self.pool.price
        self.prices.append(p)
        a_f = 2.0 / (t.momentum_fast_days + 1.0)
        a_s = 2.0 / (t.momentum_slow_days + 1.0)
        self.ema_fast += a_f * (p - self.ema_fast)
        self.ema_slow += a_s * (p - self.ema_slow)
        self.high = max(p, self.high * self.high_decay)
        self.ath = max(self.ath, p)
        if len(self.prices) >= 3:
            window = self.prices[-8:]
            rets = [math.log(max(b, 1e-300) / max(a, 1e-300)) for a, b in zip(window, window[1:])]
            mean = sum(rets) / len(rets)
            self.vol7 = math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets))

    def _detect_events(self, day: int, p_open: float) -> None:
        p = self.pool.price
        move = p / p_open - 1.0 if p_open > 0 else 0.0
        if not self.stable:
            if move <= -0.25:
                self._event(day, "crash", f"Crash: {move:.0%} in a day", "", 3 if move <= -0.5 else 2, cooldown=4)
            elif move >= 0.5:
                self._event(day, "rally", f"Rips {move:+.0%} in a day", "", 2, cooldown=4)
            if p > 2 * self.launch_price and p > self._ath_event_price * 1.5 and p >= self.ath:
                self._ath_event_price = p
                self._dd_marks_hit = set()
                self._event(
                    day,
                    "ath",
                    f"New high: {_fmt_usd(p)}",
                    f"{p / self.launch_price:.1f}x the launch price.",
                    1,
                    cooldown=7,
                )
            dd = 1.0 - p / self.ath
            for mark in (0.5, 0.9, 0.99):
                if dd >= mark and mark not in self._dd_marks_hit and self.ath > self.launch_price * 1.2:
                    self._dd_marks_hit.add(mark)
                    self._event(
                        day,
                        "drawdown",
                        f"Down {mark:.0%} from the top",
                        f"Peak was {_fmt_usd(self.ath)}.",
                        3 if mark >= 0.9 else 2,
                    )
        else:
            depeg = 1.0 - p / self.peg
            if not self._depegged and depeg > 0.02:
                self._depegged = True
                self._event(
                    day,
                    "depeg",
                    f"Depeg: {_fmt_usd(p)}",
                    f"{depeg:.1%} under the peg. Collateral ratio {self._collateral_ratio():.0%}.",
                    3,
                )
            elif self._depegged and abs(depeg) < 0.005:
                self._depegged = False
                self._event(day, "repeg", "Back on peg", f"Trading at {_fmt_usd(p)}.", 2)
            start = self.series["reserve_usd"][0]
            if start > 0 and not self._reserve_empty and self.reserve.usd < 0.01 * start:
                self._reserve_empty = True
                self._event(
                    day,
                    "reserve_empty",
                    "Reserve is empty",
                    "Redemptions are now paid in freshly minted governance tokens."
                    if self.cfg.stable.fallback_to_gov and not self.protocol_failed
                    else "Redemptions stop.",
                    3,
                )
            cr = self._collateral_ratio()
            start_cr = self.series["collateral_ratio"][0]
            for mark in (0.9, 0.8, 0.7, 0.5, 0.25):
                if start_cr >= mark and cr < mark - 1e-9 and mark not in self._cr_marks_hit:
                    self._cr_marks_hit.add(mark)
                    why = (
                        f"Gold is {self.peg - 1:+.0%} since launch and the reserve is in dollars."
                        if self.cfg.stable.peg_target == "gold"
                        else "Redemptions and minting moved the ratio."
                    )
                    self._event(day, "collateral", f"Reserve covers {cr:.0%} of the coins", why, 2)
            g_supply = self.gov0 + self.gov_minted
            crossed = [
                m for m in (2, 10, 100, 1000) if g_supply >= m * self.gov_total0 and m not in self._gov_marks_hit
            ]
            if crossed:
                self._gov_marks_hit.update(crossed)
                mark = max(crossed)
                self._event(
                    day,
                    "gov_inflation",
                    f"Governance supply x{mark}",
                    f"{_fmt_num(g_supply)} tokens now exist. Each redemption mints more.",
                    3 if mark >= 10 else 2,
                )

    def step(self, day: int) -> None:
        """Advance one day. ``day`` runs from 1 to ``days``."""
        self.flows = {g: 0.0 for g in FLOW_GROUPS}
        self.volume = 0.0
        self.redeemed_today = 0.0
        p_open = self.pool.price
        self._update_regime(day)
        self._update_hype_and_shocks(day)
        if self.stable:
            self._update_peg()
            outstanding = self.token0 + self.minted - self.burned
            self.redeem_cap_left = self.cfg.stable.max_redeem_per_day * outstanding
        self._vest(day)
        self._emit(day)
        paused = self.pause_left > 0
        self._users_and_holders_prep(day, paused)
        if not paused:
            actors: Sequence[Callable[[int], None]] = (
                self._act_holders,
                self._act_momentum,
                self._act_noise,
                self._act_whales,
                self._act_insiders,
                self._act_attacker,
                self._act_treasury,
            )
            for idx in self.rng.permutation(len(actors)):
                actors[int(idx)](day)
                self._arbitrage(day)
            self._act_lps(day)
        if self.stable:
            self._act_gov_holders(day)
            if not paused:
                self._arbitrage(day)
        self._update_trackers()
        self._detect_events(day, p_open)
        tc = self.cfg.treasury
        if paused:
            self.pause_left -= 1
        elif not self.stable and tc.circuit_breaker and abs(self.pool.price / p_open - 1.0) >= tc.breaker_move:
            self.pause_left = int(tc.breaker_days)
            self._event(
                day,
                "breaker",
                "Circuit breaker trips",
                f"Trading paused for {tc.breaker_days} day(s) after a {self.pool.price / p_open - 1:+.0%} day.",
                2,
            )
        self._record(day)

    def run(self) -> SimResult:
        self._record(0)
        for day in range(1, self.days + 1):
            self.step(day)
        return SimResult(
            config=self.cfg,
            series=self.series,
            events=sorted(self.events, key=lambda e: (e.day, -e.severity)),
            warnings=self.warnings,
            max_conservation_error=self.max_error,
        )


def run(cfg: SimConfig, strict: bool = False) -> SimResult:
    """Run one simulation. ``strict=True`` raises on any ledger drift or negative balance."""
    return Simulation(cfg, strict=strict).run()


def run_many(cfg: SimConfig, seeds: Sequence[int], series: str = "price") -> np.ndarray:
    """Run the same config under several seeds; returns an array of shape (len(seeds), days + 1)."""
    out = []
    for seed in seeds:
        res = run(replace(cfg, run=replace(cfg.run, seed=int(seed))))
        out.append(res.series[series])
    return np.vstack(out)


__all__ = ["Simulation", "SimResult", "Event", "Wallet", "run", "run_many", "FLOW_GROUPS", "ConservationError"]
