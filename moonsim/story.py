"""Turn a run into words: a headline, the key numbers, and who moved the price.

Pure functions over :class:`~moonsim.engine.SimResult`, so the UI and the tests share them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .engine import FLOW_GROUPS, Event, SimResult, _fmt_num, _fmt_usd

GROUP_LABELS: Dict[str, str] = {
    "holders": "Holders and newcomers",
    "momentum": "Momentum traders",
    "noise": "Noise traders",
    "whales": "Whales",
    "insiders": "Insiders (vesting)",
    "attacker": "Attacker",
    "treasury": "Treasury",
    "arbitrageurs": "Arbitrageurs",
    "lps": "Fleeing LPs",
}


@dataclass(frozen=True)
class Stat:
    label: str
    value: str
    note: str = ""


def pct(x: float) -> str:
    return f"{x:+.0%}" if abs(x) < 10 else f"{x + 1:,.0f}x"


def headline(res: SimResult) -> str:
    s = res.series
    p = s["price"]
    days = int(s["day"][-1])
    if res.config.run.mode == "stablecoin":
        low = float(np.min(p / s["peg"]))
        end = float(p[-1] / s["peg"][-1])
        if end < 0.5:
            return (
                f"The peg broke and never came back: the coin ends at {end:.0%} of its target after {days} days, "
                f"and governance supply grew {s['gov_supply'][-1] / s['gov_supply'][0]:,.0f}x."
            )
        if low < 0.98:
            return (
                f"The peg wobbled to {low:.0%} of target but held: the coin ends at {end:.1%} of its target "
                f"after {days} days."
            )
        return f"The peg held all {days} days; the worst close was {low:.1%} of target."
    start, end = float(p[0]), float(p[-1])
    peak_i = int(np.argmax(p))
    change = end / start - 1.0
    trough = float(np.min(p[peak_i:])) / float(p[peak_i]) - 1.0
    if peak_i > 0 and p[peak_i] > 1.5 * start and end < 0.5 * p[peak_i]:
        return (
            f"Up {p[peak_i] / start:.1f}x by day {peak_i}, then down {-trough:.0%} from that top. "
            f"It ends at {_fmt_usd(end)} ({pct(change)} from launch)."
        )
    if change >= 0:
        return f"Ends at {_fmt_usd(end)} after {days} days, {pct(change)} from launch."
    return f"Ends at {_fmt_usd(end)} after {days} days, {pct(change)} from launch. Worst drawdown {-trough:.0%}."


def key_stats(res: SimResult) -> List[Stat]:
    s = res.series
    sm = res.summary()
    stats: List[Stat] = []
    if res.config.run.mode == "stablecoin":
        rel = s["price"] / s["peg"]
        stats.append(
            Stat("Final price vs peg", f"{rel[-1]:.1%}", f"low {np.min(rel):.1%} on day {int(np.argmin(rel))}")
        )
        stats.append(
            Stat("Dollar reserve", _fmt_usd(s["reserve_usd"][-1]), f"collateral ratio {s['collateral_ratio'][-1]:.0%}")
        )
        stats.append(
            Stat(
                "Governance supply",
                f"{s['gov_supply'][-1] / s['gov_supply'][0]:,.1f}x",
                f"price {_fmt_usd(s['gov_price'][0])} to {_fmt_usd(s['gov_price'][-1])}",
            )
        )
        stats.append(
            Stat(
                "Coins redeemed",
                _fmt_num(float(np.sum(s["redeemed"]))),
                f"supply now {_fmt_num(s['total_supply'][-1])}",
            )
        )
    else:
        stats.append(
            Stat("Price", _fmt_usd(sm["end_price"]), f"{pct(sm['end_price'] / sm['start_price'] - 1)} from launch")
        )
        stats.append(
            Stat("Peak", _fmt_usd(sm["peak_price"]), f"day {sm['peak_day']}, max drawdown {sm['max_drawdown']:.0%}")
        )
        stats.append(
            Stat(
                "Market cap vs. exit value",
                _fmt_usd(sm["end_market_cap"]),
                f"selling it all would fetch {_fmt_usd(sm['end_exit_value'])}",
            )
        )
        stats.append(Stat("Pool depth", _fmt_usd(sm["end_pool_usd"]), f"started at {_fmt_usd(sm['start_pool_usd'])}"))
    stats.append(Stat("Holders", _fmt_num(sm["end_users"]), f"started with {_fmt_num(s['users'][0])}"))
    exit_day = res.config.whales.exit_day
    if 0 <= exit_day < len(s["day"]) - 1 and res.config.whales.count > 0:
        paper = float(s["whale_token"][exit_day - 1] * s["price"][exit_day - 1]) / res.config.whales.count
        got = float(s["whale_usd"][-1] - s["whale_usd"][exit_day - 1])
        stats.append(Stat("What the exiting whale got", _fmt_usd(got), f"for a bag worth {_fmt_usd(paper)} on paper"))
    if res.config.attacker.style != "off":
        stats.append(Stat("Attacker result", _fmt_usd(sm["attacker_pnl"]), "vs. selling everything on day 0"))
    return stats


def flow_totals(res: SimResult, start: int = 1, end: int | None = None) -> List[Tuple[str, float]]:
    """Net dollars each group put into (+) or pulled out of (-) the pool over a window, biggest first."""
    s = res.series
    end = len(s["day"]) if end is None else end
    totals = [(g, float(np.sum(s[f"flow_{g}"][start:end]))) for g in FLOW_GROUPS]
    totals = [(g, v) for g, v in totals if abs(v) > 1.0]
    return sorted(totals, key=lambda kv: -abs(kv[1]))


def who_moved_it(res: SimResult) -> str:
    totals = flow_totals(res)
    if not totals:
        return "Nobody traded."
    buyers = [(g, v) for g, v in totals if v > 0]
    sellers = [(g, v) for g, v in totals if v < 0]
    parts = []
    if buyers:
        g, v = max(buyers, key=lambda kv: kv[1])
        parts.append(f"biggest net buyer: {GROUP_LABELS[g].lower()} ({_fmt_usd(v)})")
    if sellers:
        g, v = min(sellers, key=lambda kv: kv[1])
        parts.append(f"biggest net seller: {GROUP_LABELS[g].lower()} ({_fmt_usd(-v)})")
    return "Over the whole run, the " + "; the ".join(parts) + "."


def key_events(res: SimResult, min_severity: int = 2, limit: int = 12) -> List[Event]:
    events = [e for e in res.events if e.severity >= min_severity]
    if len(events) > limit:
        # Keep the most severe, then restore chronological order.
        events = sorted(sorted(events, key=lambda e: (-e.severity, e.day))[:limit], key=lambda e: e.day)
    return events


def narrative(res: SimResult, limit: int = 12) -> List[str]:
    lines = []
    for e in key_events(res, limit=limit):
        text = f"Day {e.day}: {e.title}."
        if e.detail:
            text += f" {e.detail}"
        lines.append(text)
    return lines
