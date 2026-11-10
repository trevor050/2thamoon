"""Plotly figures built from a SimResult. No Streamlit in here, so they are easy to test."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from moonsim import FLOW_GROUPS, SimResult
from moonsim.engine import Event
from moonsim.story import GROUP_LABELS

# Categorical slots in a fixed order (validated CVD-safe order from the dataviz reference palette).
# Noise traders get neutral gray: they are texture, not a story.
GROUP_COLORS: Dict[str, str] = {
    "holders": "#2a78d6",
    "attacker": "#eb6834",
    "momentum": "#1baf7a",
    "whales": "#eda100",
    "insiders": "#e87ba4",
    "treasury": "#008300",
    "arbitrageurs": "#6250d6",
    "lps": "#e34948",
    "noise": "#a3a29b",
}
PRICE = "#2a78d6"
PEG = "#8a8984"
SECOND = "#eb6834"
THIRD = "#1baf7a"
REGIME_FILL = {0: "rgba(227, 73, 72, 0.09)", 2: "rgba(27, 175, 122, 0.10)"}  # bear, bull

BOTTOM_LEGEND = dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0, title_text="")
BASE_LAYOUT = dict(
    margin=dict(l=8, r=8, t=36, b=8),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
    xaxis=dict(title_text="Day", showgrid=False, zeroline=False),
    yaxis=dict(zeroline=False),
)


def _usd_tick(series: np.ndarray) -> str:
    top = float(np.nanmax(np.abs(series))) if len(series) else 1.0
    if top < 0.01:
        return "$.2e"
    if top < 10:
        return "$.3~f"
    return "$.3~s"


def _segments(values: np.ndarray, target: int) -> List[tuple]:
    out, start = [], None
    for i, v in enumerate(values):
        if v == target and start is None:
            start = i
        elif v != target and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(values) - 1))
    return out


def _add_regimes(fig: go.Figure, res: SimResult, row: int | None = None) -> None:
    if not res.config.market.regimes:
        return
    regime = res.series["regime"].astype(int)
    for state, fill in REGIME_FILL.items():
        for a, b in _segments(regime, state):
            kw = dict(row=row, col=1) if row else {}
            fig.add_vrect(x0=a - 0.5, x1=b + 0.5, fillcolor=fill, line_width=0, layer="below", **kw)


def price_story(res: SimResult, events: Sequence[Event], log_scale: bool | None = None) -> go.Figure:
    """The headline chart: price over time, regime bands, and numbered markers for the events that mattered."""
    s = res.series
    day = s["day"]
    stable = res.config.run.mode == "stablecoin"
    price = s["price"]
    fig = go.Figure()
    _add_regimes(fig, res)
    if stable:
        fig.add_trace(
            go.Scatter(
                x=day,
                y=s["peg"],
                name="Peg target",
                line=dict(color=PEG, width=1.5, dash="dash"),
                hovertemplate="%{y:$.4f}",
            )
        )
    fig.add_trace(go.Scatter(x=day, y=price, name="Price", line=dict(color=PRICE, width=2), hovertemplate="%{y:$.4~g}"))
    if events:
        # One marker per day; same-day events share it ("1-3") so numbers never pile up.
        by_day: Dict[int, List[int]] = {}
        for i, e in enumerate(events):
            by_day.setdefault(e.day, []).append(i)
        xs = list(by_day)
        ys = [price[min(d, len(price) - 1)] for d in xs]
        labels = [f"{ix[0] + 1}" if len(ix) == 1 else f"{ix[0] + 1}-{ix[-1] + 1}" for ix in by_day.values()]
        hover = [
            "<br>".join(f"<b>{i + 1}. {events[i].title}</b><br>{_wrap(events[i].detail)}" for i in ix)
            for ix in by_day.values()
        ]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name="Events",
                text=labels,
                textposition=["top left" if i % 2 else "top right" for i in range(len(labels))],
                textfont=dict(size=13),
                marker=dict(size=11, color="#ffffff", line=dict(color="#0b0b0b", width=2)),
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
    if log_scale is None:
        lo = float(np.min(price))
        log_scale = (not stable) and lo > 0 and float(np.max(price)) / lo > 20
    fig.update_layout(**BASE_LAYOUT, height=430)
    fig.update_yaxes(
        type="log" if log_scale else "linear",
        tickformat=_usd_tick(price),
        title_text="Price (log scale)" if log_scale else "Price",
    )
    if log_scale:
        _tidy_log_axis(fig, price)
    return fig


def _tidy_log_axis(fig: go.Figure, values: np.ndarray, **kw) -> None:
    """One label per decade (or 1-2-5 for short ranges) instead of a tick on every digit."""
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    decades = np.log10(hi / lo) if lo > 0 else 0
    fig.update_yaxes(dtick=1 if decades > 2.5 else "D2", **kw)


def _wrap(text: str, width: int = 60) -> str:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return "<br>".join(lines)


def flows(res: SimResult) -> go.Figure:
    """Net dollars each group put into (+) or pulled out of (-) the pool, stacked per day or week."""
    s = res.series
    n = len(s["day"]) - 1
    bucket = 7 if n > 150 else 1
    fig = go.Figure()
    for g in FLOW_GROUPS:
        vals = s[f"flow_{g}"][1:]
        if not np.any(np.abs(vals) > 1.0):
            continue
        if bucket > 1:
            pad = (-len(vals)) % bucket
            vals = np.concatenate([vals, np.zeros(pad)]).reshape(-1, bucket).sum(axis=1)
            x = np.arange(len(vals)) * bucket + 1
        else:
            x = s["day"][1:]
        fig.add_trace(
            go.Bar(
                x=x,
                y=vals,
                name=GROUP_LABELS[g],
                marker=dict(color=GROUP_COLORS[g], line_width=0),
                hovertemplate="%{y:$,.3~s}",
            )
        )
    fig.update_layout(**BASE_LAYOUT, barmode="relative", bargap=0.15, height=340)
    fig.update_yaxes(tickformat="$.2~s", title_text="Net $ per week" if bucket > 1 else "Net $ per day")
    return fig


def liquidity(res: SimResult) -> go.Figure:
    s = res.series
    day = s["day"]
    fig = go.Figure()
    _add_regimes(fig, res)
    fig.add_trace(
        go.Scatter(
            x=day,
            y=s["market_cap"],
            name="Market cap (price x circulating)",
            line=dict(color=PRICE, width=2),
            hovertemplate="%{y:$,.3~s}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=day,
            y=s["exit_value"],
            name="What everyone could sell it for",
            line=dict(color=SECOND, width=2),
            hovertemplate="%{y:$,.3~s}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=day,
            y=s["pool_usd"],
            name="Dollars in the pool",
            line=dict(color=THIRD, width=2),
            hovertemplate="%{y:$,.3~s}",
        )
    )
    fig.update_layout(**BASE_LAYOUT, height=360)
    fig.update_yaxes(type="log", tickformat="$.2~s", title_text="USD (log scale)")
    _tidy_log_axis(fig, np.concatenate([s["market_cap"], s["exit_value"], s["pool_usd"]]))
    return fig


def supply(res: SimResult) -> go.Figure:
    s = res.series
    day = s["day"]
    fig = go.Figure()
    series = [
        ("total_supply", "Total supply", PRICE),
        ("circulating", "Circulating", THIRD),
        ("locked", "Locked (vesting)", GROUP_COLORS["insiders"]),
        ("treasury_token", "Held by treasury", GROUP_COLORS["treasury"]),
        ("burned", "Burned or redeemed", GROUP_COLORS["lps"]),
    ]
    for key, label, color in series:
        if np.any(s[key] > 0):
            fig.add_trace(
                go.Scatter(x=day, y=s[key], name=label, line=dict(color=color, width=2), hovertemplate="%{y:,.4~s}")
            )
    fig.update_layout(**BASE_LAYOUT, height=360)
    fig.update_yaxes(tickformat=".3~s", title_text="Tokens")
    return fig


def crowd(res: SimResult) -> go.Figure:
    s = res.series
    day = s["day"]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Holders", "Conviction (0 to 1)", "Hype and fear"),
    )
    fig.add_trace(
        go.Scatter(x=day, y=s["users"], name="Holders", line=dict(color=PRICE, width=2), hovertemplate="%{y:,.0f}"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=day, y=s["conviction"], name="Conviction", line=dict(color=THIRD, width=2), hovertemplate="%{y:.2f}"
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=day,
            y=s["hype"],
            name="Hype index",
            line=dict(color=GROUP_COLORS["whales"], width=2),
            hovertemplate="%{y:.2f}",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=day, y=s["fear"], name="Fear", line=dict(color=GROUP_COLORS["lps"], width=2), hovertemplate="%{y:.2f}"
        ),
        row=3,
        col=1,
    )
    fig.update_layout(**{**BASE_LAYOUT, "xaxis": {}, "legend": BOTTOM_LEGEND}, height=560, showlegend=True)
    fig.update_xaxes(showgrid=False)
    fig.update_xaxes(title_text="Day", row=3, col=1)
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    return fig


def stablecoin(res: SimResult) -> go.Figure:
    s = res.series
    day = s["day"]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Reserve dollars per coin owed (collateral ratio)",
            "Governance token price",
            "Governance token supply",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=day,
            y=s["collateral_ratio"],
            name="Collateral ratio",
            line=dict(color=PRICE, width=2),
            hovertemplate="%{y:.1%}",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=day,
            y=np.minimum(s["backing_ratio"], 5),
            name="Backing incl. governance tokens (capped at 500%)",
            line=dict(color=PEG, width=1.5, dash="dash"),
            hovertemplate="%{y:.1%}",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=day,
            y=s["gov_price"],
            name="Governance price",
            line=dict(color=SECOND, width=2),
            hovertemplate="%{y:$.4~g}",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=day,
            y=s["gov_supply"],
            name="Governance supply",
            line=dict(color=GROUP_COLORS["arbitrageurs"], width=2),
            hovertemplate="%{y:,.4~s}",
        ),
        row=3,
        col=1,
    )
    fig.update_layout(**{**BASE_LAYOUT, "xaxis": {}, "legend": BOTTOM_LEGEND}, height=600)
    fig.update_xaxes(showgrid=False)
    fig.update_xaxes(title_text="Day", row=3, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    gp = s["gov_price"]
    if np.nanmax(gp) / max(np.nanmin(gp), 1e-300) > 20:
        fig.update_yaxes(type="log", exponentformat="power", tickprefix="$", row=2, col=1)
        _tidy_log_axis(fig, gp, row=2, col=1)
    gs = s["gov_supply"]
    if np.nanmax(gs) / max(np.nanmin(gs), 1e-300) > 20:
        fig.update_yaxes(type="log", tickformat=".2~s", row=3, col=1)
        _tidy_log_axis(fig, gs, row=3, col=1)
    return fig


def attacker(res: SimResult) -> go.Figure:
    s = res.series
    day = s["day"]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("What the attacker could walk away with", "Tokens the attacker holds"),
    )
    fig.add_trace(
        go.Scatter(
            x=day,
            y=s["attacker_value"],
            name="Cash + bag at exit value",
            line=dict(color=GROUP_COLORS["attacker"], width=2),
            hovertemplate="%{y:$,.3~s}",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=day, y=s["attacker_token"], name="Tokens held", line=dict(color=PEG, width=2), hovertemplate="%{y:,.3~s}"
        ),
        row=2,
        col=1,
    )
    fig.update_layout(**{**BASE_LAYOUT, "xaxis": {}}, height=440, showlegend=False)
    fig.update_xaxes(showgrid=False)
    fig.update_xaxes(title_text="Day", row=2, col=1)
    fig.update_yaxes(tickformat="$.3~s", row=1, col=1)
    return fig


def luck(prices: np.ndarray, this_run: np.ndarray, stable: bool) -> go.Figure:
    """Fan chart: the same knobs under many seeds, with this run drawn on top."""
    day = np.arange(prices.shape[1])
    p10, p50, p90 = np.percentile(prices, [10, 50, 90], axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=day, y=p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=day,
            y=p10,
            fill="tonexty",
            fillcolor="rgba(42,120,214,0.15)",
            line=dict(width=0),
            name="10th to 90th percentile",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=day, y=p50, name="Median seed", line=dict(color=PEG, width=2, dash="dash"), hovertemplate="%{y:$.4~g}"
        )
    )
    fig.add_trace(
        go.Scatter(x=day, y=this_run, name="This run", line=dict(color=PRICE, width=2), hovertemplate="%{y:$.4~g}")
    )
    lo = float(np.min(prices))
    log_scale = (not stable) and lo > 0 and float(np.max(prices)) / lo > 20
    fig.update_layout(**BASE_LAYOUT, height=360)
    fig.update_yaxes(type="log" if log_scale else "linear", tickformat=_usd_tick(prices))
    if log_scale:
        _tidy_log_axis(fig, prices)
    return fig
