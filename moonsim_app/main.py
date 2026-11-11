"""The Streamlit front end. Everything here is presentation; the model lives in ``moonsim``."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from moonsim import SimResult, from_dict, from_flat, run, run_many, story, to_dict, to_flat
from moonsim.config import KNOB_BY_PATH, KNOBS, SECTION_TITLES, KnobSpec, ManualOrder, SimConfig
from moonsim.presets import PRESET_BY_KEY

from . import charts

DEFAULT_PRESET = "pump_and_dump"
PRESET_ORDER = ["pump_and_dump", "whale_exit", "death_spiral", "peg_defense", "gold_peg", "unlock_cliff", "sandbox"]
LUCK_SEEDS = 30

SECTIONS = {
    "token": ["run", "market", "crowd", "liquidity", "supply", "whales", "traders", "attacker", "treasury"],
    "stablecoin": [
        "run",
        "stable",
        "crowd",
        "liquidity",
        "supply",
        "whales",
        "traders",
        "attacker",
        "treasury",
        "market",
    ],
}
STABLE_ONLY = {
    "crowd.depeg_tolerance",
    "crowd.depeg_panic",
    "crowd.depeg_sensitivity",
    "crowd.backing_sensitivity",
    "crowd.yield_sensitivity",
    "crowd.dilution_fear",
}
TOKEN_ONLY = {
    "supply.launch_price",
    "supply.insiders_share",
    "supply.vesting_cliff_days",
    "supply.vesting_days",
    "supply.insider_sell_rate",
    "supply.max_supply",
    "treasury.band_defense",
    "treasury.floor_x_launch",
    "treasury.ceiling_x_launch",
    "treasury.buyback_per_year",
    "treasury.burn_buybacks",
    "treasury.circuit_breaker",
    "treasury.breaker_move",
    "treasury.breaker_days",
    "crowd.fomo",
    "crowd.return_sensitivity",
    "whales.take_profit_multiple",
    "whales.take_profit_rate",
    "whales.basis_multiple",
    "attacker.accumulate_days",
    "attacker.accumulate_share",
    "attacker.pump_days",
    "attacker.promo_hype",
    "attacker.promo_usd_per_hype",
    "attacker.dump_multiple",
    "attacker.dump_days",
    "attacker.dump_slippage",
}
CHOICE_LABELS = {
    "token": "Token",
    "stablecoin": "Stablecoin",
    "off": "Off",
    "pump_and_dump": "Pump and dump",
    "bear_raid": "Bear raid",
    "manual": "Manual orders",
    "usd": "US dollar",
    "gold": "Gold",
    "bull": "Bull",
    "bear": "Bear",
    "neutral": "Sideways",
}

ss = st.session_state


def md(text: str) -> str:
    """Escape dollar signs so Streamlit does not render '$3M ... $7M' as LaTeX."""
    return text.replace("$", "\\$")


# --------------------------------------------------------------------------- state


def _widget_key(path: str) -> str:
    return f"k:{path}"


def _typed(spec: KnobSpec, value):
    if spec.kind == "int":
        return int(value)
    if spec.kind == "float":
        return float(value)
    return value


def load_flat(flat: Dict[str, object], orders: List[dict], label: str) -> None:
    ss.flat = {path: _typed(KNOB_BY_PATH[path], v) for path, v in flat.items()}
    ss.base_flat = dict(ss.flat)
    ss.orders = [dict(o) for o in orders]
    ss.loaded = label
    for path, v in ss.flat.items():
        ss[_widget_key(path)] = v


def load_preset(key: str) -> None:
    preset = PRESET_BY_KEY[key]
    load_flat(to_flat(preset.config), [], key)


def on_pick() -> None:
    key = ss.get("preset_pick")
    if key:
        load_preset(key)
        st.query_params["scenario"] = key


def on_knob(path: str) -> None:
    ss.flat[path] = ss[_widget_key(path)]


def reroll() -> None:
    seed = int(np.random.default_rng().integers(0, 1_000_000))
    ss.flat["run.seed"] = seed
    ss[_widget_key("run.seed")] = seed


def reset() -> None:
    if ss.loaded in PRESET_BY_KEY:
        load_preset(ss.loaded)
    else:
        load_flat(ss.base_flat, ss.orders, ss.loaded)


def on_upload() -> None:
    file = ss.get("upload")
    if file is None:
        return
    try:
        cfg = from_dict(json.loads(file.getvalue().decode("utf-8")))
    except (ValueError, TypeError, KeyError) as exc:
        ss.upload_error = f"Could not read that file: {exc}"
        return
    ss.upload_error = ""
    load_flat(to_flat(cfg), [vars(o) for o in cfg.attacker.manual_orders], "custom")
    ss.preset_pick = None


def init_state() -> None:
    if "flat" not in ss:
        # ?scenario=<key> deep-links straight into a preset.
        key = st.query_params.get("scenario", DEFAULT_PRESET)
        key = key if key in PRESET_BY_KEY else DEFAULT_PRESET
        load_preset(key)
        ss.preset_pick = key
        ss.deep = False
        ss.upload_error = ""


def current_config() -> SimConfig:
    cfg = from_flat(ss.flat)
    if cfg.attacker.style == "manual" and ss.orders:
        orders = tuple(
            ManualOrder(day=int(o["day"]), days=max(int(o["days"]), 1), side=str(o["side"]), usd=float(o["usd"]))
            for o in ss.orders
            if o.get("day") is not None and o.get("usd") is not None and o.get("side") in ("buy", "sell")
        )
        cfg = replace(cfg, attacker=replace(cfg.attacker, manual_orders=orders))
    return cfg


def changed_knobs() -> int:
    return sum(1 for p, v in ss.flat.items() if ss.base_flat.get(p) != v)


# ------------------------------------------------------------------------- compute


def _key(cfg: SimConfig) -> str:
    return json.dumps(to_dict(cfg), sort_keys=True)


@st.cache_data(max_entries=64, show_spinner=False)
def simulate(cfg_json: str) -> SimResult:
    return run(from_dict(json.loads(cfg_json)))


@st.cache_data(max_entries=16, show_spinner=False)
def luck_check(cfg_json: str, n: int) -> np.ndarray:
    cfg = from_dict(json.loads(cfg_json))
    prices = run_many(cfg, range(10_000, 10_000 + n))
    if cfg.run.mode == "stablecoin":
        prices = prices / run_many(cfg, range(10_000, 10_000 + n), series="peg")
    return prices


# ------------------------------------------------------------------------- widgets


def render_knob(spec: KnobSpec) -> None:
    key = _widget_key(spec.path)
    if key not in ss:
        ss[key] = ss.flat[spec.path]
    common = dict(key=key, help=spec.help or None, on_change=on_knob, args=(spec.path,))
    if spec.kind == "bool":
        st.toggle(spec.label, **common)
    elif spec.kind == "choice":
        st.selectbox(spec.label, spec.choices, format_func=lambda c: CHOICE_LABELS.get(c, c), **common)
    elif spec.kind == "int":
        lo, hi = int(spec.lo), int(spec.hi)
        step = int(spec.step or 1)
        if hi - lo > 2000:
            st.number_input(spec.label, min_value=lo, max_value=hi, step=step, **common)
        else:
            st.slider(spec.label, min_value=lo, max_value=hi, step=step, **common)
    else:
        lo, hi = float(spec.lo), float(spec.hi)
        step = float(spec.step or (hi - lo) / 100)
        if spec.log:
            st.number_input(spec.label, min_value=lo, max_value=hi, step=step, format=spec.fmt or "%.4g", **common)
        else:
            st.slider(spec.label, min_value=lo, max_value=hi, step=step, format=spec.fmt, **common)


def visible(spec: KnobSpec, mode: str) -> bool:
    if spec.path in ("run.seed",):
        return False  # lives next to the scenario picker
    if mode == "token" and (spec.section == "stable" or spec.path in STABLE_ONLY):
        return False
    if mode == "stablecoin" and spec.path in TOKEN_ONLY:
        return False
    return True


KNOB_COLUMNS = 4


def render_grid(specs: List[KnobSpec]) -> None:
    """Lay knobs out in rows across the page so a whole section can be scanned at once."""
    for start in range(0, len(specs), KNOB_COLUMNS):
        row = specs[start : start + KNOB_COLUMNS]
        cols = st.columns(KNOB_COLUMNS, gap="medium", vertical_alignment="bottom")
        for col, spec in zip(cols, row):
            with col:
                render_knob(spec)


def render_knobs(mode: str) -> None:
    """The knobs sit on the page, right above the chart they change, one tab per part of the market."""
    shown = [k for k in KNOBS if visible(k, mode)]
    n_deep = sum(k.advanced for k in shown)
    title_col, deep_col = st.columns([3, 1.4], vertical_alignment="bottom")
    with title_col:
        st.markdown("**Mess with it.** Every knob reruns the market below.")
    with deep_col:
        st.toggle(
            f"Show all {len(shown)} knobs",
            key="deep",
            help=f"Off: the {len(shown) - n_deep} knobs that matter most. On: every knob, including {n_deep} deeper ones.",
        )
    deep = ss.deep
    by_section: Dict[str, List[KnobSpec]] = {}
    for k in shown:
        by_section.setdefault(k.section, []).append(k)
    sections = [s for s in SECTIONS[mode] if by_section.get(s)]
    for section, tab in zip(sections, st.tabs([SECTION_TITLES[s] for s in sections])):
        specs = by_section[section]
        extra = [k for k in specs if k.advanced]
        with tab:
            render_grid([k for k in specs if deep or not k.advanced])
            if extra and not deep:
                st.caption(f"{len(extra)} more here with every knob shown.")
            if section == "attacker" and ss.flat["attacker.style"] == "manual":
                render_orders()


def render_orders() -> None:
    st.caption("Your orders. USD amounts; sells are sized at the price on the day they run.")
    frame = pd.DataFrame(
        ss.orders or [{"day": 30, "days": 3, "side": "sell", "usd": 50_000.0}], columns=["day", "days", "side", "usd"]
    )
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        hide_index=True,
        key="orders_editor",
        column_config={
            "day": st.column_config.NumberColumn("Start day", min_value=0, step=1),
            "days": st.column_config.NumberColumn("Days", min_value=1, step=1),
            "side": st.column_config.SelectboxColumn("Side", options=["buy", "sell"]),
            "usd": st.column_config.NumberColumn("USD", min_value=0.0, format="$%.0f"),
        },
    )
    ss.orders = edited.dropna().to_dict("records")


# ----------------------------------------------------------------------------- page


def render_header() -> None:
    st.title("2thamoon")
    st.markdown(
        "A crypto market sandbox. Every price comes from real trades against a liquidity pool, every token "
        "and dollar is accounted for, and you get the knobs. Pick a scenario to see what happens and why."
    )
    st.pills(
        "Scenario",
        PRESET_ORDER,
        format_func=lambda k: PRESET_BY_KEY[k].title,
        key="preset_pick",
        on_change=on_pick,
        label_visibility="collapsed",
    )
    loaded = PRESET_BY_KEY.get(ss.loaded)
    n_changed = changed_knobs()
    if loaded is not None:
        st.markdown(md(loaded.blurb))
    seed_col, roll_col, note_col = st.columns([1.2, 1.2, 4], vertical_alignment="bottom")
    with seed_col:
        render_knob(KNOB_BY_PATH["run.seed"])
    with roll_col:
        st.button(
            "Roll new luck",
            icon=":material/casino:",
            on_click=reroll,
            width="stretch",
            help="Same knobs, different random seed.",
        )
    with note_col:
        if n_changed:
            base = loaded.title if loaded else "your file"
            st.caption(f"{n_changed} knob{'s' if n_changed != 1 else ''} changed from {base}.")
            st.button("Reset", on_click=reset, type="tertiary")


def render_results(cfg: SimConfig, res: SimResult) -> None:
    stable = cfg.run.mode == "stablecoin"
    for w in res.warnings:
        st.warning(w)
    st.subheader(md(story.headline(res)))

    stats = story.key_stats(res)
    with st.container(key="stats"):
        cols = st.columns(len(stats))
        for col, stat in zip(cols, stats):
            with col:
                st.metric(stat.label, stat.value)
                st.caption(md(stat.note))

    events = story.key_events(res, limit=10)
    left, right = st.columns([2.3, 1], gap="medium")
    with left:
        st.plotly_chart(charts.price_story(res, events), key="price_chart", config={"displaylogo": False})
        if cfg.market.regimes:
            st.caption("Green bands are bull markets, red bands are bear markets. Numbered dots match the list.")
    with right:
        st.markdown("**What happened**")
        if events:
            st.markdown(md("\n".join(f"{i + 1}. **Day {e.day}: {e.title}.** {e.detail}" for i, e in enumerate(events))))
        else:
            st.markdown("Nothing dramatic. Sometimes the market just drifts.")

    st.markdown("**Who moved the price**")
    st.caption(
        md(story.who_moved_it(res)) + " Bars above zero are dollars coming into the pool (buying), below zero "
        "are dollars going out (selling)."
    )
    st.plotly_chart(charts.flows(res), key="flows_chart", config={"displaylogo": False})

    names = ["Liquidity", "Supply", "Crowd"]
    if stable:
        names.insert(0, "Peg mechanics")
    if cfg.attacker.style != "off":
        names.append("Attacker")
    names += ["Was it luck?", "Data"]
    tabs = dict(zip(names, st.tabs(names)))
    if stable:
        with tabs["Peg mechanics"]:
            if res.protocol_failure_day is not None:
                st.warning(
                    f"Protocol failed on day {res.protocol_failure_day}. Governance payouts and dependent "
                    "minting have stopped permanently. Redemptions receive only their pro-rata available collateral."
                )
            st.caption(
                "Normally, redemptions pay from the dollar reserve at the collateral ratio, the rest in newly "
                "minted governance tokens that arbitrageurs dump. Watch the supply line."
            )
            st.plotly_chart(charts.stablecoin(res), key="stable_chart", config={"displaylogo": False})
    with tabs["Liquidity"]:
        st.caption(
            "Market cap multiplies the price by every token. The orange line is what all those tokens would "
            "actually fetch if sold into the pool right now. The gap is the whole point."
        )
        st.plotly_chart(charts.liquidity(res), key="liq_chart", config={"displaylogo": False})
    with tabs["Supply"]:
        st.plotly_chart(charts.supply(res), key="supply_chart", config={"displaylogo": False})
    with tabs["Crowd"]:
        st.plotly_chart(charts.crowd(res), key="crowd_chart", config={"displaylogo": False})
    if "Attacker" in tabs:
        with tabs["Attacker"]:
            st.caption(
                "Valued at what the bag would actually sell for, including promotion costs paid from attacker cash."
            )
            st.plotly_chart(charts.attacker(res), key="attacker_chart", config={"displaylogo": False})
    with tabs["Was it luck?"]:
        st.caption(f"Runs these exact knobs under {LUCK_SEEDS} other seeds and shows the spread.")
        if st.toggle("Run the luck check", key="luck_on"):
            prices = luck_check(_key(cfg), LUCK_SEEDS)
            this = res.series["price"] / (res.series["peg"] if stable else 1.0)
            st.plotly_chart(charts.luck(prices, this, stable), key="luck_chart", config={"displaylogo": False})
            ends = prices[:, -1] / prices[:, 0]
            up = int(np.sum(ends > 1.0))
            st.caption(
                f"{up} of {LUCK_SEEDS} seeds end above where they started. "
                f"Median end: {np.median(ends):.2f}x the start."
            )
    with tabs["Data"]:
        if res.inapplicable_series:
            st.caption("Stablecoin-only fields are inapplicable in token mode and omitted from the table and CSV.")
        frame = res.frame()
        st.dataframe(frame, height=320, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download this run (CSV)",
                res.to_csv().encode(),
                "2thamoon_run.csv",
                "text/csv",
                width="stretch",
            )
        with c2:
            st.download_button(
                "Download these knobs (JSON)",
                json.dumps(to_dict(cfg), indent=2).encode(),
                "2thamoon_knobs.json",
                "application/json",
                width="stretch",
            )
        st.file_uploader("Load knobs from a JSON file", type=["json"], key="upload", on_change=on_upload)
        if ss.get("upload_error"):
            st.error(ss.upload_error)


def render_footer() -> None:
    with st.expander("How the simulation works (and what it ignores)"):
        st.markdown(
            "- **Price** is the ratio of reserves in a constant-product pool (x * y = k, like Uniswap v2). "
            "Nothing else sets it. Big orders get real slippage, and thin pools move a lot.\n"
            "- **Agents** trade against that pool with the money they actually have: holders who join along a Bass "
            "adoption curve and churn when scared, momentum and noise traders, whales, vesting insiders, "
            "liquidity providers who flee volatility, an optional attacker and a treasury.\n"
            "- **Stablecoin mode** adds mint/redeem arbitrage at the peg, a dollar reserve, and a governance "
            "token that is minted to pay redemptions the reserve cannot cover.\n"
            "- **Every step** the engine checks that tokens, dollars, governance tokens and LP shares add up. "
            "Same seed, platform and Python/NumPy versions give the same run.\n"
            "- **Not modelled:** order books and multiple venues, leverage and shorting, MEV, gas, oracles with "
            "delay, real people. It is a toy for building intuition, not a forecast. Details in the "
            "[README](https://github.com/trevor050/2thamoon#how-the-sim-works)."
        )


def main() -> None:
    st.set_page_config(page_title="2thamoon", page_icon="🌕", layout="wide")
    st.markdown(
        "<style>[data-testid='stMetricValue']{font-size:1.65rem}"
        "[data-testid='stMetric']{margin-bottom:-0.6rem}"
        # Two stat tiles per row on phones instead of six stacked rows.
        "@media (max-width: 640px){.st-key-stats [data-testid='stHorizontalBlock']{flex-flow:row wrap;gap:0.5rem 1rem}"
        ".st-key-stats [data-testid='stColumn']"
        "{flex:1 1 calc(50% - 1rem)!important;min-width:calc(50% - 1rem)!important}}"
        "</style>",
        unsafe_allow_html=True,
    )
    init_state()
    mode = ss.flat["run.mode"]
    render_header()
    render_knobs(mode)
    cfg = current_config()
    res = simulate(_key(cfg))
    render_results(cfg, res)
    render_footer()
