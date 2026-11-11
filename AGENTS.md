# 2thamoon: agent notes

Crypto market sandbox. Streamlit app on Streamlit Community Cloud (https://2thamoon.streamlit.app/), deployed from
`master` running `app.py` at repo root with `requirements.txt`. Default branch is `master` (not main).

## Layout
- `moonsim/` pure engine (numpy only; a test asserts no streamlit/plotly import). `amm.py` x*y=k pool,
  `engine.py` daily loop + agents + events + ledger checks, `config.py` frozen dataclass knobs (metadata drives UI),
  `presets.py` scenarios, `story.py` headline/stats/narrative.
- `moonsim_app/` Streamlit UI. `main.py` state + layout, `charts.py` plotly figures. `app.py` just calls `main()`.
- `scripts/` `audit_legacy.py` (reproduces README v1 numbers from commit 2e4c96a), `bench.py`, `screenshots.py`.

## Commands
- `pip install -r requirements-dev.txt && pytest` (~8s, includes AppTest UI smoke tests).
- `ruff check . && ruff format --check .` (CI enforces both).
- `streamlit run app.py`; `?scenario=<preset key>` deep-links.
- Screenshots: run app on :8599, `python scripts/screenshots.py http://localhost:8599` (needs playwright + chromium).

## Invariants / hazards
- Never set price directly. All value moves via `Simulation.buy/sell`, pool LP ops, explicit mints (`_mint_tokens`,
  `gov_minted`) or `ext_in/ext_out`. `check_conservation` + strict-mode property tests catch any leak.
- New knob: add with `knob(...)` (lo/hi/label required, default in range; `test_config` checks). Mode-specific knobs
  must be listed in `STABLE_ONLY`/`TOKEN_ONLY` in `moonsim_app/main.py` or they show in the wrong mode.
- Presets are calibrated; scenario tests assert their story across seeds 0-4. Any engine behaviour change can break
  them: rerun `pytest tests/test_scenarios.py` and retune presets, not the assertions.
- `death_spiral` and `peg_defense` must differ only in `stable.reserve_ratio` and `stable.collateral_ratio` (tested).
- Streamlit renders `$...$` as LaTeX: wrap user-facing text containing dollars in `md()` in `main.py`.
- Widget state for hidden (non-rendered) knobs is garbage-collected by Streamlit; canonical values live in
  `st.session_state.flat`, widgets sync via `on_knob`. Keep that pattern.
- Streamlit dev server does not hot-reload `moonsim` reliably; restart it after engine edits.
- Attacker PnL = cash + exit value of bag now, minus the same on day 0 (not paper value).

## PC test findings (2026-11-10)
- Native Python 3.10.0 breaks current pytest/Hypothesis via CPython #88815; use a patched Python (3.13 tested).
- Exact results vary across OS/Python/NumPy builds. Same-install determinism passes; do not claim cross-platform bit identity.
- Unresolved: stablecoin CR=0.1795372879407694 + redeem_fee=0.0812446960837556, seed 7, crashes by day 110 as governance price underflows. Large cumulative churn can also trip the strict dollar ledger. Do not hide either with price floors or wider tolerances.
