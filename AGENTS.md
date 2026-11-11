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
- `pip install -r requirements-dev.txt && pytest` (includes AppTest UI smoke tests).
- `ruff check . && ruff format --check .` (CI enforces both).
- `streamlit run app.py`; `?scenario=<preset key>` deep-links.
- Screenshots: run app on :8599, `python scripts/screenshots.py http://localhost:8599` (needs playwright + chromium).

## Invariants / hazards
- Never set price directly. All value moves via `Simulation.buy/sell`, pool LP ops, explicit mints (`_mint_tokens`,
  `_gov_minted`) or `_ext_in/_ext_out/_external_spend`. `check_conservation` + strict-mode property tests catch any leak.
- New knob: add with `knob(...)` (lo/hi/label required, default in range; `test_config` checks). Mode-specific knobs
  must be listed in `STABLE_ONLY`/`TOKEN_ONLY` in `moonsim_app/main.py` or they show in the wrong mode.
- Presets are calibrated; scenario tests assert their story across seeds 0-9. Any engine behaviour change can break
  them: rerun `pytest tests/test_scenarios.py` and retune presets, not the assertions.
- `death_spiral` and `peg_defense` must differ only in `stable.reserve_ratio` and `stable.collateral_ratio` (tested).
- `SimResult.inapplicable_series` lists the five all-NaN token-mode stablecoin fields; use `applicable_series`/`frame()`/`to_csv()` for data, never NaN equality. `run_many` rejects inapplicable series; peg chart rejects token mode. Other outputs must be finite.
- Fuzz: `python3 scripts/fuzz.py --legacy --output docs/verification` replays tester 4556 inputs (exclude new promotion knob to preserve original RNG stream); default includes the knob. Pytest runs 300 short random configs plus 144 CR/fee/seed grid, original 31 failures and two minimized cases.
- Streamlit renders `$...$` as LaTeX: wrap user-facing text containing dollars in `md()` in `main.py`.
- Widget state for hidden (non-rendered) knobs is garbage-collected by Streamlit; canonical values live in
  `st.session_state.flat`, widgets sync via `on_knob`. Keep that pattern.
- Streamlit dev server does not hot-reload `moonsim` reliably; restart it after engine edits.
- Attacker PnL = cash + exit value of bag now, minus the same on day 0 (not paper value). Paid promotion debits cash once at pump start, capped by cash; `promo_usd_per_hype` defaults to $1,000, range $1-$1M. `_external_spend` retains the dollars outside the economy and participates in conservation; `external_spend_usd` exposes cumulative spend. Include spend in attacker cost; never deduct it twice from PnL.

## PC test findings (2026-11-10)
- Native Python 3.10.0 breaks current pytest/Hypothesis via CPython #88815; use a patched Python (3.13 tested).
- Exact results vary across OS/Python/NumPy builds. Same-install determinism passes; do not claim cross-platform bit identity.
- Governance failure is permanent: price <1e-12, either reserve <1e-9, or proposed supply >1e24. Never floor prices/reset reserves. Freeze governance trades/issuance, exclude governance backing, stop dependent mint arb; redemption pays min(promised collateral, pro-rata reserve), no governance. Stablecoin trading and collateral-only arb continue. `_check_governance` runs around governance trades and before issuance; result exposes boolean `protocol_failed` and `protocol_failure_day`.
- Cash wallets and all cumulative external/mint/burn counters use `_RunningSum` (Neumaier); `check_conservation` fsum-sums signed components, including original cash components, before rounding. Never subtract rounded trillion-dollar totals, widen 1e-9 tolerance, or add a plug account. Use `wallet.add_usd(delta)` for transfers; assigning `.usd` resets its compensated balance. Spendable cash rounds down at most one ULP to avoid overdrawing retained change. Token/governance/LP residuals also use fsum.
