# Correctness verification, 2026-11-10

All implementation and local verification ran on the assigned Mac. No Windows PC or other personal machine was contacted. Baseline: `b6b173bb036f7f7ffaa0b764bca02a47ea0f9aac`. Runtime: macOS on a 2020 M1 MacBook Air, Python 3.14.7, NumPy 2.4.2. The engine still requires only NumPy. Local test tools were installed into a disposable directory, then removed; Streamlit, pandas and Plotly were not installed.

## Results

- `pytest`: **437 passed, 1 skipped**, 13.00s. The skipped module is Streamlit AppTest, delegated to the existing Python 3.10/3.13 CI matrix. New UI checks cover failure disclosure, finite chart inputs, inapplicable chart rejection and DataFrame omissions. These were not executed on the Mac.
- `ruff check . && ruff format --check .`: passed.
- All **31 original failing configurations** are preserved as regression fixtures and pass strict mode, plus both exact minimized inputs, ordinary mixed-collateral slider values, and zero-cash promotion.
- Permanent fast property test: **300** deterministic random in-range configurations across both modes, including the new promotion-price knob, with strict conservation and finite applicable outputs.
- Requested **144/144** slider grid: 120 days; CR `[.05,.1,.15,.2,.25,.3,.4,.5]`; redemption fee `[0,.01,.03,.05,.08,.1]`; `run.seed` `[7,1,2]`.
- All seven presets are deterministic and conserve across **seeds 0–9**; existing qualitative assertions pass on that expanded set. Pump-and-dump settings are unchanged, with six hype units now costing $6,000. Unlock-cliff average bull duration changed **400 → 600 days** after seed 6 exposed a pre-cliff macro reversal; assertions were not relaxed. The death-spiral preset still inflates governance supply more than 10× and loses its peg; its default seed does not reach the new terminal cutoff during its 120 days.
- `python3 scripts/fuzz.py --legacy --output docs/verification`: **4,556 runs, zero failures**, 121.27s. Worst relative ledger error **2.1167675179776582e-13**, against the unchanged **1e-9** tolerance. See [summary](fuzz-summary.json), [log](fuzz.log), [failure list](fuzz-failures.json).
- The tester's **original evidence-folder `fuzz.py` was also executed directly**: **4,556 runs, zero failures**, 121.35s, same worst error. Only its output directory was redirected here and the newly added knob excluded to preserve its original knob enumeration/RNG stream. Its source hash and adaptations are in [the receipt](original-harness/receipt.json); [summary](original-harness/fuzz-summary.json), [log](original-harness/fuzz.log), [failure list](original-harness/fuzz-failures.json). The evidence folder remained read-only.
- The evidence-folder `economic_audit.py` passed its Decimal AMM calculations, inventory caps and isolated arbitrage checks. [Output](economic-audit.json). Command: `PYTHONPATH=. python3 /Users/trevorrosato/CodeProjects/portfolio-wave/_pc-evidence/web-b/2thamoon/economic_audit.py`.

## Performance

`python3 scripts/bench.py` uses the median of seven runs per preset, then the median of three 30-seed batches. Before and after were measured locally with the same runtime; this is a shared machine, not an isolated benchmark. Individual runs remain under 32ms and the 30-seed batch under half a second. Exact summation adds roughly 4ms to the default 365-day run.

| Workload | Before (ms) | After (ms) |
|---|---:|---:|
| sandbox | 11.4 | 15.5 |
| pump_and_dump | 4.6 | 5.9 |
| whale_exit | 3.7 | 4.8 |
| death_spiral | 10.5 | 14.8 |
| peg_defense | 7.1 | 11.1 |
| gold_peg | 19.7 | 31.3 |
| unlock_cliff | 7.5 | 10.2 |
| 30-seed check | 343.0 | 480.0 |

[Raw before](benchmark-before.txt) · [raw after](benchmark-after.txt) · [data](benchmark.json) · [chart](benchmark.html)

## Numeric and output limits

The four ledgers retain the 1e-9 relative limit and strict negative/nonfinite checks. Neumaier counters retain cumulative transfer/mint/burn low bits; cash wallets also retain low bits as very large cohorts drain. `math.fsum` compares all signed components before rounding, including initial cash components. There is no balancing account, price floor or relaxed strict mode.

Governance failure is an explicit permanent protocol state, triggered before float exhaustion, with exactly one failure event. Governance trading freezes at the final reserves; governance backing and issuance cease. Redemptions pay only available pro-rata collateral up to the promised collateral share, and collateral-only arbitrage remains possible. Stablecoin trading continues. See README for cutoff units and policy.

Token-mode raw results intentionally contain NaN only in the five declared stablecoin-only series. Public data/export helpers omit those fields; comparisons reject them. Same seed and knobs on the same platform with the same Python/NumPy versions give identical results. Cross-platform bit identity remains unsupported and was not pursued.
