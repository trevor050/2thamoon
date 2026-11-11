"""Bounded strict-conservation fuzz, ported from the tester's 2026-11-10 harness.

python3 scripts/fuzz.py --legacy --output docs/verification
Replays its exact 4,556 inputs (138 original knobs, seed 88209). Without --legacy,
includes the new promotion-price knob. No UI dependencies; no network access.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moonsim import SimConfig, from_flat
from moonsim.config import KNOBS
from moonsim.engine import Simulation

NEW_KNOB = "attacker.promo_usd_per_hype"


def edge(knob, high):
    if knob.kind == "bool":
        return high
    if knob.kind == "choice":
        return knob.choices[-1 if high else 0]
    return knob.hi if high else knob.lo


def endpoint_cases(knobs=KNOBS):
    for mode in ("token", "stablecoin"):
        for knob in knobs:
            for high in (False, True):
                yield f"{mode}:{knob.path}:{high}", {"run.mode": mode, "run.days": 60, knob.path: edge(knob, high)}
        for high in (False, True):
            flat = {knob.path: edge(knob, high) for knob in knobs}
            flat["run.mode"] = mode
            yield f"all-{high}-{mode}", flat


def random_cases(count=300, *, knobs=KNOBS, days=(30, 60, 120, 365, 1095)):
    rng = random.Random(88209)
    for i in range(count):
        flat = {
            "run.mode": rng.choice(("token", "stablecoin")),
            "run.days": rng.choice(days),
            "run.seed": rng.randrange(1000001),
        }
        for knob in rng.sample(list(knobs)[3:], rng.choice((5, 25, 60, 135))):
            if rng.random() < 0.65 or knob.kind in ("bool", "choice"):
                value = edge(knob, rng.choice((False, True)))
            elif knob.log and knob.lo > 0:
                value = math.exp(rng.uniform(math.log(knob.lo), math.log(knob.hi)))
            else:
                value = rng.uniform(knob.lo, knob.hi)
            flat[knob.path] = value
        yield f"random-{i}", flat


def check_output(result):
    for name in result.inapplicable_series:
        assert np.isnan(result.series[name]).all(), name
    for name, values in result.applicable_series.items():
        assert np.isfinite(values).all(), f"nonfinite {name}"
        if name in ("price", "pool_usd", "pool_token", "gov_price"):
            assert (values > 0).all(), f"nonpositive {name}"
        if (
            (name.endswith("_usd") and not name.startswith("flow_"))
            or name.endswith("_token")
            or name in ("users", "minted", "burned", "locked", "total_supply")
        ):
            assert (values >= -1e-9).all(), f"negative {name}"


def execute(flat):
    cfg = from_flat(flat)
    sim = Simulation(cfg, strict=True)
    assert math.isclose(sim.token0, cfg.supply.total_supply, rel_tol=1e-12), "initial supply missing"
    result = sim.run()
    check_output(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--random-count", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=Path("docs/verification"))
    args = parser.parse_args()
    knobs = tuple(k for k in KNOBS if not args.legacy or k.path != NEW_KNOB)
    args.output.mkdir(parents=True, exist_ok=True)
    failures, count, worst = [], 0, 0.0
    started = time.perf_counter()
    for label, flat in (*endpoint_cases(knobs), *random_cases(args.random_count, knobs=knobs)):
        count += 1
        try:
            result = execute(flat)
            worst = max(worst, result.max_conservation_error)
        except Exception as exc:
            failures.append(
                {
                    "label": label,
                    "flat": flat,
                    "type": type(exc).__name__,
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                }
            )
            print(f"FAIL {label}: {type(exc).__name__}: {exc}", flush=True)
        if count % 200 == 0:
            print(f"{count} runs, {len(failures)} failures, {time.perf_counter() - started:.1f}s", flush=True)
    raw = []
    for path, values in {
        "liquidity.pool_usd": [0.0],
        "liquidity.lp_fee": [1.0],
        "supply.transfer_tax": [1.0],
        "run.days": [0, 10000],
        "crowd.initial_users": [1.0],
    }.items():
        for value in values:
            cfg = SimConfig()
            section, name = path.split(".")
            cfg = replace(cfg, **{section: replace(getattr(cfg, section), **{name: value})})
            try:
                Simulation(cfg, strict=True).run()
                outcome = "ok"
            except Exception as exc:
                outcome = f"{type(exc).__name__}: {exc}"
            raw.append({"path": path, "value": value, "outcome": outcome})
    summary = {
        "runs": count,
        "failures": len(failures),
        "worst_error": worst,
        "seconds": time.perf_counter() - started,
        "legacy_inputs": args.legacy,
        "raw_api_probes": raw,
    }
    (args.output / "fuzz-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    (args.output / "fuzz-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
