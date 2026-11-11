"""Time every preset (median of several runs) and the 30-seed luck check.

python scripts/bench.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moonsim import run, run_many
from moonsim.presets import PRESETS


def timed(fn, repeat: int = 7) -> float:
    samples = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return statistics.median(samples) * 1000


def main() -> None:
    run(PRESETS[0].config)  # warm-up
    for p in PRESETS:
        ms = timed(lambda p=p: run(p.config))
        print(f"{p.key:15s} {p.config.run.days:4d} days  {ms:6.1f} ms")
    cfg = PRESETS[0].config
    print(f"luck check: 30 seeds x {cfg.run.days} days  {timed(lambda: run_many(cfg, range(30)), 3):6.0f} ms")


if __name__ == "__main__":
    main()
