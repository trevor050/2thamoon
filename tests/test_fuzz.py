"""Fast permanent version of the tester's broad fuzz, including the new cost knob."""

import pytest

from moonsim import from_flat, run
from scripts.fuzz import execute, random_cases


@pytest.mark.parametrize(
    "label,flat", list(random_cases(300, days=(30, 60, 120))), ids=lambda v: v if isinstance(v, str) else None
)
def test_random_in_range_configs(label, flat):
    execute(flat)


def test_mixed_collateral_slider_grid():
    count = 0
    for cr in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        for fee in (0.0, 0.01, 0.03, 0.05, 0.08, 0.1):
            for seed in (7, 1, 2):
                result = run(
                    from_flat(
                        {
                            "run.mode": "stablecoin",
                            "run.days": 120,
                            "run.seed": seed,
                            "stable.collateral_ratio": cr,
                            "stable.redeem_fee": fee,
                        }
                    ),
                    strict=True,
                )
                assert result.max_conservation_error < 1e-9, (cr, fee, seed)
                count += 1
    assert count == 144
