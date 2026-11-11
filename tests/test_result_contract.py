"""Inapplicable series stay NaN internally, and never become chart/export data."""

import csv
import io
import math
import re

import numpy as np
import pytest

from moonsim import from_flat, run, run_many, story


@pytest.mark.parametrize("mode", ["token", "stablecoin"])
def test_only_declared_inapplicable_series_are_nonfinite(mode):
    res = run(from_flat({"run.mode": mode, "run.days": 30}), strict=True)
    expected = {"peg", "collateral_ratio", "backing_ratio", "gov_price", "gov_supply"} if mode == "token" else set()
    assert set(res.inapplicable_series) == expected
    assert set(res.applicable_series) == set(res.series) - expected
    for name, values in res.series.items():
        if name in expected:
            assert np.isnan(values).all()
        else:
            assert np.isfinite(values).all(), name
    for value in res.summary().values():
        assert value is None or math.isfinite(value)
    text = str(story.key_stats(res)) + story.headline(res) + story.who_moved_it(res)
    assert not re.search(r"\bnan\b", text.lower())
    rows = list(csv.DictReader(io.StringIO(res.to_csv())))
    assert len(rows) == 31
    assert set(rows[0]) == set(res.applicable_series)
    assert all(value.lower() != "nan" for row in rows for value in row.values())


def test_comparison_rejects_inapplicable_series():
    cfg = from_flat({"run.days": 30})
    with pytest.raises(ValueError, match="inapplicable"):
        run_many(cfg, [1, 2], series="peg")
