import json

import pytest

from moonsim import SimConfig, from_dict, from_flat, to_dict, to_flat
from moonsim.config import KNOBS, ManualOrder, coerce_value, override


def test_every_knob_is_documented_and_default_in_range():
    assert len(KNOBS) >= 100
    for k in KNOBS:
        assert k.label
        if k.kind in ("float", "int"):
            assert k.lo is not None and k.hi is not None, k.path
            assert k.lo <= k.default <= k.hi, k.path
        if k.kind == "choice":
            assert k.default in k.choices


def test_flat_round_trip():
    cfg = override(SimConfig(), run__days=90, crowd__bass_q=2.5, run__mode="stablecoin")
    assert from_flat(to_flat(cfg)) == cfg


def test_json_round_trip_keeps_manual_orders():
    cfg = SimConfig()
    cfg = from_flat({"attacker.style": "manual"}, base=cfg)
    from dataclasses import replace

    cfg = replace(cfg, attacker=replace(cfg.attacker, manual_orders=(ManualOrder(day=5, days=2, side="buy", usd=1e4),)))
    again = from_dict(json.loads(json.dumps(to_dict(cfg))))
    assert again == cfg


def test_unknown_keys():
    with pytest.raises(KeyError):
        from_flat({"crowd.not_a_knob": 1})
    # Old exports with extra keys still load.
    assert from_dict({"crowd": {"bass_q": 2.0, "legacy_field": 3}}).crowd.bass_q == 2.0


def test_coerce_clamps_and_casts():
    spec = next(k for k in KNOBS if k.path == "run.days")
    assert coerce_value(spec, 10_000) == spec.hi
    assert coerce_value(spec, "40.6") == 41
    choice = next(k for k in KNOBS if k.path == "run.mode")
    with pytest.raises(ValueError):
        coerce_value(choice, "lol")
