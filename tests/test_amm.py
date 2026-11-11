import math

import pytest

from moonsim.amm import Pool


def make(fee=0.003):
    return Pool(token=1_000_000.0, usd=100_000.0, fee=fee)


def test_price_is_reserve_ratio():
    assert make().price == pytest.approx(0.1)


def test_buy_moves_price_up_and_conserves():
    p = make()
    k0 = p.k
    out = p.buy(10_000.0)
    assert out > 0
    assert p.token == pytest.approx(1_000_000.0 - out)
    assert p.usd == pytest.approx(110_000.0)
    assert p.price > 0.1
    assert p.k >= k0  # fees only ever grow k


def test_slippage_grows_with_size():
    p = make()
    small = p.quote_buy(1_000.0) / 1_000.0
    big = p.quote_buy(50_000.0) / 50_000.0
    assert big < small  # fewer tokens per dollar for the bigger order


def test_round_trip_loses_money():
    p = make()
    tokens = p.buy(25_000.0)
    back = p.sell(tokens)
    assert back < 25_000.0


def test_zero_fee_round_trip_is_exact():
    p = make(fee=0.0)
    tokens = p.buy(25_000.0)
    assert p.sell(tokens) == pytest.approx(25_000.0)


@pytest.mark.parametrize("target", [0.12, 0.5, 3.0])
def test_usd_to_reach_price(target):
    p = make()
    p.buy(p.usd_to_reach_price(target))
    assert p.price == pytest.approx(target, rel=1e-9)


@pytest.mark.parametrize("target", [0.09, 0.02, 0.0001])
def test_tokens_to_reach_price(target):
    p = make()
    p.sell(p.tokens_to_reach_price(target))
    assert p.price == pytest.approx(target, rel=1e-9)


def test_usd_in_for_tokens_out_inverts_quote():
    p = make()
    usd = p.usd_in_for_tokens_out(12_345.0)
    assert p.quote_buy(usd) == pytest.approx(12_345.0, rel=1e-12)
    assert p.usd_in_for_tokens_out(p.token) == math.inf


def test_liquidity_add_remove_is_pro_rata():
    p = make()
    price = p.price
    t, u, minted = p.add_liquidity(500_000.0, 10_000.0)  # extra tokens are left unused
    assert u == pytest.approx(10_000.0)
    assert t == pytest.approx(100_000.0)
    assert p.price == pytest.approx(price)
    t_out, u_out, burned = p.remove_liquidity(minted)
    assert burned == pytest.approx(minted)
    assert t_out == pytest.approx(t) and u_out == pytest.approx(u)


def test_remove_never_empties_pool():
    p = make()
    p.remove_liquidity(p.shares * 10)
    assert p.token > 0 and p.usd > 0


def test_rejects_bad_pools():
    with pytest.raises(ValueError):
        Pool(token=0.0, usd=1.0)
    with pytest.raises(ValueError):
        Pool(token=1.0, usd=1.0, fee=1.0)


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_extreme_trade_does_not_round_the_output_reserve_to_zero(side):
    pool = Pool(token=1e-12, usd=1e-12)
    getattr(pool, side)(1e8)
    assert pool.token > 0 and pool.usd > 0
    assert math.isfinite(pool.price) and pool.price > 0
    assert pool.k >= 1e-24
