"""Constant-product (x * y = k) liquidity pool, Uniswap v2 style.

Why this price model: most new tokens trade first, and often only, on an AMM pool like this.
It gives real slippage that depends on depth, it makes "thin liquidity" a number you can
read off the reserves, and it conserves everything by construction: every token that
leaves the pool went to someone who paid for it.

The swap fee is charged on the input and stays in the pool, so ``k`` only ever grows
from trading. That is how liquidity providers get paid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DUST = 1e-12


@dataclass
class Pool:
    token: float
    usd: float
    fee: float = 0.003
    shares: float = 0.0

    def __post_init__(self) -> None:
        if self.token <= 0 or self.usd <= 0:
            raise ValueError("pool reserves must be positive")
        if not 0.0 <= self.fee < 1.0:
            raise ValueError("fee must be in [0, 1)")
        if self.shares <= 0:
            self.shares = math.sqrt(self.token * self.usd)

    # -- read-only ----------------------------------------------------------

    @property
    def price(self) -> float:
        return self.usd / self.token

    @property
    def k(self) -> float:
        return self.token * self.usd

    def quote_buy(self, usd_in: float) -> float:
        """Tokens out for ``usd_in`` (no state change)."""
        if usd_in <= 0:
            return 0.0
        eff = usd_in * (1.0 - self.fee)
        return self.token * eff / (self.usd + eff)

    def quote_sell(self, token_in: float) -> float:
        """USD out for ``token_in`` (no state change)."""
        if token_in <= 0:
            return 0.0
        eff = token_in * (1.0 - self.fee)
        return self.usd * eff / (self.token + eff)

    def usd_in_for_tokens_out(self, tokens_out: float) -> float:
        """USD that must go in to receive exactly ``tokens_out``."""
        if tokens_out <= 0:
            return 0.0
        if tokens_out >= self.token:
            return math.inf
        eff = tokens_out * self.usd / (self.token - tokens_out)
        return eff / (1.0 - self.fee)

    def usd_to_reach_price(self, target: float) -> float:
        """USD to spend buying so that the post-trade price is exactly ``target`` (0 if already above).

        The fee stays in the pool, so with input ``a``: usd' = y + a and token' = k / (y + a(1 - f)).
        Setting usd'/token' = target gives a quadratic in ``a``.
        """
        if target <= self.price:
            return 0.0
        x, y, f = self.token, self.usd, self.fee
        return _positive_root(1.0 - f, y * (2.0 - f), y * y - target * x * y)

    def tokens_to_reach_price(self, target: float) -> float:
        """Tokens to sell so that the post-trade price is exactly ``target`` (0 if already below)."""
        if target >= self.price or target <= 0:
            return 0.0
        x, y, f = self.token, self.usd, self.fee
        return _positive_root(1.0 - f, x * (2.0 - f), x * x - x * y / target)

    def exit_value(self, tokens: float) -> float:
        """USD you would actually get by market-selling ``tokens`` right now."""
        return self.quote_sell(tokens)

    # -- state-changing ------------------------------------------------------

    def buy(self, usd_in: float) -> float:
        if usd_in <= 0:
            return 0.0
        out = self.quote_buy(usd_in)
        self.usd += usd_in
        self.token -= out
        return out

    def sell(self, token_in: float) -> float:
        if token_in <= 0:
            return 0.0
        out = self.quote_sell(token_in)
        self.token += token_in
        self.usd -= out
        return out

    def add_liquidity(self, token: float, usd: float) -> tuple[float, float, float]:
        """Add at the current ratio. Returns (token_used, usd_used, shares_minted)."""
        if token <= DUST or usd <= DUST:
            return 0.0, 0.0, 0.0
        ratio = min(token / self.token, usd / self.usd)
        t_used = self.token * ratio
        u_used = self.usd * ratio
        minted = self.shares * ratio
        self.token += t_used
        self.usd += u_used
        self.shares += minted
        return t_used, u_used, minted

    def remove_liquidity(self, shares: float) -> tuple[float, float, float]:
        """Burn LP shares for a pro-rata slice of both reserves. Returns (token_out, usd_out, shares_burned).

        Never takes more than 99.9% of the pool, so reserves stay positive."""
        shares = min(shares, self.shares * 0.999)
        if shares <= DUST:
            return 0.0, 0.0, 0.0
        frac = shares / self.shares
        t_out = self.token * frac
        u_out = self.usd * frac
        self.token -= t_out
        self.usd -= u_out
        self.shares -= shares
        return t_out, u_out, shares


def _positive_root(a: float, b: float, c: float) -> float:
    """Positive root of a*z^2 + b*z + c = 0 for a, b > 0 and c < 0, in the cancellation-free form."""
    if c >= 0:
        return 0.0
    return (-2.0 * c) / (b + math.sqrt(b * b - 4.0 * a * c))
