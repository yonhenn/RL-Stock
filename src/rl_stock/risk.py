"""Risk metrics and portfolio analytics utilities.

These helpers are used both inside the trading environment (online reward
computation) and during post-episode evaluation. All functions are pure and
operate on numpy arrays; they never mutate their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-8
TRADING_DAYS_PER_YEAR = 252


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio of a return series.

    ``returns`` should be simple period returns (not log returns), already in
    decimal form (``0.01`` for 1%).
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0

    excess = returns - risk_free_rate / periods_per_year
    std = float(np.std(excess, ddof=1))
    if std < EPS:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sortino ratio. Only penalises downside deviation."""
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0

    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if downside.size == 0:
        return 0.0
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std < EPS:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Return the maximum drawdown (positive number, e.g. ``0.21`` = 21%)."""
    equity = np.asarray(equity_curve, dtype=float)
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / np.maximum(running_max, EPS)
    return float(-np.min(drawdown))


def compute_calmar_ratio(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised return divided by max drawdown."""
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return 0.0
    max_dd = compute_max_drawdown(equity_curve)
    if max_dd < EPS:
        return 0.0
    annual_return = float(np.mean(returns) * periods_per_year)
    return annual_return / max_dd


def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range using Wilder's smoothing.

    Output has the same length as the inputs; warm-up entries are filled with
    the simple mean of the first ``period`` true ranges.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    atr = np.zeros(n, dtype=float)
    if n == 0:
        return atr

    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    if n <= period:
        atr[:] = float(np.mean(tr))
        return atr

    initial = float(np.mean(tr[:period]))
    atr[:period] = initial
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_turbulence(
    close: np.ndarray,
    lookback: int = 252,
) -> np.ndarray:
    """Single-asset turbulence index (Kritzman & Li, 2010), 1-D special case.

    The original turbulence index is the Mahalanobis distance of a *cross-section*
    of asset returns from their historical mean and covariance -- a market-wide
    "how unusual is today" gauge that spikes during crashes. For a single series
    the covariance collapses to the variance, so turbulence reduces to the
    squared z-score of the latest return relative to its trailing distribution::

        turbulence_t = ((r_t - mu_t) / sigma_t) ** 2

    where ``mu_t`` and ``sigma_t`` are the mean and standard deviation of the
    ``lookback`` returns *strictly before* ``t`` (causal, no look-ahead). Large
    values flag an unusually large move in *either* direction -- the crash /
    melt-up days that cluster with elevated volatility. Output is non-negative,
    the same length as ``close``; warm-up bars without a full trailing window are
    ``0`` (never flagged as turbulent).
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    turbulence = np.zeros(n, dtype=float)
    if lookback < 2 or n < lookback + 2:
        return turbulence

    returns = np.zeros(n, dtype=float)
    returns[1:] = np.diff(close) / np.maximum(np.abs(close[:-1]), EPS)

    # Start at lookback + 1 so the trailing window excludes the returns[0]
    # placeholder and only ever looks at real, prior returns.
    for t in range(lookback + 1, n):
        window = returns[t - lookback : t]
        sigma = float(np.std(window))
        if sigma < EPS:
            continue
        z = (returns[t] - float(np.mean(window))) / sigma
        turbulence[t] = z * z
    return turbulence


def compute_win_rate(trade_returns: np.ndarray) -> float:
    """Fraction of trades with strictly positive return."""
    trade_returns = np.asarray(trade_returns, dtype=float)
    if trade_returns.size == 0:
        return 0.0
    return float(np.sum(trade_returns > 0) / trade_returns.size)


def compute_turnover(position_history: np.ndarray) -> float:
    """Average absolute position change per step (proxy for turnover)."""
    positions = np.asarray(position_history, dtype=float)
    if positions.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(positions))))


@dataclass
class IncrementalSharpe:
    """Online Differential Sharpe Ratio (Moody & Saffell, 2001).

    Updates a running mean (``a``) and second moment (``b``) of returns and
    returns the marginal contribution of the latest return to the Sharpe
    ratio. Suitable as a per-step reward signal.
    """

    eta: float = 0.01
    a: float = 0.0
    b: float = 0.0
    initialised: bool = field(default=False, repr=False)

    def update(self, return_t: float) -> float:
        if not self.initialised:
            self.a = return_t
            self.b = return_t * return_t
            self.initialised = True
            return 0.0

        delta_a = return_t - self.a
        delta_b = return_t * return_t - self.b
        variance = self.b - self.a * self.a
        if variance < EPS:
            dsr = 0.0
        else:
            dsr = (self.b * delta_a - 0.5 * self.a * delta_b) / (variance ** 1.5)

        self.a += self.eta * delta_a
        self.b += self.eta * delta_b
        return float(dsr)

    def reset(self) -> None:
        self.a = 0.0
        self.b = 0.0
        self.initialised = False
