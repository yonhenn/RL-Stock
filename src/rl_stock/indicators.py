"""Technical indicators used as trading environment features."""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate the relative strength index."""
    prices = np.asarray(prices, dtype=float)
    rsi = np.full(len(prices), 50.0, dtype=float)
    if len(prices) <= period:
        return rsi

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gains = np.zeros(len(prices), dtype=float)
    avg_losses = np.zeros(len(prices), dtype=float)
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])

    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i - 1]) / period

    rs = np.where(avg_losses != 0, avg_gains / avg_losses, 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0
    return rsi


def compute_macd(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate MACD line, signal line, and histogram."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    def ema(data: np.ndarray, period: int) -> np.ndarray:
        result = np.zeros(len(data), dtype=float)
        result[0] = data[0]
        multiplier = 2.0 / (period + 1.0)
        for i in range(1, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    prices: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands using the notebook's centered SMA logic."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    sma = historical_moving_average(prices, period)
    rolling_std = np.array(
        [
            np.std(prices[max(0, i - period + 1) : i + 1])
            if i >= period - 1
            else np.std(prices[: i + 1])
            for i in range(len(prices))
        ],
        dtype=float,
    )
    upper_band = sma + std_dev * rolling_std
    lower_band = sma - std_dev * rolling_std
    return upper_band, sma, lower_band


def historical_moving_average(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate a rolling mean using only current and previous rows."""
    data = np.asarray(data, dtype=float)
    ma = np.zeros(len(data), dtype=float)
    for i in range(len(data)):
        start_idx = max(0, i - period + 1)
        ma[i] = np.mean(data[start_idx : i + 1])
    return ma


def compute_sma_slope(
    prices: np.ndarray,
    period: int = 20,
    slope_window: int = 5,
) -> np.ndarray:
    """Scale-invariant per-day slope of the trailing ``period``-day SMA.

    Trend *direction*: positive when the moving average is rising, negative
    when falling. Dividing by the SMA level makes the value independent of the
    absolute price (so it behaves the same on a 3,000-point index as on a
    30-point stock), and dividing by ``slope_window`` turns it into a true
    per-day rate of change. Output has the same length as ``prices``; the first
    ``slope_window`` entries are zero (no prior SMA to compare against).
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    slope = np.zeros(n, dtype=float)
    if n <= slope_window:
        return slope

    sma = historical_moving_average(prices, period)
    delta = sma[slope_window:] - sma[:-slope_window]
    denom = slope_window * np.maximum(np.abs(sma[slope_window:]), EPS)
    slope[slope_window:] = delta / denom
    return slope


def compute_bollinger_bandwidth(
    prices: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> np.ndarray:
    """Normalized Bollinger Band width = ``(upper - lower) / sma``.

    Trend *strength* / volatility regime: a dimensionless, scale-invariant
    measure that widens in volatile markets and contracts in quiet ones. Always
    non-negative. Same length as ``prices``.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) == 0:
        return np.array([], dtype=float)

    upper, sma, lower = compute_bollinger_bands(prices, period, std_dev)
    return (upper - lower) / np.maximum(np.abs(sma), EPS)


def compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average Directional Index (Wilder). Raw output in ``[0, 100]``.

    Trend *strength*, independent of direction: high values indicate a strong
    trend (up or down), low values a range-bound market. Uses Wilder's
    smoothing for the true range, directional movement, and the final ADX line.
    Warm-up entries ``[0 : 2*period - 1]`` are zero (no look-ahead back-fill).
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = len(close)
    adx = np.zeros(n, dtype=float)
    if n <= 2 * period:
        return adx

    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    up_move = high - np.concatenate(([high[0]], high[:-1]))
    down_move = np.concatenate(([low[0]], low[:-1])) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    tr[0] = plus_dm[0] = minus_dm[0] = 0.0  # no prior bar at index 0

    atr_s = np.zeros(n, dtype=float)
    plus_s = np.zeros(n, dtype=float)
    minus_s = np.zeros(n, dtype=float)
    atr_s[period] = tr[1 : period + 1].sum()
    plus_s[period] = plus_dm[1 : period + 1].sum()
    minus_s[period] = minus_dm[1 : period + 1].sum()
    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
        plus_s[i] = plus_s[i - 1] - plus_s[i - 1] / period + plus_dm[i]
        minus_s[i] = minus_s[i - 1] - minus_s[i - 1] / period + minus_dm[i]

    plus_di = 100.0 * plus_s / np.maximum(atr_s, EPS)
    minus_di = 100.0 * minus_s / np.maximum(atr_s, EPS)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, EPS)

    seed = 2 * period - 1
    adx[seed] = float(np.mean(dx[period : seed + 1]))
    for i in range(seed + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


def compute_macd_cross(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> np.ndarray:
    """MACD golden/death-cross regime: ``+1`` bullish, ``-1`` bearish, ``0`` warm-up.

    A rule-based trading prior derived from the MACD line versus its signal
    line. ``sign(macd_line - signal_line)`` stays ``+1`` after a golden cross
    until the next death cross flips it to ``-1`` (and vice-versa), giving a
    dense regime feature distinct from the continuous MACD-histogram magnitude.
    The first ``slow`` entries are zeroed while the EMAs settle.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    cross = np.zeros(n, dtype=float)
    if n == 0:
        return cross

    macd_line, signal_line, _ = compute_macd(prices, fast, slow, signal)
    cross = np.sign(macd_line - signal_line).astype(float)
    cross[: min(slow, n)] = 0.0
    return cross
