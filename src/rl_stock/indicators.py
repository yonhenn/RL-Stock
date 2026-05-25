"""Technical indicators used as trading environment features."""

from __future__ import annotations

import numpy as np


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

    sma = np.convolve(prices, np.ones(period) / period, mode="same")
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
