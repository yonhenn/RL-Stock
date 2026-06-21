"""Explicit trend judgment: turn raw indicators into a market-regime decision.

The :mod:`rl_stock.indicators` module exposes the raw *ingredients* of a trend
(SMA slope, ADX, MACD cross, ...). This module fuses them into an explicit,
interpretable **judgment**:

* :func:`compute_trend_score` -- a continuous, signed trend strength in
  ``[-1, 1]`` (``direction x strength``); a smooth feature for the agent.
* :func:`classify_trend_regime` -- a discrete regime label in
  ``{-2, -1, 0, +1, +2}`` (the actual "trend judgment"); a human-readable state.
* :func:`regime_to_base_position` -- map a regime to a trend-following base
  position; the anchor the RL agent tilts around (residual policy).

All functions are pure, vectorized, causal (no look-ahead) and scale-invariant.
Warm-up bars where the underlying ADX has not settled collapse to a neutral
``trend_score == 0`` / ``SIDEWAYS`` regime.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from rl_stock.indicators import (
    compute_adx,
    compute_macd_cross,
    compute_sma_slope,
    historical_moving_average,
)


class TrendRegime(IntEnum):
    """Discrete market-trend judgment.

    The integer values are deliberately ordinal and symmetric so that
    ``regime / 2`` maps linearly onto a signed base exposure in ``[-1, 1]``.
    """

    STRONG_DOWN = -2
    DOWN = -1
    SIDEWAYS = 0
    UP = 1
    STRONG_UP = 2


def _direction_consensus(
    close: np.ndarray,
    sma_period: int = 20,
    slope_window: int = 5,
) -> np.ndarray:
    """Signed directional consensus in ``[-1, 1]`` (ignores trend *strength*).

    The mean of three independent, scale-invariant sign votes:

    * the sign of the trailing SMA slope (is the average rising or falling?),
    * the MACD golden/death-cross regime (``+1`` / ``-1``),
    * whether price sits above or below its trailing SMA.

    Returns ``+1`` only when all three agree on an up-move, ``-1`` when all
    agree on a down-move, and intermediate values when they disagree.
    """
    close = np.asarray(close, dtype=float)
    slope_sign = np.sign(compute_sma_slope(close, sma_period, slope_window))
    macd_cross = compute_macd_cross(close)
    ma = historical_moving_average(close, sma_period)
    price_sign = np.sign(close - ma)
    return (slope_sign + macd_cross + price_sign) / 3.0


def compute_trend_score(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    sma_period: int = 20,
    slope_window: int = 5,
    adx_period: int = 14,
    adx_strong: float = 40.0,
) -> np.ndarray:
    """Continuous signed trend strength in ``[-1, 1]``.

    ``trend_score = direction x strength`` where ``direction`` is the
    :func:`_direction_consensus` and ``strength`` is the ADX normalized by
    ``adx_strong`` and clipped to ``[0, 1]``. Positive means a (strong) uptrend,
    negative a downtrend, near-zero a weak or range-bound market. Same length as
    ``close``; warm-up bars (ADX still zero) are ``0``.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    if close.size == 0:
        return np.array([], dtype=float)

    direction = _direction_consensus(close, sma_period, slope_window)
    adx = compute_adx(high, low, close, adx_period)
    strength = np.clip(adx / max(adx_strong, 1e-8), 0.0, 1.0)
    return np.clip(direction * strength, -1.0, 1.0)


def classify_trend_regime(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    sma_period: int = 20,
    slope_window: int = 5,
    adx_period: int = 14,
    adx_trend: float = 20.0,
    adx_strong: float = 40.0,
) -> np.ndarray:
    """Explicit discrete trend judgment, one :class:`TrendRegime` per bar.

    Strength comes from the ADX, direction from the :func:`_direction_consensus`:

    * ``ADX < adx_trend``                  -> ``SIDEWAYS`` (no tradable trend),
    * ``adx_trend <= ADX < adx_strong``    -> ``UP`` / ``DOWN`` by direction,
    * ``ADX >= adx_strong``                -> ``STRONG_UP`` / ``STRONG_DOWN``.

    Conflicting direction votes (consensus exactly ``0``) stay ``SIDEWAYS``.
    Returns an ``int`` array the same length as ``close``; warm-up bars are
    ``SIDEWAYS`` (ADX is zero there, so never classified as trending).
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    n = close.size
    regime = np.zeros(n, dtype=int)
    if n == 0:
        return regime

    adx = compute_adx(high, low, close, adx_period)
    direction = np.sign(_direction_consensus(close, sma_period, slope_window))
    trending = adx >= adx_trend
    strong = adx >= adx_strong

    up = trending & (direction > 0)
    down = trending & (direction < 0)
    regime[up] = int(TrendRegime.UP)
    regime[up & strong] = int(TrendRegime.STRONG_UP)
    regime[down] = int(TrendRegime.DOWN)
    regime[down & strong] = int(TrendRegime.STRONG_DOWN)
    return regime


def regime_to_base_position(
    regime: np.ndarray,
    base_exposure: float = 0.8,
) -> np.ndarray:
    """Map a discrete regime to a trend-following base position in ``[-1, 1]``.

    Linear and symmetric: ``regime / 2 * base_exposure`` so the magnitudes are
    ``{0, base/2, base}``. This is the anchor a residual-policy agent tilts
    around -- ``STRONG_UP`` anchors a large long, ``SIDEWAYS`` anchors flat,
    ``STRONG_DOWN`` a large short.
    """
    regime = np.asarray(regime, dtype=float)
    base = regime / 2.0 * base_exposure
    return np.clip(base, -abs(base_exposure), abs(base_exposure))
