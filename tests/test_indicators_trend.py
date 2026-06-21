"""Tests for the trend-aware technical indicators."""

from __future__ import annotations

import numpy as np
import pytest

from rl_stock.indicators import (
    compute_adx,
    compute_bollinger_bandwidth,
    compute_macd_cross,
    compute_sma_slope,
)


# --------------------------------------------------------------------------- #
# compute_sma_slope
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_sma_slope_length_and_warmup():
    prices = np.linspace(100.0, 200.0, 120)
    slope = compute_sma_slope(prices, period=20, slope_window=5)
    assert slope.shape == prices.shape
    assert np.all(slope[:5] == 0.0)
    assert np.all(np.isfinite(slope))


@pytest.mark.unit
def test_sma_slope_sign_tracks_direction():
    up = np.linspace(100.0, 200.0, 120)
    down = np.linspace(200.0, 100.0, 120)
    slope_up = compute_sma_slope(up)
    slope_down = compute_sma_slope(down)
    # After both the SMA and slope warm-ups, direction must be unambiguous.
    assert np.all(slope_up[30:] > 0.0)
    assert np.all(slope_down[30:] < 0.0)


@pytest.mark.unit
def test_sma_slope_is_scale_invariant():
    prices = 100.0 + np.cumsum(np.sin(np.linspace(0, 12, 200)))
    slope_a = compute_sma_slope(prices)
    slope_b = compute_sma_slope(prices * 1000.0)
    assert np.allclose(slope_a, slope_b, atol=1e-9)


# --------------------------------------------------------------------------- #
# compute_bollinger_bandwidth
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_bandwidth_length_and_nonnegative():
    rng = np.random.default_rng(0)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, 150))
    bw = compute_bollinger_bandwidth(prices, period=20)
    assert bw.shape == prices.shape
    assert np.all(bw >= 0.0)
    assert np.all(np.isfinite(bw))


@pytest.mark.unit
def test_bandwidth_wider_for_volatile_series():
    rng = np.random.default_rng(1)
    calm = 100.0 + rng.normal(0, 0.2, 200)
    volatile = 100.0 + rng.normal(0, 8.0, 200)
    bw_calm = compute_bollinger_bandwidth(calm, period=20)
    bw_vol = compute_bollinger_bandwidth(volatile, period=20)
    assert bw_vol[20:].mean() > bw_calm[20:].mean()


# --------------------------------------------------------------------------- #
# compute_adx
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_adx_bounds_and_warmup():
    rng = np.random.default_rng(2)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 120))
    high = close + 1.0
    low = close - 1.0
    adx = compute_adx(high, low, close, period=14)
    assert adx.shape == close.shape
    assert np.all((adx >= 0.0) & (adx <= 100.0))
    assert np.all(adx[: 2 * 14 - 1] == 0.0)
    assert np.all(np.isfinite(adx))


@pytest.mark.unit
def test_adx_higher_on_trend_than_chop():
    n = 150
    trend_close = np.linspace(100.0, 200.0, n)
    trend = compute_adx(trend_close + 1.0, trend_close - 1.0, trend_close, period=14)

    chop_close = 100.0 + 3.0 * np.sin(np.linspace(0, 30, n))
    chop = compute_adx(chop_close + 1.0, chop_close - 1.0, chop_close, period=14)

    assert trend[40:].mean() > chop[40:].mean()


# --------------------------------------------------------------------------- #
# compute_macd_cross
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_macd_cross_values_and_warmup():
    rng = np.random.default_rng(3)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    cross = compute_macd_cross(prices)
    assert cross.shape == prices.shape
    assert np.all(np.isin(cross, (-1.0, 0.0, 1.0)))
    assert np.all(cross[:26] == 0.0)


@pytest.mark.unit
def test_macd_cross_sign_tracks_trend():
    up = np.linspace(100.0, 300.0, 200)
    down = np.linspace(300.0, 100.0, 200)
    assert compute_macd_cross(up)[40:].mean() > 0.0
    assert compute_macd_cross(down)[40:].mean() < 0.0
