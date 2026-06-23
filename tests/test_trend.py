"""Tests for the explicit trend-judgment module."""

from __future__ import annotations

import numpy as np
import pytest

from rl_stock.trend import (
    TrendRegime,
    classify_trend_regime,
    compute_trend_score,
    regime_to_base_position,
)


# --------------------------------------------------------------------------- #
# compute_trend_score
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_trend_score_shape_warmup_and_bounds():
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 150))
    score = compute_trend_score(close + 1, close - 1, close)
    assert score.shape == close.shape
    assert np.all((score >= -1.0) & (score <= 1.0))
    assert np.all(np.isfinite(score))
    # ADX has no signal during its warm-up window -> score must be exactly zero.
    assert np.all(score[: 2 * 14 - 1] == 0.0)


@pytest.mark.unit
def test_trend_score_sign_tracks_direction():
    up = np.linspace(100.0, 200.0, 150)
    down = np.linspace(200.0, 100.0, 150)
    up_score = compute_trend_score(up + 1, up - 1, up)
    down_score = compute_trend_score(down + 1, down - 1, down)
    assert up_score[40:].mean() > 0.5
    assert down_score[40:].mean() < -0.5


@pytest.mark.unit
def test_trend_score_is_scale_invariant():
    rng = np.random.default_rng(1)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    a = compute_trend_score(close + 1, close - 1, close)
    b = compute_trend_score((close + 1) * 1000.0, (close - 1) * 1000.0, close * 1000.0)
    assert np.allclose(a, b, atol=1e-9)


@pytest.mark.unit
def test_trend_score_stronger_on_trend_than_chop():
    n = 200
    trend = np.linspace(100.0, 220.0, n)
    chop = 100.0 + 3.0 * np.sin(np.linspace(0, 30, n))
    trend_mag = np.abs(compute_trend_score(trend + 1, trend - 1, trend))[40:].mean()
    chop_mag = np.abs(compute_trend_score(chop + 1, chop - 1, chop))[40:].mean()
    assert trend_mag > chop_mag


# --------------------------------------------------------------------------- #
# classify_trend_regime
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_regime_values_and_warmup():
    rng = np.random.default_rng(2)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    regime = classify_trend_regime(close + 1, close - 1, close)
    assert regime.shape == close.shape
    assert np.all(np.isin(regime, [-2, -1, 0, 1, 2]))
    # ADX warm-up bars cannot be classified as trending.
    assert np.all(regime[: 2 * 14 - 1] == int(TrendRegime.SIDEWAYS))


@pytest.mark.unit
def test_regime_uptrend_reaches_strong_up():
    up = np.linspace(100.0, 260.0, 200)
    regime = classify_trend_regime(up + 1, up - 1, up)
    assert np.all(regime[50:] > 0)
    assert (regime == int(TrendRegime.STRONG_UP)).any()


@pytest.mark.unit
def test_regime_downtrend_reaches_strong_down():
    down = np.linspace(260.0, 100.0, 200)
    regime = classify_trend_regime(down + 1, down - 1, down)
    assert np.all(regime[50:] < 0)
    assert (regime == int(TrendRegime.STRONG_DOWN)).any()


@pytest.mark.unit
def test_regime_sideways_for_range_bound_market():
    # A genuinely range-bound market is *noise* with no persistent direction
    # (low ADX) -- a smooth sine, by contrast, is a sequence of mini-trends.
    rng = np.random.default_rng(11)
    n = 200
    chop = 100.0 + rng.normal(0, 1.5, n)
    regime = classify_trend_regime(chop + 0.5, chop - 0.5, chop)
    assert np.mean(regime[40:] == int(TrendRegime.SIDEWAYS)) > 0.6


# --------------------------------------------------------------------------- #
# regime_to_base_position
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_base_position_linear_mapping():
    regimes = np.array([-2, -1, 0, 1, 2])
    base = regime_to_base_position(regimes, base_exposure=0.8)
    np.testing.assert_allclose(base, [-0.8, -0.4, 0.0, 0.4, 0.8])


@pytest.mark.unit
def test_base_position_respects_exposure_bound():
    regimes = np.array([-2, -1, 0, 1, 2])
    base = regime_to_base_position(regimes, base_exposure=0.5)
    assert np.all(np.abs(base) <= 0.5 + 1e-9)
    assert base[-1] == pytest.approx(0.5)
    assert base[0] == pytest.approx(-0.5)
