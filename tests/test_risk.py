"""Unit tests for risk utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from rl_stock.risk import (
    IncrementalSharpe,
    compute_atr,
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_turnover,
    compute_win_rate,
)


@pytest.mark.unit
def test_sharpe_zero_for_empty_or_constant_returns():
    assert compute_sharpe_ratio(np.array([])) == 0.0
    assert compute_sharpe_ratio(np.zeros(10)) == 0.0


@pytest.mark.unit
def test_sharpe_positive_for_consistent_gains():
    returns = np.array([0.01, 0.012, 0.009, 0.011, 0.013, 0.008])
    assert compute_sharpe_ratio(returns) > 0.0


@pytest.mark.unit
def test_sortino_handles_no_downside():
    returns = np.array([0.01, 0.02, 0.03])
    assert compute_sortino_ratio(returns) == 0.0


@pytest.mark.unit
def test_sortino_positive_when_mostly_gains():
    returns = np.array([0.02, 0.03, -0.005, 0.015, 0.01])
    assert compute_sortino_ratio(returns) > 0.0


@pytest.mark.unit
def test_max_drawdown_known_curve():
    equity = np.array([100, 110, 120, 80, 95, 90, 130])
    # peak 120 -> trough 80 = 33.3% drawdown
    assert compute_max_drawdown(equity) == pytest.approx(40 / 120, rel=1e-6)


@pytest.mark.unit
def test_calmar_zero_when_no_drawdown():
    equity = np.array([100, 101, 102, 103])
    returns = np.diff(equity) / equity[:-1]
    assert compute_calmar_ratio(returns, equity) == 0.0


@pytest.mark.unit
def test_atr_length_matches_input():
    n = 50
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    atr = compute_atr(high, low, close, period=14)
    assert atr.shape == (n,)
    assert np.all(atr >= 0)


@pytest.mark.unit
def test_win_rate_and_turnover():
    trades = np.array([0.02, -0.01, 0.03, -0.005])
    assert compute_win_rate(trades) == pytest.approx(0.5)

    positions = np.array([0.0, 0.5, 0.5, -0.5])
    assert compute_turnover(positions) == pytest.approx((0.5 + 0.0 + 1.0) / 3)


@pytest.mark.unit
def test_incremental_sharpe_runs_and_resets():
    dsr = IncrementalSharpe(eta=0.05)
    rewards = [dsr.update(r) for r in [0.01, 0.012, -0.005, 0.02, 0.015]]
    assert all(np.isfinite(rewards))
    dsr.reset()
    assert dsr.a == 0.0 and dsr.b == 0.0 and not dsr.initialised
