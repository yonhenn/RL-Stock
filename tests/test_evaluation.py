"""Tests for evaluation helpers (buy-and-hold benchmark)."""

from __future__ import annotations

import numpy as np
import pytest

from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.evaluation import buy_and_hold_result


@pytest.mark.unit
def test_buy_and_hold_matches_price_ratio(trending_ohlc):
    """The benchmark profit equals the raw price ratio over the traded window."""
    env = PositionMgmtEnv(
        df=trending_ohlc,
        window_size=20,
        frame_bound=(60, len(trending_ohlc)),
        risk_config=RiskConfig(),
    )
    result = buy_and_hold_result(env, num_episodes=3)

    prices = np.asarray(env.prices, dtype=float)
    start, end = env.window_size, env._end_tick
    expected_profit = float(prices[end] / prices[start])

    assert result["profits"] == pytest.approx([expected_profit] * 3)
    info = result["infos"][0]
    assert info["total_profit"] == pytest.approx(expected_profit)
    assert all(p == 1.0 for p in info["position_history"])  # fully invested
    assert len(info["equity_curve"]) == end - start + 1


@pytest.mark.unit
def test_buy_and_hold_positive_on_uptrend(trending_ohlc):
    """On the strongly uptrending fixture, holding the index makes money."""
    env = PositionMgmtEnv(
        df=trending_ohlc,
        window_size=20,
        frame_bound=(60, len(trending_ohlc)),
        risk_config=RiskConfig(),
    )
    result = buy_and_hold_result(env)
    assert result["profits"][0] > 1.0
