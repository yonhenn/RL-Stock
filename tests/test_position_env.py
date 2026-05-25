"""Tests for the PositionMgmtEnv continuous-action environment."""

from __future__ import annotations

import numpy as np
import pytest

from rl_stock.envs_v2 import N_PORTFOLIO_FEATURES, N_TECH_FEATURES, PositionMgmtEnv, RiskConfig


@pytest.fixture
def env(synthetic_ohlc):
    return PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=RiskConfig(initial_cash=1_000_000.0),
    )


@pytest.mark.unit
def test_action_and_observation_spaces(env):
    assert env.action_space.shape == (1,)
    assert env.action_space.low == -1.0
    assert env.action_space.high == 1.0
    assert env.observation_space.shape == (20, N_TECH_FEATURES + N_PORTFOLIO_FEATURES)


@pytest.mark.unit
def test_reset_returns_observation_and_info(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert info["equity"] == pytest.approx(env.risk.initial_cash)
    assert info["position_ratio"] == 0.0


@pytest.mark.unit
def test_step_returns_finite_reward(env):
    env.reset(seed=0)
    obs, reward, term, trunc, info = env.step(np.array([0.5], dtype=np.float32))
    assert np.isfinite(reward)
    assert obs.shape == env.observation_space.shape
    assert not term


@pytest.mark.unit
def test_stop_loss_flattens_position(synthetic_ohlc):
    env = PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=RiskConfig(stop_loss_pct=0.001, initial_cash=1_000_000.0),
    )
    env.reset(seed=0)
    env.step(np.array([1.0], dtype=np.float32))
    # After several steps with tight stop loss, position must hit zero at least once.
    stopped_out = False
    for _ in range(40):
        _, _, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
        if info.get("stop_triggered"):
            stopped_out = True
            assert info["position_ratio"] == 0.0
            break
        if term or trunc:
            break
    assert stopped_out, "Tight stop-loss should trigger at least once on noisy data."


@pytest.mark.unit
def test_episode_terminates_at_frame_end(synthetic_ohlc):
    env = PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, 60),
    )
    env.reset(seed=0)
    truncated = False
    for _ in range(200):
        _, _, term, trunc, _ = env.step(np.zeros(1, dtype=np.float32))
        if term or trunc:
            truncated = True
            break
    assert truncated


@pytest.mark.unit
def test_invalid_frame_bound_raises(synthetic_ohlc):
    with pytest.raises(ValueError):
        PositionMgmtEnv(df=synthetic_ohlc, window_size=20, frame_bound=(5, 50))
    with pytest.raises(ValueError):
        PositionMgmtEnv(df=synthetic_ohlc, window_size=20, frame_bound=(20, 20))


@pytest.mark.unit
def test_info_dict_contains_history(env):
    env.reset(seed=0)
    for _ in range(5):
        _, _, term, trunc, info = env.step(np.array([0.3], dtype=np.float32))
        if term or trunc:
            break
    assert "equity_curve" in info
    assert "position_history" in info
    assert len(info["equity_curve"]) >= 2
    assert len(info["position_history"]) >= 2
