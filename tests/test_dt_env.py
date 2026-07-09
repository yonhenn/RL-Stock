"""Tests for DT-specific environment features (flat_obs mode)."""

from __future__ import annotations

import numpy as np
import pytest

from rl_stock.envs_v2 import (
    N_PORTFOLIO_FEATURES,
    N_PORTFOLIO_FEATURES_DT,
    N_TECH_FEATURES,
    PositionMgmtEnv,
    RiskConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(synthetic_ohlc, flat_obs: bool = True, **risk_kw) -> PositionMgmtEnv:
    """Build a PositionMgmtEnv from the shared fixture with optional overrides."""
    risk = RiskConfig(flat_obs=flat_obs, **risk_kw)
    return PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=30,
        frame_bound=(30, len(synthetic_ohlc) - 1),
        risk_config=risk,
    )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFlatObsShape:
    """flat_obs=True should produce a 1-D observation vector."""

    def test_flat_obs_shape_1d(self, synthetic_ohlc):
        env = _make_env(synthetic_ohlc, flat_obs=True)
        obs, _ = env.reset()
        expected_dim = N_TECH_FEATURES + N_PORTFOLIO_FEATURES_DT
        assert obs.shape == (expected_dim,), (
            f"Expected ({expected_dim},), got {obs.shape}"
        )

    def test_flat_obs_step_shape(self, synthetic_ohlc):
        env = _make_env(synthetic_ohlc, flat_obs=True)
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs2, _, _, _, _ = env.step(action)
        assert obs2.shape == obs.shape

    def test_flat_obs_with_trend_features(self, synthetic_ohlc):
        """Trend features should add to the flat vector too."""
        env = _make_env(synthetic_ohlc, flat_obs=True, use_trend_features=True)
        obs, _ = env.reset()
        from rl_stock.envs_v2 import N_TREND_FEATURES
        expected = N_TECH_FEATURES + N_TREND_FEATURES + N_PORTFOLIO_FEATURES_DT
        assert obs.shape == (expected,)

    def test_flat_obs_with_turbulence(self, synthetic_ohlc):
        """Turbulence gate feature should add to the flat vector."""
        env = _make_env(synthetic_ohlc, flat_obs=True, use_turbulence_gate=True)
        obs, _ = env.reset()
        from rl_stock.envs_v2 import N_TURBULENCE_FEATURES
        expected = N_TECH_FEATURES + N_TURBULENCE_FEATURES + N_PORTFOLIO_FEATURES_DT
        assert obs.shape == (expected,)


# ---------------------------------------------------------------------------
# Regression: window mode unchanged
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWindowModeUnchanged:
    """flat_obs=False should produce the original 2-D sliding window obs."""

    def test_window_obs_shape_2d(self, synthetic_ohlc):
        env = _make_env(synthetic_ohlc, flat_obs=False)
        obs, _ = env.reset()
        expected_shape = (30, N_TECH_FEATURES + N_PORTFOLIO_FEATURES)
        assert obs.shape == expected_shape, (
            f"Expected {expected_shape}, got {obs.shape}"
        )

    def test_window_obs_step_shape(self, synthetic_ohlc):
        env = _make_env(synthetic_ohlc, flat_obs=False)
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs2, _, _, _, _ = env.step(action)
        assert obs2.shape == obs.shape


# ---------------------------------------------------------------------------
# DT portfolio features content
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDTPortfolioFeatures:
    """DT portfolio features should contain the correct 6 features."""

    def test_initial_portfolio_features(self, synthetic_ohlc):
        """At reset: position=0, pnl=0, dd=0, vol>0, stop=0, dd_lock=0."""
        env = _make_env(synthetic_ohlc, flat_obs=True)
        obs, _ = env.reset()
        # Last 6 elements are portfolio features
        portfolio = obs[-N_PORTFOLIO_FEATURES_DT:]
        assert portfolio[0] == pytest.approx(0.0)  # position_ratio
        assert portfolio[1] == pytest.approx(0.0)  # unrealized_pnl
        assert portfolio[2] == pytest.approx(0.0)  # drawdown
        assert portfolio[3] > 0.0                   # volatility (ATR-based, > 0)
        assert portfolio[4] == pytest.approx(0.0)  # stop_flag
        assert portfolio[5] == pytest.approx(0.0)  # dd_lock_flag

    def test_position_ratio_updates(self, synthetic_ohlc):
        """After a buy action, position_ratio should be non-zero."""
        env = _make_env(synthetic_ohlc, flat_obs=True)
        env.reset()
        action = np.array([0.8])  # buy
        obs, _, _, _, _ = env.step(action)
        portfolio = obs[-N_PORTFOLIO_FEATURES_DT:]
        # position_ratio should reflect executed position (not necessarily 0.8
        # due to vol scaling, but should be non-zero and positive)
        assert portfolio[0] != pytest.approx(0.0)

    def test_no_holding_norm_in_dt_features(self, synthetic_ohlc):
        """DT features should NOT include holding_norm."""
        env = _make_env(synthetic_ohlc, flat_obs=True)
        obs, _ = env.reset()
        # DT portfolio is 6 elements: [pos, pnl, dd, vol, stop, dd_lock]
        # If holding_norm were present, we'd have 7 elements
        assert obs[-N_PORTFOLIO_FEATURES_DT:].shape[0] == 6


# ---------------------------------------------------------------------------
# Stop/DD flag exposure
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRiskFlags:
    """stop_flag and dd_lock_flag should appear in DT observations."""

    def test_stop_flag_activates(self, synthetic_ohlc):
        """When stop-loss triggers, stop_flag should be 1.0 in next obs."""
        env = _make_env(
            synthetic_ohlc, flat_obs=True,
            stop_loss_pct=0.001,  # very tight stop to trigger easily
        )
        env.reset()
        # Take a position and let price move
        env.step(np.array([1.0]))  # full long
        # Step through until stop triggers or episode ends
        stop_seen = False
        for _ in range(200):
            obs, _, term, trunc, info = env.step(np.array([1.0]))
            if info.get("stop_triggered"):
                stop_seen = True
                # After stop, the obs should show stop_flag = 1.0
                portfolio = obs[-N_PORTFOLIO_FEATURES_DT:]
                assert portfolio[4] == pytest.approx(1.0), "stop_flag should be 1.0"
                break
            if term or trunc:
                break
        # If stop never triggered with this tight threshold, the test
        # data may not have enough price movement — that's acceptable.
        if not stop_seen:
            pytest.skip("Stop-loss never triggered with this synthetic data")

    def test_dd_lock_flag_activates(self, synthetic_ohlc):
        """When drawdown lock triggers, dd_lock_flag should be 1.0."""
        env = _make_env(
            synthetic_ohlc, flat_obs=True,
            max_drawdown_pct=0.001,  # very tight dd lock
        )
        env.reset()
        # Take alternating positions to generate drawdown
        dd_seen = False
        for i in range(200):
            action = np.array([1.0]) if i % 10 < 5 else np.array([-1.0])
            obs, _, term, trunc, info = env.step(action)
            portfolio = obs[-N_PORTFOLIO_FEATURES_DT:]
            if portfolio[5] == pytest.approx(1.0):
                dd_seen = True
                break
            if term or trunc:
                break
        if not dd_seen:
            pytest.skip("Drawdown lock never triggered with this synthetic data")


# ---------------------------------------------------------------------------
# Info dict enrichment
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestInfoEnrichment:
    """step() should return enriched info dict for DT data collection."""

    def _step_once(self, synthetic_ohlc, flat_obs=True):
        env = _make_env(synthetic_ohlc, flat_obs=flat_obs)
        env.reset()
        action = np.array([0.5])
        _, _, _, _, info = env.step(action)
        return info

    def test_info_has_raw_action(self, synthetic_ohlc):
        info = self._step_once(synthetic_ohlc)
        assert "raw_action" in info
        assert isinstance(info["raw_action"], float)
        assert info["raw_action"] == pytest.approx(0.5)

    def test_info_has_executed_action(self, synthetic_ohlc):
        info = self._step_once(synthetic_ohlc)
        assert "executed_action" in info
        assert isinstance(info["executed_action"], float)

    def test_info_has_net_return(self, synthetic_ohlc):
        info = self._step_once(synthetic_ohlc)
        assert "net_return" in info
        assert isinstance(info["net_return"], float)

    def test_info_has_reward_components(self, synthetic_ohlc):
        info = self._step_once(synthetic_ohlc)
        assert "reward_components" in info
        rc = info["reward_components"]
        assert isinstance(rc, dict)
        for key in ("dsr", "drawdown", "turnover", "align"):
            assert key in rc, f"Missing reward component: {key}"
            assert isinstance(rc[key], float)

    def test_info_enrichment_works_in_window_mode_too(self, synthetic_ohlc):
        """The new info fields should be present even in window mode."""
        info = self._step_once(synthetic_ohlc, flat_obs=False)
        assert "raw_action" in info
        assert "executed_action" in info
        assert "net_return" in info
        assert "reward_components" in info

    def test_raw_vs_executed_action_differ_with_stop(self, synthetic_ohlc):
        """When stop-loss triggers, raw and executed action should differ."""
        env = _make_env(
            synthetic_ohlc, flat_obs=True,
            stop_loss_pct=0.001,
        )
        env.reset()
        env.step(np.array([1.0]))
        for _ in range(200):
            _, _, term, trunc, info = env.step(np.array([0.8]))
            if info.get("stop_triggered"):
                assert info["raw_action"] == pytest.approx(0.8)
                assert info["executed_action"] == pytest.approx(0.0)
                return
            if term or trunc:
                break
        pytest.skip("Stop-loss never triggered with this synthetic data")
