"""Tests for the PositionMgmtEnv continuous-action environment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rl_stock.envs_v2 import (
    N_PORTFOLIO_FEATURES,
    N_TECH_FEATURES,
    N_TREND_FEATURES,
    N_TURBULENCE_FEATURES,
    PositionMgmtEnv,
    RiskConfig,
)
from rl_stock.trend import TrendRegime


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
    assert env.action_space.low == 0.0  # long-only: no short side
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
def test_stop_loss_disabled_when_non_positive(synthetic_ohlc):
    # stop_loss_pct <= 0 disables the hard stop: it must never flatten the book,
    # even on the same noisy data where a tight stop triggers.
    env = PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=RiskConfig(stop_loss_pct=0.0, initial_cash=1_000_000.0),
    )
    env.reset(seed=0)
    info: dict = {}
    for _ in range(60):
        _, _, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
        assert not info.get("stop_triggered"), "Stop must never trigger when disabled."
        if term or trunc:
            break
    assert info["stop_loss_events"] == 0


def _declining_ohlc(n: int = 140, total_drop: float = 0.35) -> pd.DataFrame:
    """Deterministic sustained downtrend: close falls ~``total_drop`` over n bars.

    A full-long agent on this series bleeds equity monotonically, so the
    peak-to-trough drawdown grows past any threshold -- the clean way to
    exercise (and to prove the disabling of) the equity drawdown lock.
    """
    rng = np.random.default_rng(5)
    ramp = np.linspace(0.0, -total_drop, n)
    close = 3000.0 * (1.0 + ramp) + rng.normal(0, 1.0, n)
    high = close + rng.uniform(1, 5, n)
    low = close - rng.uniform(1, 5, n)
    open_ = close + rng.normal(0, 1.0, n)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close},
        index=pd.Index(dates, name="Date"),
    )


@pytest.mark.unit
def test_hard_constraints_off_by_default():
    """Default RiskConfig ships both hard rails disabled (reward-driven risk)."""
    cfg = RiskConfig()
    assert cfg.stop_loss_pct == 0.0
    assert cfg.max_drawdown_pct == 0.0


@pytest.mark.unit
def test_drawdown_lock_triggers_on_decline():
    """A sustained decline past the threshold engages the drawdown lock."""
    df = _declining_ohlc()
    env = PositionMgmtEnv(
        df=df,
        window_size=20,
        frame_bound=(20, len(df)),
        # Stop-loss off so the decline isn't cut short before the lock engages.
        risk_config=RiskConfig(stop_loss_pct=0.0, max_drawdown_pct=0.08),
    )
    env.reset(seed=0)
    locked = False
    for _ in range(len(df)):
        _, _, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
        if info["dd_locked"]:
            locked = True
            break
        if term or trunc:
            break
    assert locked, "A drawdown past max_drawdown_pct should engage the lock."


@pytest.mark.unit
def test_drawdown_lock_disabled_when_non_positive():
    # max_drawdown_pct <= 0 disables the lock: it must never engage, even on a
    # deep sustained decline that easily breaches any positive threshold.
    df = _declining_ohlc()
    env = PositionMgmtEnv(
        df=df,
        window_size=20,
        frame_bound=(20, len(df)),
        risk_config=RiskConfig(stop_loss_pct=0.0, max_drawdown_pct=0.0),
    )
    env.reset(seed=0)
    for _ in range(len(df)):
        _, _, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
        assert not info["dd_locked"], "Lock must never engage when disabled."
        if term or trunc:
            break


# --------------------------------------------------------------------------- #
# Reward modes: return-driven (default) vs legacy scale-invariant DSR
# --------------------------------------------------------------------------- #
def _cumulative_reward(df, action: float, **risk_kwargs) -> float:
    """Sum the per-step reward for a constant action over a full episode."""
    env = PositionMgmtEnv(
        df=df,
        window_size=20,
        frame_bound=(60, len(df)),
        risk_config=RiskConfig(**risk_kwargs),
    )
    env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        _, reward, term, trunc, _ = env.step(np.array([action], dtype=np.float32))
        total += reward
        done = term or trunc
    return total


@pytest.mark.unit
def test_reward_mode_defaults_to_return():
    assert RiskConfig().reward_mode == "return"


@pytest.mark.unit
def test_return_reward_rewards_sizing_into_uptrend(trending_ohlc):
    """Return mode: on an uptrend, bigger long => strictly more reward.

    This is the property the DSR reward lacked -- it is what makes the agent
    want to participate in (and size up into) a profitable trend.
    """
    flat = _cumulative_reward(trending_ohlc, 0.0, reward_mode="return")
    half = _cumulative_reward(trending_ohlc, 0.5, reward_mode="return")
    full = _cumulative_reward(trending_ohlc, 1.0, reward_mode="return")
    assert full > half > flat
    assert flat == pytest.approx(0.0, abs=1e-6)  # flat earns (and risks) nothing


@pytest.mark.unit
def test_return_reward_tracks_realized_profit(trending_ohlc):
    """Return-mode cumulative reward moves with realized profit across sizes.

    The whole point of the return objective: more profit => more reward. (The
    legacy DSR objective is scale-invariant and breaks this on real noisy data,
    which is why the agents under-participated in the trend.)
    """
    profits, rewards = [], []
    for action in (0.25, 0.5, 1.0):
        env = PositionMgmtEnv(
            df=trending_ohlc,
            window_size=20,
            frame_bound=(60, len(trending_ohlc)),
            risk_config=RiskConfig(reward_mode="return"),
        )
        env.reset(seed=0)
        total, done, info = 0.0, False, {}
        while not done:
            _, reward, term, trunc, info = env.step(np.array([action], dtype=np.float32))
            total += reward
            done = term or trunc
        profits.append(info["total_profit"])
        rewards.append(total)
    # Both profit and reward are strictly increasing in position size.
    assert profits[0] < profits[1] < profits[2]
    assert rewards[0] < rewards[1] < rewards[2]


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


@pytest.mark.unit
def test_trend_feature_count_constant():
    assert N_TREND_FEATURES == 6


@pytest.fixture
def trend_env(synthetic_ohlc):
    return PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=RiskConfig(initial_cash=1_000_000.0, use_trend_features=True),
    )


@pytest.mark.unit
def test_trend_flag_expands_observation_space(trend_env):
    expected = N_TECH_FEATURES + N_TREND_FEATURES + N_PORTFOLIO_FEATURES
    assert trend_env.observation_space.shape == (20, expected)  # (20, 25)


@pytest.mark.unit
def test_trend_env_reset_and_step_shapes(trend_env):
    obs, _ = trend_env.reset(seed=0)
    assert obs.shape == trend_env.observation_space.shape
    obs, reward, term, _, _ = trend_env.step(np.array([0.5], dtype=np.float32))
    assert obs.shape == trend_env.observation_space.shape
    assert np.isfinite(reward)
    assert not term


@pytest.mark.unit
def test_trend_flag_preserves_base_columns(synthetic_ohlc):
    """Enabling trend features must not alter the original 13 tech columns."""
    base = PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=RiskConfig(use_trend_features=False),
    )
    trend = PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=RiskConfig(use_trend_features=True),
    )
    assert base.signal_features.shape[1] == N_TECH_FEATURES
    assert trend.signal_features.shape[1] == N_TECH_FEATURES + N_TREND_FEATURES
    np.testing.assert_array_equal(
        base.signal_features, trend.signal_features[:, :N_TECH_FEATURES]
    )


# --------------------------------------------------------------------------- #
# Trend-prior residual policy + regime-aware risk caps
# --------------------------------------------------------------------------- #
def _step_from_reset(env, action: float) -> dict:
    """Reset then take a single action; return the resulting info dict."""
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([action], dtype=np.float32))
    return info


@pytest.fixture
def prior_env(trending_ohlc):
    return PositionMgmtEnv(
        df=trending_ohlc,
        window_size=20,
        frame_bound=(60, len(trending_ohlc)),
        risk_config=RiskConfig(
            use_trend_prior=True,
            trend_base_exposure=0.8,
            residual_band=0.5,
            regime_risk_scaling=True,
        ),
    )


@pytest.mark.unit
def test_uptrend_resolves_to_positive_regime(prior_env):
    _, info = prior_env.reset(seed=0)
    assert info["regime"] >= 1.0  # UP or STRONG_UP
    assert info["trend_base"] > 0.0


@pytest.mark.unit
def test_trend_prior_anchors_zero_action(prior_env):
    """A zero residual on an uptrend anchors near the positive prior, not flat."""
    info = _step_from_reset(prior_env, 0.0)
    assert info["position_ratio"] > 0.4


@pytest.mark.unit
def test_residual_is_monotonic_in_action(prior_env):
    """Larger actions never produce a smaller long target around the trend anchor.

    Long-only: the action is in ``[0, 1]``; ``0`` anchors the base long and
    larger actions only add exposure.
    """
    base = _step_from_reset(prior_env, 0.0)["position_ratio"]
    mid = _step_from_reset(prior_env, 0.5)["position_ratio"]
    up = _step_from_reset(prior_env, 1.0)["position_ratio"]
    assert up >= mid >= base
    assert up > base  # the band actually tilts the position


@pytest.mark.unit
def test_regime_cap_blocks_counter_trend():
    """With caps on (prior off), a full long into a STRONG_DOWN regime is capped.

    Long-only: the surviving counter-trend risk is a large long held into a
    strong down-trend, so the cap clamps it to ``counter_trend_cap``.
    """
    df = _declining_ohlc()
    capped = PositionMgmtEnv(
        df=df,
        window_size=20,
        frame_bound=(60, len(df)),
        risk_config=RiskConfig(
            use_trend_prior=False, regime_risk_scaling=True, counter_trend_cap=0.2
        ),
    )
    uncapped = PositionMgmtEnv(
        df=df,
        window_size=20,
        frame_bound=(60, len(df)),
        risk_config=RiskConfig(use_trend_prior=False, regime_risk_scaling=False),
    )
    # Confirm the regime really is STRONG_DOWN, otherwise the cap is a no-op.
    _, info = capped.reset(seed=0)
    assert info["regime"] <= float(TrendRegime.STRONG_DOWN)
    # Full long order inside a STRONG_DOWN regime: the cap forbids large longs.
    capped_pos = _step_from_reset(capped, 1.0)["position_ratio"]
    uncapped_pos = _step_from_reset(uncapped, 1.0)["position_ratio"]
    assert capped_pos <= 0.2 + 1e-6
    assert uncapped_pos > capped_pos  # without the cap the long is much larger


@pytest.mark.unit
def test_default_target_passes_action_through(synthetic_ohlc):
    """Default config (vol scaling off): the target is just the clipped action."""
    env = PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=RiskConfig(),  # defaults: use_vol_scaling=False
    )
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))
    assert info["position_ratio"] == pytest.approx(0.5)


@pytest.mark.unit
def test_vol_scaling_opt_in_matches_scaled_target(synthetic_ohlc):
    """With vol scaling on, the target is exactly the vol-scaled action."""
    env = PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=RiskConfig(use_vol_scaling=True),
    )
    env.reset(seed=0)
    expected = env._volatility_scaled_target(0.5)  # same tick as the first step
    _, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))
    assert info["position_ratio"] == pytest.approx(expected)


@pytest.mark.unit
def test_vol_scaling_off_by_default():
    """Volatility scaling ships opt-in (off), so a 1.0 target isn't silently cut."""
    assert RiskConfig().use_vol_scaling is False


@pytest.mark.unit
def test_prior_env_step_finite_reward(prior_env):
    obs, _ = prior_env.reset(seed=0)
    assert obs.shape == prior_env.observation_space.shape
    _, reward, term, _, _ = prior_env.step(np.array([0.5], dtype=np.float32))
    assert np.isfinite(reward)
    assert not term


# --------------------------------------------------------------------------- #
# Turbulence circuit breaker (FinRL-style market-crash risk-off)
# --------------------------------------------------------------------------- #
def _shock_ohlc(n: int = 120, shock_idx: int = 60, shock: float = 0.12) -> pd.DataFrame:
    """Noisy calm OHLC with a single large return shock at ``shock_idx``."""
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0005, 0.005, n)
    returns[shock_idx] = shock
    close = 3000.0 * np.cumprod(1.0 + returns)
    high = close * 1.002
    low = close * 0.998
    open_ = close * (1.0 + rng.normal(0, 0.0005, n))
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close},
        index=pd.Index(dates, name="Date"),
    )


def _turbulence_config(**overrides) -> RiskConfig:
    params = dict(
        use_turbulence_gate=True,
        turbulence_lookback=20,
        turbulence_threshold=9.0,
        turbulence_exposure_cap=0.0,
    )
    params.update(overrides)
    return RiskConfig(**params)


@pytest.mark.unit
def test_turbulence_feature_count_constant():
    assert N_TURBULENCE_FEATURES == 1


@pytest.mark.unit
def test_turbulence_flag_expands_observation_space(synthetic_ohlc):
    env = PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=_turbulence_config(turbulence_lookback=30),
    )
    expected = N_TECH_FEATURES + N_TURBULENCE_FEATURES + N_PORTFOLIO_FEATURES
    assert env.observation_space.shape == (20, expected)


@pytest.mark.unit
def test_turbulence_flag_preserves_base_columns(synthetic_ohlc):
    """Enabling the gate appends one column without altering the base 13."""
    base = PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=RiskConfig(),
    )
    gated = PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=_turbulence_config(turbulence_lookback=30),
    )
    assert gated.signal_features.shape[1] == N_TECH_FEATURES + N_TURBULENCE_FEATURES
    np.testing.assert_array_equal(
        base.signal_features, gated.signal_features[:, :N_TECH_FEATURES]
    )


@pytest.mark.unit
def test_turbulence_info_keys_present(synthetic_ohlc):
    env = PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=_turbulence_config(turbulence_lookback=30),
    )
    _, info = env.reset(seed=0)
    assert "turbulence" in info
    _, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))
    assert "turbulence" in info
    assert "turbulence_triggered" in info
    assert info["turbulence"] >= 0.0


@pytest.mark.unit
def test_turbulence_gate_clamps_target(synthetic_ohlc):
    """Above threshold the target is clamped to the cap; below, it passes through."""
    env = PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=_turbulence_config(turbulence_exposure_cap=0.3, turbulence_lookback=30),
    )
    env.reset(seed=0)
    tick = env._current_tick

    env._turbulence[tick] = 100.0  # force a turbulent tick
    assert env._apply_turbulence_gate(0.9) == pytest.approx(0.3)
    assert env._apply_turbulence_gate(-0.9) == pytest.approx(0.0)  # long-only: floor at 0
    assert env._apply_turbulence_gate(0.1) == pytest.approx(0.1)  # within cap, untouched

    env._turbulence[tick] = 0.0  # calm tick -> full pass-through
    assert env._apply_turbulence_gate(0.9) == pytest.approx(0.9)


@pytest.mark.unit
def test_turbulence_gate_flattens_on_shock():
    """End-to-end: a full-long agent is forced flat on every turbulent day."""
    df = _shock_ohlc()
    env = PositionMgmtEnv(
        df=df,
        window_size=20,
        frame_bound=(20, len(df)),
        risk_config=_turbulence_config(),  # cap 0.0 -> flatten
    )
    env.reset(seed=0)
    triggered_any = False
    for _ in range(len(df)):
        _, _, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
        if info["turbulence_triggered"]:
            triggered_any = True
            assert info["position_ratio"] == pytest.approx(0.0)
        if term or trunc:
            break
    assert triggered_any, "The injected shock must trip the turbulence gate."


@pytest.mark.unit
def test_turbulence_gate_off_by_default(synthetic_ohlc):
    """Default config leaves the observation space and behaviour unchanged."""
    env = PositionMgmtEnv(
        df=synthetic_ohlc,
        window_size=20,
        frame_bound=(20, len(synthetic_ohlc)),
        risk_config=RiskConfig(),
    )
    assert env.observation_space.shape == (20, N_TECH_FEATURES + N_PORTFOLIO_FEATURES)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))
    assert info["turbulence_triggered"] is False
