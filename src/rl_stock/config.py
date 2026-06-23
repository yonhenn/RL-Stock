"""Experiment configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: Path
    test_start_date: str = "2024-01-01"
    window_size: int = 30
    total_timesteps: int = 100_000
    num_eval_episodes: int = 20
    seed: int = 42
    output_dir: Path = Path("outputs")
    show_plots: bool = False
    save_models: bool = True

    # Environment selection: "position" (continuous risk-managed) or "t1" (legacy
    # discrete; T+1 is a China A-share rule and does not apply to NASDAQ).
    env_type: str = "position"
    # Algorithms to train. Valid: "A2C", "PPO", "SAC", "TD3", "DDPG".
    algorithms: tuple[str, ...] = ("SAC", "PPO")

    # Risk-control knobs (only consumed by env_type == "position").
    initial_cash: float = 1_000_000.0
    max_position: float = 1.0
    # Hard risk rails, both opt-in (0.0 disables). Default off: the agent
    # manages risk through the reward (DSR + drawdown/turnover penalties)
    # rather than a forced stop-loss / drawdown lock.
    stop_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_fee_rate: float = 0.001
    vol_target: float = 0.02
    drawdown_coeff: float = 0.5
    turnover_coeff: float = 0.1
    sharpe_coeff: float = 1.0
    dsr_eta: float = 0.01
    # Upgrade the state space with trend-aware features (SMA slope, BB width,
    # ADX, MACD golden/death cross, trend score, regime). Only consumed by
    # env_type == "position".
    use_trend_features: bool = False
    # Trend-aware position management (residual policy): the explicit trend
    # judgment anchors a base position and the agent tilts around it, with
    # optional regime-aware counter-trend caps and an alignment reward.
    use_trend_prior: bool = False
    trend_base_exposure: float = 0.8
    residual_band: float = 0.5
    regime_risk_scaling: bool = False
    counter_trend_cap: float = 0.2
    trend_align_coeff: float = 0.0
    adx_trend_threshold: float = 20.0
    adx_strong_threshold: float = 40.0
    # Turbulence circuit breaker (FinRL-style market-crash risk-off, single-asset
    # proxy). Adds a turbulence state feature and caps exposure on outsized
    # moves. Only consumed by env_type == "position".
    use_turbulence_gate: bool = False
    turbulence_lookback: int = 252
    turbulence_threshold: float = 9.0
    turbulence_exposure_cap: float = 0.0
