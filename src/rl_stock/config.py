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

    # Environment selection: "t1" (legacy discrete) or "position" (continuous risk-managed).
    env_type: str = "t1"
    # Algorithms to train. Valid: "A2C", "PPO", "SAC", "TD3".
    algorithms: tuple[str, ...] = ("A2C", "PPO")

    # Risk-control knobs (only consumed by env_type == "position").
    initial_cash: float = 1_000_000.0
    max_position: float = 1.0
    stop_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    trade_fee_rate: float = 0.001
    vol_target: float = 0.02
    drawdown_coeff: float = 0.5
    turnover_coeff: float = 0.1
    sharpe_coeff: float = 1.0
    dsr_eta: float = 0.01
