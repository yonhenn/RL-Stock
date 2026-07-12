#!/usr/bin/env python3
"""CLI script to collect trajectories and build a Decision Transformer dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src/ to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3 import DDPG, PPO, SAC

from rl_stock.config import ExperimentConfig
from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.data_collection import build_dt_dataset
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig


def main():
    parser = argparse.ArgumentParser(description="Collect offline RL datasets for Decision Transformer.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="nasdaq100.xlsx",
        help="Path to NASDAQ-100 index Excel file.",
    )
    parser.add_argument(
        "--sac-model",
        type=str,
        default=None,
        help="Path to trained SAC model zip file (optional).",
    )
    parser.add_argument(
        "--ppo-model",
        type=str,
        default=None,
        help="Path to trained PPO model zip file (optional).",
    )
    parser.add_argument(
        "--ddpg-model",
        type=str,
        default=None,
        help="Path to trained DDPG model zip file (optional).",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of episodes to collect per policy (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for collection (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/dt_dataset.pkl",
        help="Path to output pickle file (default: outputs/dt_dataset.pkl).",
    )
    parser.add_argument(
        "--test-start-date",
        type=str,
        default="2024-01-01",
        help="Date to split train and test (default: 2024-01-01).",
    )
    parser.add_argument(
        "--long-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to restrict action space to [0, 1] (long-only). Default is True.",
    )
    args = parser.parse_args()

    # Load market data
    print(f"Loading market data from {args.data_path}...")
    df = load_market_data(args.data_path)
    train_bound, _ = split_frame_bounds(df, window_size=30, test_start_date=args.test_start_date)

    # Initialize environment in standard window mode
    # SAC/PPO were trained in window mode, but we will extract the 1D flat_obs
    # from info["flat_obs"] during collection.
    risk_config = RiskConfig(
        flat_obs=False,
        stop_loss_pct=0.05,        # 5% hard stop loss
        max_drawdown_pct=0.15,      # 15% drawdown circuit breaker
        use_trend_features=True,    # Use trend indicators in state
        long_only=args.long_only,
    )
    
    print("Initializing PositionMgmtEnv in window mode for policy compatibility...")
    env = PositionMgmtEnv(
        df=df,
        window_size=30,
        frame_bound=train_bound,
        risk_config=risk_config,
    )

    # Load policies
    policies = {"Random": None}

    if args.sac_model and Path(args.sac_model).exists():
        print(f"Loading SAC model from {args.sac_model}...")
        try:
            sac_model = SAC.load(args.sac_model, env=env)
            policies["SAC"] = sac_model
        except Exception as e:
            print(f"Failed to load SAC model: {e}", file=sys.stderr)
    elif args.sac_model:
        print(f"Warning: SAC model file not found: {args.sac_model}", file=sys.stderr)

    if args.ppo_model and Path(args.ppo_model).exists():
        print(f"Loading PPO model from {args.ppo_model}...")
        try:
            ppo_model = PPO.load(args.ppo_model, env=env)
            policies["PPO"] = ppo_model
        except Exception as e:
            print(f"Failed to load PPO model: {e}", file=sys.stderr)
    elif args.ppo_model:
        print(f"Warning: PPO model file not found: {args.ppo_model}", file=sys.stderr)

    if args.ddpg_model and Path(args.ddpg_model).exists():
        print(f"Loading DDPG model from {args.ddpg_model}...")
        try:
            ddpg_model = DDPG.load(args.ddpg_model, env=env)
            policies["DDPG"] = ddpg_model
        except Exception as e:
            print(f"Failed to load DDPG model: {e}", file=sys.stderr)
    elif args.ddpg_model:
        print(f"Warning: DDPG model file not found: {args.ddpg_model}", file=sys.stderr)

    print(f"Active collection policies: {list(policies.keys())}")

    # Build dataset
    build_dt_dataset(
        env=env,
        policies=policies,
        n_episodes_per_policy=args.n_episodes,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
