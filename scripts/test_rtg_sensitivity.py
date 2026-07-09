#!/usr/bin/env python3
"""Script to run sensitivity analysis on different target RTG values for Decision Transformer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Add src/ to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.dt_inference import DTAgent
from rl_stock.dt_model import DecisionTransformer
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.risk import compute_max_drawdown, compute_sharpe_ratio


def run_dt_eval_with_rtg(agent: DTAgent, env: PositionMgmtEnv, target_rtg: float, seed: int = 42) -> dict:
    # Set the target RTG on the agent
    agent.target_rtg = target_rtg
    agent.reset()

    obs, _ = env.reset(seed=seed)
    done = False
    
    actions = []
    positions = []
    net_returns = []
    equities = []

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        
        actions.append(action[0])
        positions.append(info.get("position_ratio", 0.0))
        net_returns.append(info.get("net_return", 0.0))
        equities.append(info.get("equity", 1.0))

    actions = np.array(actions)
    positions = np.array(positions)
    net_returns = np.array(net_returns)
    equities = np.array(equities)

    # Compute stats
    profit = info.get("total_profit", 1.0)
    sharpe = compute_sharpe_ratio(net_returns)
    max_dd = compute_max_drawdown(equities)
    
    # Action saturation: percentage of actions near the boundaries (+1 or -1)
    # Since actions are in [-1, 1], we check if |action| > 0.95
    saturation_rate = np.mean(np.abs(actions) > 0.95) * 100.0
    avg_pos = np.mean(positions)
    stop_outs = info.get("stop_loss_events", 0)

    return {
        "profit": (profit - 1.0) * 100.0,  # in %
        "sharpe": sharpe,
        "max_dd": max_dd * 100.0,          # in %
        "saturation_rate": saturation_rate,
        "avg_pos": avg_pos,
        "stop_outs": stop_outs,
    }


def main():
    data_path = ROOT / "nasdaq100.xlsx"
    dt_ckpt_path = ROOT / "outputs" / "dt" / "dt_model.pt"

    if not dt_ckpt_path.exists():
        print(f"Error: DT checkpoint not found at {dt_ckpt_path}")
        return

    # Load market data
    df = load_market_data(data_path)
    _, test_bound = split_frame_bounds(df, window_size=30, test_start_date="2024-01-01")

    # Initialize environment in flat mode
    risk_config = RiskConfig(
        flat_obs=True,
        stop_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trend_features=True,
    )
    env = PositionMgmtEnv(
        df=df,
        window_size=30,
        frame_bound=test_bound,
        risk_config=risk_config,
    )

    # Load DT checkpoint
    checkpoint = torch.load(dt_ckpt_path, map_location="cpu", weights_only=False)
    model = DecisionTransformer(
        state_dim=checkpoint["state_dim"],
        action_dim=1,
        context_length=checkpoint["context_length"],
        max_episode_len=checkpoint.get("max_episode_len", checkpoint["state_dict"]["embed_timestep.weight"].shape[0]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    
    agent = DTAgent(
        model=model,
        target_rtg=0.3,
        state_mean=checkpoint["state_mean"],
        state_std=checkpoint["state_std"],
        rtg_scale=checkpoint["rtg_scale"],
        device="cpu",
    )

    # Target RTG test list
    test_rtgs = [-0.5, 0.0, 0.05, 0.3, 0.8, 1.5, 3.0, 5.0]

    print("=" * 80)
    print(f"{'Target RTG':<12} | {'Profit (%)':<12} | {'Sharpe':<8} | {'Max DD (%)':<12} | {'Satur. (%)':<12} | {'Avg Pos':<8} | {'Stops':<5}")
    print("-" * 80)

    for rtg in test_rtgs:
        stats = run_dt_eval_with_rtg(agent, env, rtg, seed=42)
        print(f"{rtg:<12.2f} | {stats['profit']:<+12.2f} | {stats['sharpe']:<8.3f} | {stats['max_dd']:<12.2f} | {stats['saturation_rate']:<12.1f} | {stats['avg_pos']:<8.3f} | {stats['stop_outs']:<5}")
    print("=" * 80)


if __name__ == "__main__":
    main()
