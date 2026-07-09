#!/usr/bin/env python3
"""Script to generate risk dashboards for boundary RTG values (-0.5, 0.8, 5.0)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add src/ to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.dt_inference import DTAgent
from rl_stock.dt_model import DecisionTransformer
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.visualization import configure_matplotlib, visualize_trading


def main():
    configure_matplotlib()
    data_path = ROOT / "nasdaq100.xlsx"
    dt_ckpt_path = ROOT / "outputs" / "dt" / "dt_model.pt"
    output_dir = ROOT / "outputs"

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

    # RTGs to evaluate and generate dashboards for
    boundary_rtgs = [-0.5, 0.8, 5.0]
    filename_map = {
        -0.5: "trades_dt_neg05.png",
        0.8: "trades_dt_08.png",
        5.0: "trades_dt_50.png",
    }

    for rtg in boundary_rtgs:
        print(f"Generating risk dashboard for RTG = {rtg}...")
        agent.target_rtg = rtg
        agent.reset()
        
        save_path = output_dir / filename_map[rtg]
        visualize_trading(
            model=agent,
            env=env,
            model_name=f"DT_RTG_{rtg}",
            seed=42,
            save_path=save_path,
        )
        print(f"Saved dashboard to {save_path}")


if __name__ == "__main__":
    main()
