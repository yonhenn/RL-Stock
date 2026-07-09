#!/usr/bin/env python3
"""Script to run and plot the equity curves of all tested target RTG values on the same chart."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src/ to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.dt_inference import DTAgent
from rl_stock.dt_model import DecisionTransformer
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.visualization import configure_matplotlib


def main():
    configure_matplotlib()
    data_path = ROOT / "nasdaq100.xlsx"
    dt_ckpt_path = ROOT / "outputs" / "dt" / "dt_model.pt"
    output_dir = ROOT / "outputs"

    # Load market data
    df = load_market_data(data_path)
    _, test_bound = split_frame_bounds(df, window_size=30, test_start_date="2024-01-01")

    # Initialize environment
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

    # All Target RTGs
    test_rtgs = [-0.5, 0.0, 0.05, 0.3, 0.8, 1.5, 3.0, 5.0]
    
    # Custom colormap/colors for visualization clarity
    colors = {
        -0.5: "#7f7f7f",  # Gray (Risk off)
        0.0: "#c7c7c7",   # Light Gray
        0.05: "#bcbd22",  # Olive
        0.3: "#2ca02c",   # Green (Baseline)
        0.8: "#1f77b4",   # Deep Blue (Optimal Sweet Spot - thick)
        1.5: "#ff7f0e",   # Orange
        3.0: "#9467bd",   # Purple (Max profit before collapse)
        5.0: "#d62728",   # Red (Collapse - thick)
    }

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Evaluate each RTG and plot
    for rtg in test_rtgs:
        print(f"Evaluating curve for Target RTG = {rtg}...")
        agent.target_rtg = rtg
        agent.reset()
        
        obs, _ = env.reset(seed=42)
        done = False
        equity_curve = []
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            equity_curve.append(info.get("equity", 1.0))
            
        dates = df.index[test_bound[0] : test_bound[0] + len(equity_curve)]
        
        # Style highlights
        linewidth = 2.8 if rtg in [0.8, 5.0] else 1.5
        label = f"Target RTG = {rtg:.2f}"
        if rtg == 0.8:
            label += " (🏆 Sweet Spot)"
        elif rtg == 5.0:
            label += " (❌ Collapse)"
            
        ax.plot(dates, np.array(equity_curve) / 1e6, label=label, color=colors[rtg], linewidth=linewidth)

    ax.set_title("Equity Curve Comparison Across All Target RTGs", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Equity ($ Millions)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.tight_layout()
    plot_path = output_dir / "dt_all_rtgs_curves.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Saved all RTG comparison curves to {plot_path}")


if __name__ == "__main__":
    main()
