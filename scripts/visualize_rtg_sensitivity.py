#!/usr/bin/env python3
"""Script to evaluate different target RTGs and plot the sensitivity results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src/ to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.dt_inference import DTAgent
from rl_stock.dt_model import DecisionTransformer
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.visualization import configure_matplotlib
from scripts.test_rtg_sensitivity import run_dt_eval_with_rtg


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

    # Target RTGs to evaluate
    test_rtgs = [-0.5, 0.0, 0.05, 0.3, 0.8, 1.5, 3.0, 5.0]
    
    profits = []
    sharpes = []
    saturations = []
    positions = []

    print("Running evaluations for sensitivity visualization...")
    for rtg in test_rtgs:
        stats = run_dt_eval_with_rtg(agent, env, rtg, seed=42)
        profits.append(stats["profit"])
        sharpes.append(stats["sharpe"])
        saturations.append(stats["saturation_rate"])
        positions.append(stats["avg_pos"])

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Target RTG vs Profit and Sharpe
    ax1 = axes[0]
    color = "#1f77b4"
    ax1.set_xlabel("Target RTG (Commanded Goal)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Actual Profit (%)", color=color, fontsize=11, fontweight="bold")
    line1 = ax1.plot(test_rtgs, profits, color=color, marker="o", linewidth=2.5, label="Profit (%)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Instantiate a second y-axis that shares the same x-axis for Sharpe Ratio
    ax1_twin = ax1.twinx()  
    color_twin = "#2ca02c"
    ax1_twin.set_ylabel("Annualized Sharpe Ratio", color=color_twin, fontsize=11, fontweight="bold")
    line2 = ax1_twin.plot(test_rtgs, sharpes, color=color_twin, marker="s", linewidth=2.5, linestyle="--", label="Sharpe")
    ax1_twin.tick_params(axis="y", labelcolor=color_twin)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")
    ax1.set_title("Target RTG vs. Actual Return & Sharpe", fontsize=13, fontweight="bold", pad=10)

    # Panel 2: Target RTG vs Action Saturation and Avg Position
    ax2 = axes[1]
    color_sat = "#d62728"
    ax2.set_xlabel("Target RTG (Commanded Goal)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Action Saturation Rate (%)", color=color_sat, fontsize=11, fontweight="bold")
    line3 = ax2.plot(test_rtgs, saturations, color=color_sat, marker="^", linewidth=2.5, label="Saturation Rate (%)")
    ax2.tick_params(axis="y", labelcolor=color_sat)
    ax2.grid(True, alpha=0.3)

    ax2_twin = ax2.twinx()
    color_pos = "#9467bd"
    ax2_twin.set_ylabel("Average Position Exposure", color=color_pos, fontsize=11, fontweight="bold")
    line4 = ax2_twin.plot(test_rtgs, positions, color=color_pos, marker="v", linewidth=2.5, linestyle="--", label="Avg Position")
    ax2_twin.tick_params(axis="y", labelcolor=color_pos)

    lines2 = line3 + line4
    labels2 = [l.get_label() for l in lines2]
    ax2.legend(lines2, labels2, loc="upper left")
    ax2.set_title("Target RTG vs. Action Saturation & Avg Position", fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    plot_path = output_dir / "dt_rtg_sensitivity.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Saved sensitivity plot to {plot_path}")


if __name__ == "__main__":
    main()
