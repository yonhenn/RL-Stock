#!/usr/bin/env python3
"""Script to evaluate and generate performance comparison plots for PPO, SAC, and DT."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO, SAC

# Add src/ to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.dt_inference import DTAgent
from rl_stock.dt_model import DecisionTransformer
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.visualization import configure_matplotlib, visualize_trading


def main():
    print("Configuring matplotlib and loading models...")
    configure_matplotlib()

    # Paths
    data_path = ROOT / "nasdaq100.xlsx"
    sac_path = ROOT / "outputs" / "sac.zip"
    ppo_path = ROOT / "outputs" / "ppo.zip"
    dt_ckpt_path = ROOT / "outputs" / "dt" / "dt_model.pt"
    output_dir = ROOT / "outputs"

    # Load market data
    df = load_market_data(data_path)
    _, test_bound = split_frame_bounds(df, window_size=30, test_start_date="2024-01-01")

    # 1. Environment in standard windowed mode (for PPO/SAC)
    risk_config_window = RiskConfig(
        flat_obs=False,
        stop_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trend_features=True,
    )
    env_window = PositionMgmtEnv(
        df=df,
        window_size=30,
        frame_bound=test_bound,
        risk_config=risk_config_window,
    )

    # 2. Environment in flat mode (for DT)
    risk_config_flat = RiskConfig(
        flat_obs=True,
        stop_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trend_features=True,
    )
    env_flat = PositionMgmtEnv(
        df=df,
        window_size=30,
        frame_bound=test_bound,
        risk_config=risk_config_flat,
    )

    # 3. Load DT model
    if not dt_ckpt_path.exists():
        print(f"Error: DT checkpoint not found at {dt_ckpt_path}")
        return

    print("Loading DT checkpoint...")
    checkpoint = torch.load(dt_ckpt_path, map_location="cpu", weights_only=False)
    model = DecisionTransformer(
        state_dim=checkpoint["state_dim"],
        action_dim=1,
        context_length=checkpoint["context_length"],
        max_episode_len=checkpoint.get("max_episode_len", checkpoint["state_dict"]["embed_timestep.weight"].shape[0]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    
    dt_agent = DTAgent(
        model=model,
        target_rtg=0.3,  # Target RTG
        state_mean=checkpoint["state_mean"],
        state_std=checkpoint["state_std"],
        rtg_scale=checkpoint["rtg_scale"],
        device="cpu",
    )

    # 4. Load SAC/PPO models
    sac_model = SAC.load(sac_path, env=env_window) if sac_path.exists() else None
    ppo_model = PPO.load(ppo_path, env=env_window) if ppo_path.exists() else None

    # Run evaluations and generate risk dashboards (trades_*.png)
    print("\nRunning evaluation and generating dashboards...")
    
    print("Evaluating Random baseline...")
    visualize_trading(None, env_window, "Random", seed=42, save_path=output_dir / "trades_random.png")

    if sac_model:
        print("Evaluating SAC...")
        visualize_trading(sac_model, env_window, "SAC", seed=42, save_path=output_dir / "trades_sac.png")

    if ppo_model:
        print("Evaluating PPO...")
        visualize_trading(ppo_model, env_window, "PPO", seed=42, save_path=output_dir / "trades_ppo.png")

    print("Evaluating Decision Transformer...")
    dt_agent.reset()
    dt_info, _ = visualize_trading(dt_agent, env_flat, "Decision_Transformer", seed=42, save_path=output_dir / "trades_dt.png")

    # 5. Build comparative Equity Curve overlay plot
    print("\nGenerating unified comparative equity curve plot...")
    
    # Run helper function to extract equity histories
    def run_ep(model, env, seed):
        obs, _ = env.reset(seed=seed)
        if hasattr(model, "reset"):
            model.reset()
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True) if model else (env.action_space.sample(), None)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        return info["equity_curve"]

    equity_random = run_ep(None, env_window, 42)
    equity_sac = run_ep(sac_model, env_window, 42) if sac_model else None
    equity_ppo = run_ep(ppo_model, env_window, 42) if ppo_model else None
    equity_dt = run_ep(dt_agent, env_flat, 42)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = df.index[test_bound[0] : test_bound[0] + len(equity_random)]
    
    ax.plot(dates, np.array(equity_random) / 1e6, label="Random Baseline", color="gray", alpha=0.6, linestyle="--")
    if equity_sac:
        ax.plot(dates, np.array(equity_sac) / 1e6, label="SAC Agent", color="#ff7f0e", alpha=0.8)
    if equity_ppo:
        ax.plot(dates, np.array(equity_ppo) / 1e6, label="PPO Agent", color="#2ca02c", alpha=0.8)
    
    # DT Highlight (thick dark blue line)
    ax.plot(dates, np.array(equity_dt) / 1e6, label="Decision Transformer (DT)", color="#1f77b4", linewidth=2.5)

    ax.set_title("Strategy Performance Comparison (NASDAQ-100 Test Period)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Equity ($ Millions)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper left")

    plt.tight_layout()
    comp_plot_path = output_dir / "dt_comparison.png"
    fig.savefig(comp_plot_path, dpi=160, bbox_inches="tight")
    print(f"Saved comparative performance plot to {comp_plot_path}")

    # Generate Learning Curves from DT history
    history_path = output_dir / "dt" / "history.pkl"
    if history_path.exists():
        with open(history_path, "rb") as f:
            hist = pickle.load(f)
        
        fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Subplot 1: Train Loss
        axes[0].plot(range(1, len(hist["train_loss"]) + 1), hist["train_loss"], color="#d62728", marker="o", linewidth=2)
        axes[0].set_title("DT Training Loss Decay", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Masked MSE Loss", fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Subplot 2: Eval Sharpe
        epochs = hist["eval_epoch"]
        axes[1].plot(epochs, hist["eval_sharpe"], color="#1f77b4", marker="s", linewidth=2)
        axes[1].set_title("DT Online Evaluation Sharpe Ratio", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Annualized Sharpe Ratio", fontsize=11)
        axes[1].axhline(y=0.455, color="green", linestyle="--", label="PPO Baseline (0.455)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        curves_path = output_dir / "dt_learning_curves.png"
        fig2.savefig(curves_path, dpi=160, bbox_inches="tight")
        print(f"Saved DT learning curves to {curves_path}")


if __name__ == "__main__":
    main()
