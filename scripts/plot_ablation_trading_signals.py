#!/usr/bin/env python3
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("/Users/rickchen/Antigravity/RL wit Nasdaq")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.dt_inference import DTAgent
from rl_stock.dt_model import DecisionTransformer
from rl_stock.bc_inference import BCAgent
from rl_stock.bc_model import BehaviorCloning
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.visualization import configure_matplotlib

def run_agent_history(agent, env):
    obs, _ = env.reset(seed=42)
    if hasattr(agent, "reset"):
        agent.reset()
    done = False
    positions = []
    prices = []

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        positions.append(info.get("position_ratio", 0.0))
        prices.append(env.prices[info.get("tick")])

    return np.array(positions), np.array(prices)

def detect_signals(positions):
    buys = []
    sells = []
    for t in range(1, len(positions)):
        w_prev = positions[t-1]
        w_curr = positions[t]
        if (w_prev <= 0 and w_curr > 0) or (w_prev < 0 and w_curr >= 0):
            buys.append(t)
        elif (w_prev >= 0 and w_curr < 0) or (w_prev > 0 and w_curr <= 0):
            sells.append(t)
    return buys, sells

def make_dt_agent(ckpt_path, target_rtg):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DecisionTransformer(
        state_dim=ck["state_dim"],
        action_dim=1,
        context_length=ck["context_length"],
        max_episode_len=ck.get("max_episode_len", ck["state_dict"]["embed_timestep.weight"].shape[0]),
        action_activation="sigmoid" if ck["config"]["space"] == "long_only" else "tanh"
    )
    model.load_state_dict(ck["state_dict"])
    return DTAgent(model, target_rtg, ck["state_mean"], ck["state_std"], ck["rtg_scale"])

def make_bc_agent(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BehaviorCloning(
        state_dim=ck["state_dim"],
        action_dim=1,
        context_length=ck["context_length"],
        max_episode_len=ck.get("max_episode_len", ck["state_dict"]["embed_timestep.weight"].shape[0]),
        action_activation="sigmoid" if ck["config"]["space"] == "long_only" else "tanh"
    )
    model.load_state_dict(ck["state_dict"])
    return BCAgent(model, ck["state_mean"], ck["state_std"])

def main():
    configure_matplotlib()

    df = load_market_data(PROJECT_ROOT / "nasdaq100.xlsx")
    _, test_bound = split_frame_bounds(df, 30, "2024-01-01")

    # 1. Long-Only Plots (DT e40_s10 vs BC e60_s5)
    env_lo = PositionMgmtEnv(
        df, 30, test_bound,
        RiskConfig(flat_obs=True, stop_loss_pct=.05, max_drawdown_pct=.15, use_trend_features=True, long_only=True)
    )

    dt_lo_path = PROJECT_ROOT / "outputs/optimization_ablation/long_only/dt/e40_s10/epoch_040.pt"
    bc_lo_path = PROJECT_ROOT / "outputs/optimization_ablation/long_only/bc/e60_s5/epoch_060.pt"

    dt_lo_agent = make_dt_agent(dt_lo_path, 2.30)
    bc_lo_agent = make_bc_agent(bc_lo_path)

    print("Running Long-Only DT e40_s10...")
    dt_lo_pos, dt_lo_prices = run_agent_history(dt_lo_agent, env_lo)
    print("Running Long-Only BC e60_s5...")
    bc_lo_pos, bc_lo_prices = run_agent_history(bc_lo_agent, env_lo)

    dt_lo_buys, dt_lo_sells = detect_signals(dt_lo_pos)
    bc_lo_buys, bc_lo_sells = detect_signals(bc_lo_pos)

    dates_lo = df.index[test_bound[0] : test_bound[0] + len(dt_lo_prices)]

    # Plot Long-only ablation signals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

    # Subplot 1: DT
    ax1.plot(dates_lo, dt_lo_prices, color="#555555", linewidth=1.5, alpha=0.8, label="NASDAQ-100 Close")
    if dt_lo_buys:
        ax1.scatter(dates_lo[dt_lo_buys], dt_lo_prices[dt_lo_buys], color="green", marker="^", s=80, label="Buy / Cover", zorder=5)
    if dt_lo_sells:
        ax1.scatter(dates_lo[dt_lo_sells], dt_lo_prices[dt_lo_sells], color="red", marker="v", s=80, label="Sell / Short", zorder=5)
    ax1.set_ylabel("NASDAQ Close Price", fontsize=11)
    ax1.set_title("DT V2 P90 e40_s10 (Long-Only Space, Profit: +22.13%)", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(dates_lo, dt_lo_pos, color="#9467bd", linewidth=1.2, linestyle="--", alpha=0.7, label="Position Ratio")
    ax1_twin.fill_between(dates_lo, 0, dt_lo_pos, color="#9467bd", alpha=0.1)
    ax1_twin.set_ylabel("Target Position Ratio", color="#9467bd", fontsize=11)
    ax1_twin.tick_params(axis="y", labelcolor="#9467bd")
    ax1_twin.set_ylim(-0.05, 1.05)

    # Subplot 2: BC
    ax2.plot(dates_lo, bc_lo_prices, color="#555555", linewidth=1.5, alpha=0.8, label="NASDAQ-100 Close")
    if bc_lo_buys:
        ax2.scatter(dates_lo[bc_lo_buys], bc_lo_prices[bc_lo_buys], color="green", marker="^", s=80, label="Buy / Cover", zorder=5)
    if bc_lo_sells:
        ax2.scatter(dates_lo[bc_lo_sells], bc_lo_prices[bc_lo_sells], color="red", marker="v", s=80, label="Sell / Short", zorder=5)
    ax2.set_ylabel("NASDAQ Close Price", fontsize=11)
    ax2.set_title("BC e60_s5 (Long-Only Space, Profit: +11.42%)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(dates_lo, bc_lo_pos, color="#e377c2", linewidth=1.2, linestyle="--", alpha=0.7, label="Position Ratio")
    ax2_twin.fill_between(dates_lo, 0, bc_lo_pos, color="#e377c2", alpha=0.1)
    ax2_twin.set_ylabel("Target Position Ratio", color="#e377c2", fontsize=11)
    ax2_twin.tick_params(axis="y", labelcolor="#e377c2")
    ax2_twin.set_ylim(-0.05, 1.05)

    plt.xlabel("Date", fontsize=12)
    plt.tight_layout()
    plot_path_lo = PROJECT_ROOT / "outputs/optimization_ablation/long_only_ablation_signals.png"
    fig.savefig(plot_path_lo, dpi=160, bbox_inches="tight")
    print(f"Saved Long-Only ablation signals to {plot_path_lo}")
    plt.close(fig)

    # 2. Symmetric Plots (DT e60_s5 vs BC e60_s5)
    env_sym = PositionMgmtEnv(
        df, 30, test_bound,
        RiskConfig(flat_obs=True, stop_loss_pct=.05, max_drawdown_pct=.15, use_trend_features=True, long_only=False)
    )

    dt_sym_path = PROJECT_ROOT / "outputs/optimization_ablation/symmetric/dt/e60_s5/epoch_060.pt"
    bc_sym_path = PROJECT_ROOT / "outputs/optimization_ablation/symmetric/bc/e60_s5/epoch_060.pt"

    dt_sym_agent = make_dt_agent(dt_sym_path, 0.38)
    bc_sym_agent = make_bc_agent(bc_sym_path)

    print("Running Symmetric DT e60_s5...")
    dt_sym_pos, dt_sym_prices = run_agent_history(dt_sym_agent, env_sym)
    print("Running Symmetric BC e60_s5...")
    bc_sym_pos, bc_sym_prices = run_agent_history(bc_sym_agent, env_sym)

    dt_sym_buys, dt_sym_sells = detect_signals(dt_sym_pos)
    bc_sym_buys, bc_sym_sells = detect_signals(bc_sym_pos)

    dates_sym = df.index[test_bound[0] : test_bound[0] + len(dt_sym_prices)]

    # Plot Symmetric ablation signals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

    # Subplot 1: DT
    ax1.plot(dates_sym, dt_sym_prices, color="#555555", linewidth=1.5, alpha=0.8, label="NASDAQ-100 Close")
    if dt_sym_buys:
        ax1.scatter(dates_sym[dt_sym_buys], dt_sym_prices[dt_sym_buys], color="green", marker="^", s=80, label="Buy / Cover", zorder=5)
    if dt_sym_sells:
        ax1.scatter(dates_sym[dt_sym_sells], dt_sym_prices[dt_sym_sells], color="red", marker="v", s=80, label="Sell / Short", zorder=5)
    ax1.set_ylabel("NASDAQ Close Price", fontsize=11)
    ax1.set_title("DT V2 P90 e60_s5 (Symmetric Space, Profit: +8.84%)", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(dates_sym, dt_sym_pos, color="#2ca02c", linewidth=1.2, linestyle="--", alpha=0.7, label="Position Ratio")
    ax1_twin.fill_between(dates_sym, 0, dt_sym_pos, color="#2ca02c", alpha=0.1)
    ax1_twin.set_ylabel("Target Position Ratio", color="#2ca02c", fontsize=11)
    ax1_twin.tick_params(axis="y", labelcolor="#2ca02c")
    ax1_twin.set_ylim(-1.1, 1.1)
    ax1_twin.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    # Subplot 2: BC
    ax2.plot(dates_sym, bc_sym_prices, color="#555555", linewidth=1.5, alpha=0.8, label="NASDAQ-100 Close")
    if bc_sym_buys:
        ax2.scatter(dates_sym[bc_sym_buys], bc_sym_prices[bc_sym_buys], color="green", marker="^", s=80, label="Buy / Cover", zorder=5)
    if bc_sym_sells:
        ax2.scatter(dates_sym[bc_sym_sells], bc_sym_prices[bc_sym_sells], color="red", marker="v", s=80, label="Sell / Short", zorder=5)
    ax2.set_ylabel("NASDAQ Close Price", fontsize=11)
    ax2.set_title("BC e60_s5 (Symmetric Space, Profit: -0.74%)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(dates_sym, bc_sym_pos, color="#ff7f0e", linewidth=1.2, linestyle="--", alpha=0.7, label="Position Ratio")
    ax2_twin.fill_between(dates_sym, 0, bc_sym_pos, color="#ff7f0e", alpha=0.1)
    ax2_twin.set_ylabel("Target Position Ratio", color="#ff7f0e", fontsize=11)
    ax2_twin.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2_twin.set_ylim(-1.1, 1.1)
    ax2_twin.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.xlabel("Date", fontsize=12)
    plt.tight_layout()
    plot_path_sym = PROJECT_ROOT / "outputs/optimization_ablation/symmetric_ablation_signals.png"
    fig.savefig(plot_path_sym, dpi=160, bbox_inches="tight")
    print(f"Saved Symmetric ablation signals to {plot_path_sym}")
    plt.close(fig)

if __name__ == "__main__":
    main()
