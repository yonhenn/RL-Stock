#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.indicators import historical_moving_average, compute_rsi
from rl_stock.visualization import configure_matplotlib
from rl_stock.dt_inference import DTAgent
from rl_stock.dt_model import DecisionTransformer

def run_policy_equity(env, policy_func):
    obs, info = env.reset(seed=42)
    done = False
    equities = [1.0]
    prev_action = 0.0

    while not done:
        tick = env._current_tick
        price = env.prices[tick]
        action_val, prev_action = policy_func(obs, tick, price, prev_action)
        obs, reward, term, trunc, info = env.step(np.array([action_val], dtype=np.float32))
        done = term or trunc
        equities.append(info.get("equity", 1.0))

    return np.array(equities)

def main():
    configure_matplotlib()

    data_path = Path("nasdaq100.xlsx")
    df = load_market_data(data_path)
    _, test_bound = split_frame_bounds(df, window_size=30, test_start_date="2024-01-01")

    # Let's plot for both Long-Only and Symmetric
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Long-Only Panel
    long_only_config = RiskConfig(
        flat_obs=True,
        stop_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trend_features=True,
        long_only=True,
    )
    env_lo = PositionMgmtEnv(df=df, window_size=30, frame_bound=test_bound, risk_config=long_only_config)
    prices = env_lo.prices
    sma20 = historical_moving_average(prices, 20)
    sma5 = historical_moving_average(prices, 5)
    rsi = compute_rsi(prices, 14)

    # Policies for Long-Only
    uptrend_lo = lambda o, t, p, prev: (1.0, 1.0) if p > sma20[t] else (0.0, 0.0)
    crossover_lo = lambda o, t, p, prev: (1.0, 1.0) if sma5[t] > sma20[t] else (0.0, 0.0)
    def rsi_lo(o, t, p, prev):
        if rsi[t] < 30: return 1.0, 1.0
        elif rsi[t] > 70: return 0.0, 0.0
        return prev, prev

    eq_bh = run_policy_equity(env_lo, lambda o, t, p, prev: (1.0, 1.0))
    dates = df.index[test_bound[0] : test_bound[0] + len(eq_bh)]
    eq_uptrend = run_policy_equity(env_lo, uptrend_lo)
    eq_cross = run_policy_equity(env_lo, crossover_lo)
    eq_rsi = run_policy_equity(env_lo, rsi_lo)

    # Load DT Long-Only V2
    dt_lo_path = Path("outputs/long_only_v2_mvp/dt/dt_model.pt")
    if dt_lo_path.exists():
        ckpt = torch.load(dt_lo_path, map_location="cpu", weights_only=False)
        model = DecisionTransformer(
            state_dim=ckpt["state_dim"],
            action_dim=1,
            context_length=ckpt["context_length"],
            max_episode_len=ckpt.get("max_episode_len", ckpt["state_dict"]["embed_timestep.weight"].shape[0]),
            action_activation=ckpt.get("action_activation", "tanh")
        )
        model.load_state_dict(ckpt["state_dict"])
        agent = DTAgent(model, 2.30, ckpt["state_mean"], ckpt["state_std"], ckpt["rtg_scale"]) # P90
        eq_dt = run_policy_equity(env_lo, lambda o, t, p, prev: (agent.predict(o, deterministic=True)[0][0], 0.0))
        ax1.plot(dates[:len(eq_dt)], eq_dt, label="DT V2 P90 (RTG=2.30)", color="#1f77b4", linewidth=2.0)

    ax1.plot(dates[:len(eq_bh)], eq_bh, label="Passive Buy & Hold", color="brown", linestyle="--")
    ax1.plot(dates[:len(eq_cross)], eq_cross, label="SMA Crossover (5 vs 20)", color="#2ca02c")
    ax1.plot(dates[:len(eq_uptrend)], eq_uptrend, label="Uptrend (Close > SMA20)", color="#ff7f0e")
    ax1.plot(dates[:len(eq_rsi)], eq_rsi, label="RSI Mean Reversion", color="#7f7f7f", linestyle=":")

    ax1.set_title("Long-Only [0, 1] Space Heuristics vs. DT", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value (Base 1.0)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # 2. Symmetric Panel
    sym_config = RiskConfig(
        flat_obs=True,
        stop_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trend_features=True,
        long_only=False,
    )
    env_sym = PositionMgmtEnv(df=df, window_size=30, frame_bound=test_bound, risk_config=sym_config)

    # Policies for Symmetric
    uptrend_sym = lambda o, t, p, prev: (1.0, 1.0) if p > sma20[t] else (-1.0, -1.0)
    crossover_sym = lambda o, t, p, prev: (1.0, 1.0) if sma5[t] > sma20[t] else (-1.0, -1.0)
    def rsi_sym(o, t, p, prev):
        if rsi[t] < 30: return 1.0, 1.0
        elif rsi[t] > 70: return -1.0, -1.0
        return prev, prev

    eq_bh_sym = run_policy_equity(env_sym, lambda o, t, p, prev: (1.0, 1.0))
    eq_uptrend_sym = run_policy_equity(env_sym, uptrend_sym)
    eq_cross_sym = run_policy_equity(env_sym, crossover_sym)
    eq_rsi_sym = run_policy_equity(env_sym, rsi_sym)

    # Load DT Symmetric V2
    dt_sym_path = Path("outputs/symmetric_v2_mvp/dt/dt_model.pt")
    if dt_sym_path.exists():
        ckpt = torch.load(dt_sym_path, map_location="cpu", weights_only=False)
        model = DecisionTransformer(
            state_dim=ckpt["state_dim"],
            action_dim=1,
            context_length=ckpt["context_length"],
            max_episode_len=ckpt.get("max_episode_len", ckpt["state_dict"]["embed_timestep.weight"].shape[0]),
            action_activation=ckpt.get("action_activation", "tanh")
        )
        model.load_state_dict(ckpt["state_dict"])
        agent = DTAgent(model, 0.38, ckpt["state_mean"], ckpt["state_std"], ckpt["rtg_scale"]) # P90
        eq_dt_sym = run_policy_equity(env_sym, lambda o, t, p, prev: (agent.predict(o, deterministic=True)[0][0], 0.0))
        ax2.plot(dates[:len(eq_dt_sym)], eq_dt_sym, label="DT V2 P90 (RTG=0.38)", color="#1f77b4", linewidth=2.0)

    ax2.plot(dates[:len(eq_bh_sym)], eq_bh_sym, label="Passive Buy & Hold", color="brown", linestyle="--")
    ax2.plot(dates[:len(eq_cross_sym)], eq_cross_sym, label="SMA Crossover (5 vs 20)", color="#2ca02c")
    ax2.plot(dates[:len(eq_uptrend_sym)], eq_uptrend_sym, label="Uptrend (Close > SMA20)", color="#ff7f0e")
    ax2.plot(dates[:len(eq_rsi_sym)], eq_rsi_sym, label="RSI Mean Reversion", color="#7f7f7f", linestyle=":")

    ax2.set_title("Symmetric [-1, 1] Space Heuristics vs. DT", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Portfolio Value (Base 1.0)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plot_path = Path("outputs/heuristics/heuristics_equity_comparison.png")
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Saved heuristics comparison plot to {plot_path}")

if __name__ == "__main__":
    main()
