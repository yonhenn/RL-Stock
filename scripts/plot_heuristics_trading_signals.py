#!/usr/bin/env python3
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.indicators import historical_moving_average, compute_rsi
from rl_stock.visualization import configure_matplotlib

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

def run_policy_signals(env, policy_func):
    obs, info = env.reset(seed=42)
    done = False
    positions = []
    prices = []
    prev_action = 0.0

    while not done:
        tick = env._current_tick
        price = env.prices[tick]
        action_val, prev_action = policy_func(tick, price, prev_action)
        obs, reward, term, trunc, info = env.step(np.array([action_val], dtype=np.float32))
        done = term or trunc
        positions.append(info.get("position_ratio", 0.0))
        prices.append(env.prices[info.get("tick")])

    return np.array(positions), np.array(prices)

def main():
    configure_matplotlib()

    data_path = Path("nasdaq100.xlsx")
    df = load_market_data(data_path)
    _, test_bound = split_frame_bounds(df, window_size=30, test_start_date="2024-01-01")

    for long_only in [True, False]:
        space_label = "long_only" if long_only else "symmetric"
        space_title = "Long-Only [0, 1] Space" if long_only else "Symmetric [-1, 1] Space"

        risk_config = RiskConfig(
            flat_obs=True,
            stop_loss_pct=0.05,
            max_drawdown_pct=0.15,
            use_trend_features=True,
            long_only=long_only,
        )
        env = PositionMgmtEnv(df=df, window_size=30, frame_bound=test_bound, risk_config=risk_config)

        prices = env.prices
        sma20 = historical_moving_average(prices, 20)
        sma5 = historical_moving_average(prices, 5)
        rsi = compute_rsi(prices, 14)

        # Policy definitions
        uptrend_p = lambda t, p, prev: (1.0, 1.0) if p > sma20[t] else ((0.0 if long_only else -1.0), (0.0 if long_only else -1.0))
        crossover_p = lambda t, p, prev: (1.0, 1.0) if sma5[t] > sma20[t] else ((0.0 if long_only else -1.0), (0.0 if long_only else -1.0))
        def rsi_p(t, p, prev):
            if rsi[t] < 30: return 1.0, 1.0
            elif rsi[t] > 70: return ((0.0 if long_only else -1.0), (0.0 if long_only else -1.0))
            return prev, prev

        policies = [
            ("Uptrend (Close > SMA20)", uptrend_p, "#ff7f0e"),
            ("SMA Crossover (SMA5 > SMA20)", crossover_p, "#2ca02c"),
            ("RSI Mean Reversion (14)", rsi_p, "#d62728")
        ]

        fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

        for idx, (title, policy_func, color) in enumerate(policies):
            ax = axes[idx]
            pos, prc = run_policy_signals(env, policy_func)
            dates = df.index[test_bound[0] : test_bound[0] + len(prc)]
            buys, sells = detect_signals(pos)

            # Plot price on primary y-axis
            ax.plot(dates, prc, color="#555555", linewidth=1.5, alpha=0.8, label="NASDAQ-100 Close")

            # Plot Buy/Sell markers
            if buys:
                ax.scatter(dates[buys], prc[buys], color="green", marker="^", s=70, label="Buy / Cover", zorder=5)
            if sells:
                ax.scatter(dates[sells], prc[sells], color="red", marker="v", s=70, label="Sell / Short", zorder=5)

            ax.set_ylabel("NASDAQ Close Price", fontsize=11)
            ax.set_title(f"{title} ({space_title})", fontsize=13, fontweight="bold", pad=8)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")

            # Twin axis for position ratio
            ax2 = ax.twinx()
            ax2.plot(dates[:len(pos)], pos, color=color, linewidth=1.2, linestyle="--", alpha=0.7, label="Position Ratio")
            ax2.fill_between(dates[:len(pos)], 0, pos, color=color, alpha=0.1)
            ax2.set_ylabel("Target Position Ratio", color=color, fontsize=11)
            ax2.tick_params(axis="y", labelcolor=color)
            if long_only:
                ax2.set_ylim(-0.05, 1.05)
            else:
                ax2.set_ylim(-1.1, 1.1)
            ax2.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

        plt.xlabel("Date", fontsize=12)
        plt.tight_layout()

        plot_path = Path(f"outputs/heuristics/{space_label}_heuristics_signals.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=160, bbox_inches="tight")
        print(f"Saved signals plot to {plot_path}")
        plt.close(fig)

if __name__ == "__main__":
    main()
