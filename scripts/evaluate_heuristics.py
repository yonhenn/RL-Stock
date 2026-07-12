#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_stock.data import load_market_data, split_frame_bounds
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.indicators import historical_moving_average, compute_rsi
from rl_stock.risk import compute_max_drawdown, compute_sharpe_ratio

def evaluate_heuristic_policy(env, policy_func, long_only=True):
    obs, info = env.reset(seed=42)
    done = False

    actions = []
    positions = []
    net_returns = []
    equities = []

    # Track state for path-dependent policies (like RSI holding previous position)
    prev_action = 0.0

    while not done:
        tick = env._current_tick
        price = env.prices[tick]

        # Call policy function
        action_val, prev_action = policy_func(tick, price, prev_action)
        action = np.array([action_val], dtype=np.float32)

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

    profit = (info.get("total_profit", 1.0) - 1.0) * 100.0
    sharpe = compute_sharpe_ratio(net_returns)
    max_dd = compute_max_drawdown(equities) * 100.0

    return {
        "profit": profit,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "avg_raw_action": np.mean(actions),
        "avg_executed_position": np.mean(positions),
        "stop_outs": info.get("stop_loss_events", 0),
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate classic heuristic trading rules.")
    parser.add_argument("--data-path", type=str, default="nasdaq100.xlsx")
    parser.add_argument("--test-start-date", type=str, default="2024-01-01")
    parser.add_argument("--output-dir", type=str, default="outputs/heuristics")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_market_data(data_path)
    _, test_bound = split_frame_bounds(df, window_size=30, test_start_date=args.test_start_date)

    results = []

    for long_only in [True, False]:
        space_name = "Long-only [0,1]" if long_only else "Symmetric [-1,1]"
        print(f"Evaluating heuristics for {space_name} space...")

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

        # 1. Uptrend Heuristic (Close > SMA20)
        def uptrend_policy(tick, price, prev):
            if price > sma20[tick]:
                return 1.0, 1.0
            else:
                return (0.0 if long_only else -1.0), (0.0 if long_only else -1.0)

        # 2. Double SMA Crossover (SMA5 > SMA20)
        def crossover_policy(tick, price, prev):
            if sma5[tick] > sma20[tick]:
                return 1.0, 1.0
            else:
                return (0.0 if long_only else -1.0), (0.0 if long_only else -1.0)

        # 3. RSI Mean Reversion (RSI < 30 buy, RSI > 70 sell/short, else hold)
        def rsi_policy(tick, price, prev):
            current_rsi = rsi[tick]
            if current_rsi < 30:
                return 1.0, 1.0
            elif current_rsi > 70:
                return (0.0 if long_only else -1.0), (0.0 if long_only else -1.0)
            else:
                return prev, prev

        policies = {
            "Uptrend (Close > SMA20)": uptrend_policy,
            "SMA Crossover (SMA5 > SMA20)": crossover_policy,
            "RSI Mean Reversion (14)": rsi_policy
        }

        for name, policy_func in policies.items():
            stats = evaluate_heuristic_policy(env, policy_func, long_only=long_only)
            print(f"  {name:<30} | Profit: {stats['profit']:>+7.2f}% | Sharpe: {stats['sharpe']:>5.3f} | Max DD: {stats['max_dd']:>5.2f}% | Avg Pos: {stats['avg_executed_position']:>5.3f}")
            results.append({
                "action_space": space_name,
                "strategy": name,
                "profit": stats["profit"],
                "sharpe": stats["sharpe"],
                "max_dd": stats["max_dd"],
                "avg_raw_action": stats["avg_raw_action"],
                "avg_executed_position": stats["avg_executed_position"],
                "stop_outs": stats["stop_outs"],
            })

    res_df = pd.DataFrame(results)
    csv_path = output_dir / "heuristic_results.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"Saved heuristic evaluation results to {csv_path}")

if __name__ == "__main__":
    main()
