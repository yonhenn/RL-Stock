#!/usr/bin/env python3
"""Script to evaluate and compare Decision Transformer (DT) and Behavior Cloning (BC) baselines."""

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
from rl_stock.bc_inference import BCAgent
from rl_stock.bc_model import BehaviorCloning
from rl_stock.envs_v2 import PositionMgmtEnv, RiskConfig
from rl_stock.visualization import configure_matplotlib
from rl_stock.risk import compute_sharpe_ratio, compute_max_drawdown


class BuyAndHoldAgent:
    def reset(self):
        pass
    def predict(self, obs, deterministic=True):
        return np.array([1.0], dtype=np.float32), None


def evaluate_agent(agent, env, seed=42):
    obs, _ = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    done = False
    equity_curve = []
    raw_actions = []
    executed_positions = []
    returns = []
    stops = 0
    
    while not done:
        action, _ = agent.predict(obs, deterministic=True) if agent else (env.action_space.sample(), None)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        equity_curve.append(info.get("equity", 1.0))
        raw_actions.append(float(action[0]))
        executed_positions.append(float(info.get("position_ratio", action[0])))
        returns.append(info.get("net_return", 0.0))
        
        # Check if stop loss was triggered at this step
        if info.get("stop_triggered", False):
            stops += 1


    equity_curve = np.array(equity_curve)
    returns = np.array(returns)
    raw_actions = np.array(raw_actions)
    executed_positions = np.array(executed_positions)
    
    final_profit = info.get("total_profit", 1.0) - 1.0
    sharpe = compute_sharpe_ratio(returns)
    max_dd = compute_max_drawdown(equity_curve)
    calmar = final_profit / max_dd if max_dd > 0 else 0.0
    
    # Calculate action saturation (|action| > 0.95)
    saturation = np.mean(np.abs(raw_actions) > 0.95)
    avg_raw_action = np.mean(raw_actions)
    avg_executed_position = np.mean(executed_positions)

    return {
        "equity_curve": equity_curve,
        "profit": final_profit,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "saturation": saturation,
        # avg_pos is kept as a compatibility alias, but now correctly means
        # the position actually executed by the environment.
        "avg_pos": avg_executed_position,
        "avg_raw_action": avg_raw_action,
        "avg_executed_position": avg_executed_position,
        "stops": stops,
    }



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate and compare DT and BC models.")
    parser.add_argument("--data-path", type=str, default="nasdaq100.xlsx")
    parser.add_argument("--sac-model", type=str, default="outputs/sac.zip")
    parser.add_argument("--ppo-model", type=str, default="outputs/ppo.zip")
    parser.add_argument("--dt-checkpoint", type=str, default="outputs/dt/dt_model.pt")
    parser.add_argument("--bc-checkpoint", type=str, default="outputs/bc/bc_model.pt")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--test-start-date", type=str, default="2024-01-01")
    parser.add_argument(
        "--long-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to restrict action space to [0, 1] (long-only). Default is True.",
    )
    parser.add_argument("--target-rtg-low", type=float, default=0.3)
    parser.add_argument("--target-rtg-high", type=float, default=0.8)
    args = parser.parse_args()

    print("Configuring matplotlib and loading models...")
    configure_matplotlib()

    data_path = Path(args.data_path)
    sac_path = Path(args.sac_model)
    ppo_path = Path(args.ppo_model)
    dt_ckpt_path = Path(args.dt_checkpoint)
    bc_ckpt_path = Path(args.bc_checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load market data
    df = load_market_data(data_path)
    _, test_bound = split_frame_bounds(df, window_size=30, test_start_date=args.test_start_date)

    # Env configurations
    risk_config_window = RiskConfig(
        flat_obs=False,
        stop_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trend_features=True,
        long_only=args.long_only,
    )
    env_window = PositionMgmtEnv(df=df, window_size=30, frame_bound=test_bound, risk_config=risk_config_window)

    risk_config_flat = RiskConfig(
        flat_obs=True,
        stop_loss_pct=0.05,
        max_drawdown_pct=0.15,
        use_trend_features=True,
        long_only=args.long_only,
    )
    env_flat = PositionMgmtEnv(df=df, window_size=30, frame_bound=test_bound, risk_config=risk_config_flat)

    # 1. Load PPO / SAC
    sac_model = SAC.load(sac_path, env=env_window) if sac_path.exists() else None
    ppo_model = PPO.load(ppo_path, env=env_window) if ppo_path.exists() else None

    # 2. Load Decision Transformer
    dt_checkpoint = torch.load(dt_ckpt_path, map_location="cpu", weights_only=False)
    dt_model = DecisionTransformer(
        state_dim=dt_checkpoint["state_dim"],
        action_dim=1,
        context_length=dt_checkpoint["context_length"],
        max_episode_len=dt_checkpoint.get("max_episode_len", dt_checkpoint["state_dict"]["embed_timestep.weight"].shape[0]),
        action_activation=dt_checkpoint.get("action_activation", "tanh"),
    )
    dt_model.load_state_dict(dt_checkpoint["state_dict"])
    
    dt_agent_03 = DTAgent(
        model=dt_model,
        target_rtg=args.target_rtg_low,
        state_mean=dt_checkpoint["state_mean"],
        state_std=dt_checkpoint["state_std"],
        rtg_scale=dt_checkpoint["rtg_scale"],
        device="cpu",
    )
    
    dt_agent_08 = DTAgent(
        model=dt_model,
        target_rtg=args.target_rtg_high,
        state_mean=dt_checkpoint["state_mean"],
        state_std=dt_checkpoint["state_std"],
        rtg_scale=dt_checkpoint["rtg_scale"],
        device="cpu",
    )

    # 3. Load Behavior Cloning
    bc_checkpoint = torch.load(bc_ckpt_path, map_location="cpu", weights_only=False)
    bc_model = BehaviorCloning(
        state_dim=bc_checkpoint["state_dim"],
        action_dim=1,
        context_length=bc_checkpoint["context_length"],
        max_episode_len=bc_checkpoint.get("max_episode_len", bc_checkpoint["state_dict"]["embed_timestep.weight"].shape[0]),
        action_activation=bc_checkpoint.get("action_activation", "tanh"),
    )
    bc_model.load_state_dict(bc_checkpoint["state_dict"])
    
    bc_agent = BCAgent(
        model=bc_model,
        state_mean=bc_checkpoint["state_mean"],
        state_std=bc_checkpoint["state_std"],
        device="cpu",
    )

    # Run evaluations
    print("Evaluating Random baseline...")
    random_metrics = evaluate_agent(None, env_window, seed=42)

    print("Evaluating Buy-and-Hold baseline...")
    bh_agent = BuyAndHoldAgent()
    bh_metrics = evaluate_agent(bh_agent, env_window, seed=42)

    print("Evaluating SAC baseline...")
    sac_metrics = evaluate_agent(sac_model, env_window, seed=42) if sac_model else None

    print("Evaluating PPO baseline...")
    ppo_metrics = evaluate_agent(ppo_model, env_window, seed=42) if ppo_model else None

    print("Evaluating Behavior Cloning baseline...")
    bc_metrics = evaluate_agent(bc_agent, env_flat, seed=42)

    print(f"Evaluating Decision Transformer (RTG={args.target_rtg_low:.2f})...")
    dt03_metrics = evaluate_agent(dt_agent_03, env_flat, seed=42)

    print(f"Evaluating Decision Transformer (RTG={args.target_rtg_high:.2f})...")
    dt08_metrics = evaluate_agent(dt_agent_08, env_flat, seed=42)

    # Write comparative metrics to console and markdown
    print("\n" + "="*50)
    print("COMPARATIVE METRICS SUMMARY")
    print("="*50)
    
    def print_stat(name, metrics):
        if metrics is None:
            return
        print(f"{name:25s} | Profit: {metrics['profit']*100:6.2f}% | Sharpe: {metrics['sharpe']:5.3f} | MaxDD: {metrics['max_dd']*100:5.2f}% | Stops: {metrics['stops']}")
        
    print_stat("Random Baseline", random_metrics)
    print_stat("Buy & Hold Baseline", bh_metrics)
    print_stat("SAC Baseline", sac_metrics)
    print_stat("PPO Baseline", ppo_metrics)
    print_stat("Behavior Cloning (BC)", bc_metrics)
    print_stat(f"DT (RTG = {args.target_rtg_low:.2f})", dt03_metrics)
    print_stat(f"DT (RTG = {args.target_rtg_high:.2f} - Sweet Spot)", dt08_metrics)
    print("="*50)

    # Save metrics data to a file for reporting
    metrics_report = {
        "random": random_metrics,
        "buy_hold": bh_metrics,
        "sac": sac_metrics,
        "ppo": ppo_metrics,
        "bc": bc_metrics,
        "dt03": dt03_metrics,
        "dt08": dt08_metrics,
    }
    with open(output_dir / "comparison_dt_bc.pkl", "wb") as f:
        pickle.dump(metrics_report, f)

    # Build Comparative Equity Plot
    fig, ax = plt.subplots(figsize=(12, 6.5))
    dates = df.index[test_bound[0] : test_bound[0] + len(random_metrics["equity_curve"])]
    
    # Baselines
    ax.plot(dates, np.array(random_metrics["equity_curve"]) / 1e6, label="Random Baseline", color="#a9a9a9", linestyle="--", alpha=0.7)
    ax.plot(dates, np.array(bh_metrics["equity_curve"]) / 1e6, label="Buy & Hold Baseline", color="#8c564b", linestyle="-", alpha=0.8)
    if sac_metrics:
        ax.plot(dates, np.array(sac_metrics["equity_curve"]) / 1e6, label="SAC Baseline", color="#ffbb78", linestyle=":", alpha=0.8)
    if ppo_metrics:
        ax.plot(dates, np.array(ppo_metrics["equity_curve"]) / 1e6, label="PPO Baseline", color="#98df8a", linestyle="-.", alpha=0.8)
    
    # BC (No RTG)
    ax.plot(dates, np.array(bc_metrics["equity_curve"]) / 1e6, label="Behavior Cloning (BC, No RTG)", color="#ff7f0e", linewidth=2.0)
    
    # DTs
    ax.plot(dates, np.array(dt03_metrics["equity_curve"]) / 1e6, label=f"DT (RTG = {args.target_rtg_low:.2f})", color="#aec7e8", linewidth=1.8)
    ax.plot(dates, np.array(dt08_metrics["equity_curve"]) / 1e6, label=f"Decision Transformer (DT, RTG = {args.target_rtg_high:.2f})", color="#1f77b4", linewidth=2.8)

    ax.set_title("Equity Curve Comparison: DT vs. Behavior Cloning (BC) & Baselines", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Equity ($ Millions)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10.5, loc="upper left")

    plt.tight_layout()
    plot_path = output_dir / "dt_bc_comparison_v2.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Saved DT vs BC comparison plot to {plot_path}")

    # Generate risk dashboards for BC, DT (RTG=low), and DT (RTG=high)
    print(f"Generating risk dashboards for BC, DT ({args.target_rtg_low:.2f}) and DT ({args.target_rtg_high:.2f})...")
    from rl_stock.visualization import visualize_trading
    bc_agent.reset()
    visualize_trading(bc_agent, env_flat, "Behavior_Cloning", seed=42, save_path=output_dir / "trades_bc_v2.png")
    dt_agent_03.reset()
    visualize_trading(dt_agent_03, env_flat, f"DT_RTG_{args.target_rtg_low:.2f}", seed=42, save_path=output_dir / "trades_dt_03_v2.png")
    dt_agent_08.reset()
    visualize_trading(dt_agent_08, env_flat, f"DT_RTG_{args.target_rtg_high:.2f}", seed=42, save_path=output_dir / "trades_dt_08_v2.png")


if __name__ == "__main__":
    main()
