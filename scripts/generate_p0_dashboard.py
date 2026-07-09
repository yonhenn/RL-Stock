#!/usr/bin/env python3
"""Script to evaluate PPO, SAC, Random, and all DT target RTGs to generate P0 core paper plots."""

from __future__ import annotations

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
from rl_stock.risk import (
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
)
from rl_stock.visualization import (
    configure_matplotlib,
    plot_action_saturation_distribution,
    plot_pareto_frontier,
    plot_rtg_calibration,
)


def run_one_env_episode(model, env, seed: int = 42) -> dict:
    """Run one episode on the environment and collect trajectory statistics."""
    obs, _ = env.reset(seed=seed)
    if hasattr(model, "reset"):
        model.reset()
    done = False
    
    actions = []
    positions = []
    returns = []
    equity = []

    while not done:
        action, _ = model.predict(obs, deterministic=True) if model else (env.action_space.sample(), None)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        
        actions.append(action[0])
        positions.append(info.get("position_ratio", 0.0))
        returns.append(info.get("net_return", 0.0))
        equity.append(info.get("equity", 1e6))

    actions = np.array(actions)
    positions = np.array(positions)
    returns = np.array(returns)
    equity = np.array(equity)

    # Compute metrics
    profit_pct = (info.get("total_profit", 1.0) - 1.0) * 100.0
    max_dd = compute_max_drawdown(equity)
    sharpe = compute_sharpe_ratio(returns)
    sortino = compute_sortino_ratio(returns)
    calmar = compute_calmar_ratio(returns, equity)
    saturation = np.mean(np.abs(actions) > 0.95) * 100.0
    avg_pos = np.mean(positions)
    stops = info.get("stop_loss_events", 0)

    return {
        "profit": profit_pct,
        "max_dd": max_dd * 100.0,  # in %
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "saturation": saturation,
        "avg_pos": avg_pos,
        "stops": stops,
        "equity_curve": equity,
        "actions": actions,
    }


def main():
    print("Configuring matplotlib and loading models...")
    configure_matplotlib()

    data_path = ROOT / "nasdaq100.xlsx"
    sac_path = ROOT / "outputs" / "sac.zip"
    ppo_path = ROOT / "outputs" / "ppo.zip"
    dt_ckpt_path = ROOT / "outputs" / "dt" / "dt_model.pt"
    output_dir = ROOT / "outputs"

    df = load_market_data(data_path)
    _, test_bound = split_frame_bounds(df, window_size=30, test_start_date="2024-01-01")

    # Env windowed (PPO/SAC)
    env_window = PositionMgmtEnv(
        df=df,
        window_size=30,
        frame_bound=test_bound,
        risk_config=RiskConfig(flat_obs=False, stop_loss_pct=0.05, max_drawdown_pct=0.15, use_trend_features=True),
    )

    # Env flat (DT)
    env_flat = PositionMgmtEnv(
        df=df,
        window_size=30,
        frame_bound=test_bound,
        risk_config=RiskConfig(flat_obs=True, stop_loss_pct=0.05, max_drawdown_pct=0.15, use_trend_features=True),
    )

    # Load models
    sac_model = SAC.load(sac_path, env=env_window) if sac_path.exists() else None
    ppo_model = PPO.load(ppo_path, env=env_window) if ppo_path.exists() else None

    checkpoint = torch.load(dt_ckpt_path, map_location="cpu", weights_only=False)
    dt_model = DecisionTransformer(
        state_dim=checkpoint["state_dim"],
        action_dim=1,
        context_length=checkpoint["context_length"],
        max_episode_len=checkpoint.get("max_episode_len", checkpoint["state_dict"]["embed_timestep.weight"].shape[0]),
    )
    dt_model.load_state_dict(checkpoint["state_dict"])
    
    dt_agent = DTAgent(
        model=dt_model,
        target_rtg=0.3,
        state_mean=checkpoint["state_mean"],
        state_std=checkpoint["state_std"],
        rtg_scale=checkpoint["rtg_scale"],
        device="cpu",
    )

    # 1. Evaluate baselines
    print("\nEvaluating baselines...")
    baseline_results = {}
    
    print("  Random baseline...")
    baseline_results["Random"] = run_one_env_episode(None, env_window, seed=42)
    
    if sac_model:
        print("  SAC baseline...")
        baseline_results["SAC"] = run_one_env_episode(sac_model, env_window, seed=42)
    else:
        baseline_results["SAC"] = None

    if ppo_model:
        print("  PPO baseline...")
        baseline_results["PPO"] = run_one_env_episode(ppo_model, env_window, seed=42)
    else:
        baseline_results["PPO"] = None

    # 2. Sweep Target RTGs for DT
    test_rtgs = [-0.5, 0.0, 0.05, 0.3, 0.8, 1.5, 3.0, 5.0]
    dt_results = []
    dt_actions = {}

    print("\nSweeping Decision Transformer targets...")
    for rtg in test_rtgs:
        print(f"  Target RTG = {rtg:.2f}...")
        dt_agent.target_rtg = rtg
        res = run_one_env_episode(dt_agent, env_flat, seed=42)
        res["rtg"] = rtg
        dt_results.append(res)
        dt_actions[rtg] = res["actions"]

    # 3. Print Summary Metrics Table
    print("\n" + "="*80)
    print("SUMMARY PERFORMANCE METRICS TABLE (Markdown)")
    print("="*80)
    print("| Strategy / Target RTG | Profit (%) | Sharpe | Sortino | Max DD (%) | Calmar | Stops | Avg Pos | Saturation (%) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for name in ["Random", "SAC", "PPO"]:
        stats = baseline_results[name]
        if stats:
            print(f"| **{name} Baseline** | {stats['profit']:+.2f}% | {stats['sharpe']:.3f} | {stats['sortino']:.3f} | {stats['max_dd']:.2f}% | {stats['calmar']:.3f} | {stats['stops']} | {stats['avg_pos']:.3f} | {stats['saturation']:.1f}% |")
            
    for res in dt_results:
        rtg_lbl = f"DT (RTG = {res['rtg']:.2f})"
        if res["rtg"] == 0.8:
            rtg_lbl += " 🏆"
        elif res["rtg"] == 5.0:
            rtg_lbl += " ❌"
        print(f"| {rtg_lbl} | {res['profit']:+.2f}% | {res['sharpe']:.3f} | {res['sortino']:.3f} | {res['max_dd']:.2f}% | {res['calmar']:.3f} | {res['stops']} | {res['avg_pos']:.3f} | {res['saturation']:.1f}% |")
    print("="*80 + "\n")

    # 4. Generate dt_p0_performance.png (3-panel)
    print("Generating P0 Performance layout...")
    fig1, axes1 = plt.subplots(1, 3, figsize=(24, 7))

    # Panel 1: Equity Curve overlay
    ax_eq = axes1[0]
    dates = df.index[test_bound[0] : test_bound[0] + len(baseline_results["Random"]["equity_curve"])]
    
    ax_eq.plot(dates, baseline_results["Random"]["equity_curve"] / 1e6, label="Random Baseline", color="gray", alpha=0.5, linestyle="--")
    if baseline_results["SAC"]:
        ax_eq.plot(dates, baseline_results["SAC"]["equity_curve"] / 1e6, label="SAC Baseline", color="#ff7f0e", alpha=0.7)
    if baseline_results["PPO"]:
        ax_eq.plot(dates, baseline_results["PPO"]["equity_curve"] / 1e6, label="PPO Baseline", color="#2ca02c", alpha=0.7)
        
    # Overplot specific DT curves
    dt_by_rtg = {res["rtg"]: res for res in dt_results}
    ax_eq.plot(dates, dt_by_rtg[-0.5]["equity_curve"] / 1e6, label="DT (RTG = -0.50)", color="#7f7f7f", linewidth=1.5)
    ax_eq.plot(dates, dt_by_rtg[0.8]["equity_curve"] / 1e6, label="DT (RTG = 0.80) 🏆", color="#1f77b4", linewidth=2.8)
    ax_eq.plot(dates, dt_by_rtg[3.0]["equity_curve"] / 1e6, label="DT (RTG = 3.00)", color="#9467bd", linewidth=1.8)
    ax_eq.plot(dates, dt_by_rtg[5.0]["equity_curve"] / 1e6, label="DT (RTG = 5.00)", color="#d62728", linewidth=2.8)

    ax_eq.set_title("Equity Curve Comparison", fontsize=13, fontweight="bold")
    ax_eq.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax_eq.set_ylabel("Portfolio Equity ($ Millions)", fontsize=11, fontweight="bold")
    ax_eq.grid(True, alpha=0.3)
    ax_eq.legend(fontsize=9, loc="upper left")

    # Panel 2: Target Calibration Curve
    plot_rtg_calibration(dt_results, ax=axes1[1])

    # Panel 3: Pareto Frontier
    plot_pareto_frontier(dt_results, baseline_results, ax=axes1[2])

    plt.tight_layout()
    fig1.savefig(output_dir / "dt_p0_performance.png", dpi=160, bbox_inches="tight")
    print("Saved outputs/dt_p0_performance.png")

    # 5. Generate dt_p0_analysis.png (2-panel)
    print("Generating P0 Analysis layout...")
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: RTG Sensitivity
    ax_sens = axes2[0]
    sorted_res = sorted(dt_results, key=lambda r: r["rtg"])
    rtg_xs = [r["rtg"] for r in sorted_res]
    prof_ys = [r["profit"] for r in sorted_res]
    sharpe_ys = [r["sharpe"] for r in sorted_res]

    color_prof = "#1f77b4"
    ax_sens.set_xlabel("Target RTG (Goal)", fontsize=11, fontweight="bold")
    ax_sens.set_ylabel("Actual Profit (%)", color=color_prof, fontsize=11, fontweight="bold")
    ax_sens.plot(rtg_xs, prof_ys, color=color_prof, marker="o", linewidth=2.2, label="Profit")
    ax_sens.tick_params(axis="y", labelcolor=color_prof)
    ax_sens.grid(True, alpha=0.3)

    ax_sens_twin = ax_sens.twinx()
    color_sh = "#2ca02c"
    ax_sens_twin.set_ylabel("Annualized Sharpe Ratio", color=color_sh, fontsize=11, fontweight="bold")
    ax_sens_twin.plot(rtg_xs, sharpe_ys, color=color_sh, marker="s", linewidth=2.2, linestyle="--", label="Sharpe")
    ax_sens_twin.tick_params(axis="y", labelcolor=color_sh)
    
    ax_sens.set_title("Target RTG vs. Realized Return & Sharpe", fontsize=13, fontweight="bold")

    # Panel 2: Action Saturation Distribution
    plot_action_saturation_distribution(dt_actions, ax=axes2[1])

    plt.tight_layout()
    fig2.savefig(output_dir / "dt_p0_analysis.png", dpi=160, bbox_inches="tight")
    print("Saved outputs/dt_p0_analysis.png")


if __name__ == "__main__":
    main()
