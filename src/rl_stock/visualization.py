"""Plotting utilities for strategy results and trades."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl_stock.envs_v2 import PositionMgmtEnv
from rl_stock.risk import (
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
)


def configure_matplotlib() -> None:
    """Configure fonts for plots generated from local scripts or notebooks."""
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_strategy_returns(results: dict, save_path: str | Path | None = None, show: bool = False):
    """Plot return comparison and improvement over the random baseline."""
    strategies = list(results.keys())
    return_rates = [(np.mean(results[key]["profits"]) - 1) * 100 for key in strategies]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#F4A261", "#9B5DE5"]
    x = np.arange(len(strategies))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    bars1 = ax1.bar(x, return_rates, color=colors[: len(strategies)], edgecolor="black", linewidth=1.5)
    max_r = max(return_rates)
    top_pad = max(2.0, max_r * 0.15)
    ax1.set_ylim(min(0, min(return_rates) - 2), max_r + top_pad)
    y_range = ax1.get_ylim()[1] - ax1.get_ylim()[0]
    label_offset = y_range * 0.03

    for bar, rate in zip(bars1, return_rates):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + label_offset,
            f"{rate:+.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.set_ylabel("Return (%)")
    ax1.set_title("Strategy Return Comparison", pad=14, fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    baseline = return_rates[0]
    improvements = [rate - baseline for rate in return_rates]
    improvements[0] = 0
    bars2 = ax2.bar(
        x,
        improvements,
        color=["#CCCCCC"] + list(colors[1 : len(strategies)]),
        edgecolor="black",
        linewidth=1.5,
    )
    max_imp = max(improvements) if improvements else 0
    top_pad2 = max(1.5, max_imp * 0.15)
    ax2.set_ylim(min(-2, min(improvements) - 1), max_imp + top_pad2)
    y_range2 = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    label_offset2 = y_range2 * 0.03

    for bar, improvement in zip(bars2, improvements):
        height = bar.get_height()
        label = f"{improvement:+.2f}%" if improvement > 0 else "Baseline"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + label_offset2,
            label,
            ha="center",
            va="bottom",
            fontsize=12 if improvement > 0 else 11,
            fontweight="bold" if improvement > 0 else "normal",
            color="black" if improvement > 0 else "gray",
        )

    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="Random baseline")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies)
    ax2.set_ylabel("Improvement vs Random (%)")
    ax2.set_title("Performance Gain vs Random", pad=14, fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _run_one_episode(model, env, seed: int):
    obs, _ = env.reset(seed=seed)
    if model is None:
        env.action_space.seed(seed)
    done = False
    info: dict = {}
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def visualize_trading(
    model,
    env,
    model_name: str,
    seed: int = 42,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Run one episode and plot trading behaviour.

    Routes to the position-aware dashboard for ``PositionMgmtEnv`` and falls
    back to the legacy buy/sell scatter plot for the T+1 discrete env.
    """
    if isinstance(env, PositionMgmtEnv):
        info = _run_one_episode(model, env, seed)
        fig = plot_risk_dashboard(env, info, model_name)
        _print_position_summary(model_name, info)
        if save_path:
            fig.savefig(save_path, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        return info, fig

    info = _run_one_episode(model, env, seed)
    fig = env.render_with_trades(title=f"{model_name} T+1")

    print(f"\n{model_name} Trading Statistics:")
    print(f"  Number of Buys: {len(info.get('buy_ticks', []))}")
    print(f"  Number of Sells: {len(info.get('sell_ticks', []))}")
    print(
        f"  Final Profit: {info['total_profit']:.4f} "
        f"({(info['total_profit'] - 1) * 100:.2f}%)"
    )

    buy_ticks = info.get("buy_ticks", [])
    sell_ticks = info.get("sell_ticks", [])
    if buy_ticks and sell_ticks:
        print("\n  Trade Details:")
        for i, (buy_tick, sell_tick) in enumerate(zip(buy_ticks, sell_ticks)):
            buy_price = env.prices[buy_tick]
            sell_price = env.prices[sell_tick]
            pnl = (sell_price - buy_price) / buy_price * 100
            hold_days = sell_tick - buy_tick
            print(
                f"    Trade {i + 1}: Buy@{buy_price:.0f} -> Sell@{sell_price:.0f}, "
                f"P/L {pnl:+.2f}%, held {hold_days} days"
            )

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return info, fig


def _print_position_summary(model_name: str, info: dict) -> None:
    equity = np.asarray(info.get("equity_curve", []), dtype=float)
    returns = np.asarray(info.get("returns", []), dtype=float)
    positions = np.asarray(info.get("position_history", []), dtype=float)
    sharpe = compute_sharpe_ratio(returns) if returns.size > 1 else 0.0
    sortino = compute_sortino_ratio(returns) if returns.size > 1 else 0.0
    max_dd = compute_max_drawdown(equity) if equity.size > 1 else 0.0
    calmar = compute_calmar_ratio(returns, equity) if returns.size > 0 and equity.size > 0 else 0.0

    print(f"\n{model_name} Position Statistics:")
    print(f"  Final Equity: {equity[-1] if equity.size else 0:.2f}")
    print(f"  Total Profit: {info.get('total_profit', 1.0):.4f}")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Sortino Ratio: {sortino:.3f}")
    print(f"  Max Drawdown: {max_dd * 100:.2f}%")
    print(f"  Calmar Ratio: {calmar:.3f}")
    print(f"  Stop-loss events: {info.get('stop_loss_events', 0)}")
    if positions.size > 1:
        print(f"  Avg Position: {np.mean(positions):.3f}")
        print(f"  Position StdDev: {np.std(positions):.3f}")


def plot_position_history(
    env: PositionMgmtEnv,
    info: dict,
    title: str | None = None,
    ax=None,
):
    """Plot the target position ratio overlaid on price."""
    positions = np.asarray(info.get("position_history", []), dtype=float)
    if positions.size == 0:
        raise ValueError("info has no position_history; pass a PositionMgmtEnv info dict.")

    prices = env.prices[env.window_size : env.window_size + positions.size]
    n = min(prices.size, positions.size)
    prices = prices[:n]
    positions = positions[:n]

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.figure

    color_price = "tab:blue"
    color_pos = "tab:orange"
    ax.plot(prices, color=color_price, label="Price", alpha=0.7)
    ax.set_ylabel("Price", color=color_price)
    ax.tick_params(axis="y", labelcolor=color_price)

    ax2 = ax.twinx()
    ax2.plot(positions, color=color_pos, label="Target Position", linewidth=1.5)
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax2.set_ylabel("Position Ratio", color=color_pos)
    ax2.tick_params(axis="y", labelcolor=color_pos)
    ax2.set_ylim(-1.1, 1.1)

    ax.set_title(title or "Position Ratio vs Price", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    return fig


def plot_equity_curve(
    info: dict,
    initial_cash: float,
    title: str | None = None,
    ax=None,
):
    """Plot the equity curve with drawdown shading."""
    equity = np.asarray(info.get("equity_curve", []), dtype=float)
    if equity.size == 0:
        raise ValueError("info has no equity_curve.")

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / np.maximum(running_max, 1e-8)

    if ax is None:
        fig, (ax, ax_dd) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig = ax.figure
        ax_dd = None

    ax.plot(equity, color="tab:green", linewidth=1.5, label="Equity")
    ax.axhline(initial_cash, color="gray", linestyle="--", alpha=0.5, label="Initial")
    ax.fill_between(
        np.arange(equity.size), running_max, equity, color="red", alpha=0.15, label="Drawdown"
    )
    ax.set_ylabel("Equity")
    ax.set_title(title or "Equity Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    if ax_dd is not None:
        ax_dd.fill_between(np.arange(drawdown.size), drawdown * 100, 0, color="red", alpha=0.4)
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.set_xlabel("Step")
        ax_dd.grid(True, alpha=0.3)

    return fig


def plot_risk_dashboard(env: PositionMgmtEnv, info: dict, model_name: str):
    """Composite figure: price+position on top, equity+drawdown below."""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1], hspace=0.35)

    ax_pos = fig.add_subplot(gs[0])
    plot_position_history(env, info, title=f"{model_name}: Position vs Price", ax=ax_pos)

    ax_eq = fig.add_subplot(gs[1])
    equity = np.asarray(info.get("equity_curve", []), dtype=float)
    running_max = np.maximum.accumulate(equity) if equity.size else equity
    ax_eq.plot(equity, color="tab:green", linewidth=1.5, label="Equity")
    ax_eq.axhline(env.risk.initial_cash, color="gray", linestyle="--", alpha=0.5, label="Initial")
    if equity.size:
        ax_eq.fill_between(
            np.arange(equity.size), running_max, equity, color="red", alpha=0.15, label="Drawdown"
        )
    ax_eq.set_ylabel("Equity")
    ax_eq.set_title(f"{model_name}: Equity Curve", fontsize=12, fontweight="bold")
    ax_eq.legend(loc="upper left")
    ax_eq.grid(True, alpha=0.3)

    ax_dd = fig.add_subplot(gs[2], sharex=ax_eq)
    if equity.size:
        drawdown = (equity - running_max) / np.maximum(running_max, 1e-8) * 100
        ax_dd.fill_between(np.arange(drawdown.size), drawdown, 0, color="red", alpha=0.4)
    ax_dd.set_ylabel("DD (%)")
    ax_dd.set_xlabel("Step")
    ax_dd.grid(True, alpha=0.3)

    returns = np.asarray(info.get("returns", []), dtype=float)
    sharpe = compute_sharpe_ratio(returns) if returns.size > 1 else 0.0
    max_dd = compute_max_drawdown(equity) if equity.size > 1 else 0.0
    total_profit = info.get("total_profit", 1.0)
    fig.suptitle(
        f"{model_name} -- Profit {(total_profit - 1) * 100:+.2f}%  "
        f"Sharpe {sharpe:.2f}  MaxDD {max_dd * 100:.2f}%  "
        f"Stops {info.get('stop_loss_events', 0)}",
        fontsize=14,
        fontweight="bold",
    )
    return fig


def plot_pareto_frontier(
    dt_results: list[dict],
    baseline_results: dict[str, dict],
    ax=None,
):
    """Plot the Return vs Max Drawdown Pareto Frontier.

    Highlights dominated points and shows PPO/SAC/Random baseline benchmarks.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # Extract DT points
    x_vals = np.array([r["max_dd"] for r in dt_results])
    y_vals = np.array([r["profit"] for r in dt_results])
    sats = np.array([r["saturation"] for r in dt_results])
    rtgs = np.array([r["rtg"] for r in dt_results])

    # Find Pareto optimal points (Minimize X (Drawdown), Maximize Y (Profit))
    # Sort by X ascending, then by Y descending
    sorted_pts = sorted(zip(x_vals, y_vals, sats, rtgs), key=lambda p: (p[0], -p[1]))
    
    pareto_pts = []
    max_y = -np.inf
    for pt in sorted_pts:
        x, y, sat, rtg = pt
        if y > max_y:
            pareto_pts.append((x, y, sat, rtg))
            max_y = y

    pareto_x = np.array([p[0] for p in pareto_pts])
    pareto_y = np.array([p[1] for p in pareto_pts])

    # Plot Pareto Frontier line
    ax.plot(pareto_x, pareto_y, color="#1f77b4", linestyle="-", linewidth=2.5, zorder=2, label="Pareto Frontier")
    
    # Scatter plot all DT points colored by their action saturation rate
    sc = ax.scatter(
        x_vals,
        y_vals,
        c=sats,
        cmap="YlOrRd",
        edgecolors="black",
        s=120,
        linewidths=1.2,
        zorder=3,
        vmin=0,
        vmax=20,
    )
    
    # Add colorbar for action saturation
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Action Saturation Rate (%)", fontsize=11, fontweight="bold")

    # Annotate DT points with their target RTGs
    for x, y, sat, rtg in zip(x_vals, y_vals, sats, rtgs):
        offset = (8, -5) if rtg != 0.8 else (-35, 10)
        ax.annotate(
            f"RTG={rtg:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=offset,
            fontsize=9,
            fontweight="bold" if rtg == 0.8 else "normal",
        )

    # Plot Baselines (PPO, SAC, Random)
    markers = {"Random": "o", "PPO": "*", "SAC": "D"}
    colors = {"Random": "gray", "PPO": "#2ca02c", "SAC": "#ff7f0e"}
    
    for name, stats in baseline_results.items():
        if stats:
            ax.scatter(
                stats["max_dd"],
                stats["profit"],
                marker=markers.get(name, "X"),
                color=colors.get(name, "purple"),
                edgecolors="black",
                s=180,
                linewidths=1.5,
                zorder=4,
                label=f"{name} Baseline",
            )
            ax.annotate(
                name,
                (stats["max_dd"], stats["profit"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=10,
                fontweight="bold",
            )

    # Highlight dominated zone
    # Use fill_between to shade the dominated area below the frontier
    if len(pareto_x) > 0:
        # Extend Pareto line to the right edge of plot to cover the dominated area
        ext_x = np.append(pareto_x, [max(x_vals.max() + 1, 18)])
        ext_y = np.append(pareto_y, [pareto_y[-1]])
        ax.fill_between(
            ext_x,
            ext_y,
            ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -20,
            color="gray",
            alpha=0.08,
            zorder=1,
            label="Dominated Zone",
        )

    ax.set_title("Return vs. Drawdown Pareto Frontier", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Maximum Drawdown (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Profit (%)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower left")
    return fig


def plot_rtg_calibration(
    dt_results: list[dict],
    ax=None,
):
    """Plot target RTG vs actual realized return."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    target_rtgs = np.array([r["rtg"] for r in dt_results])
    realized_profits = np.array([r["profit"] for r in dt_results])

    # Sort for plotting line
    sort_idx = np.argsort(target_rtgs)
    xs = target_rtgs[sort_idx]
    ys = realized_profits[sort_idx]

    ax.plot(xs, ys, marker="o", color="#1f77b4", linewidth=2.5, label="DT Realized Performance")
    
    # Draw target line y = x * 100 (representing 100% target match)
    # Target RTG is cumulative return of episode (e.g. 0.3 = 30%)
    ref_xs = np.linspace(min(xs), max(xs), 100)
    ref_ys = ref_xs * 100.0
    ax.plot(ref_xs, ref_ys, color="gray", linestyle="--", alpha=0.6, label="Perfect Alignment (y = x)")

    # Shading areas
    ax.fill_between(xs, ys, xs * 100, where=(ys > xs * 100), color="green", alpha=0.08, label="Outperformance")
    ax.fill_between(xs, ys, xs * 100, where=(ys < xs * 100), color="red", alpha=0.08, label="Underperformance")

    ax.set_title("Target RTG Calibration Curve", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Commanded Target RTG (Goal)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Cumulative Return (%)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")
    return fig


def plot_action_saturation_distribution(
    results_actions: dict[float, np.ndarray],
    ax=None,
):
    """Plot action distribution Box plot for each RTG."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    sorted_rtgs = sorted(results_actions.keys())
    data = [results_actions[rtg] for rtg in sorted_rtgs]
    labels = [f"{rtg:.2f}" for rtg in sorted_rtgs]

    # Plot boxplot
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"facecolor": "#e74c3c", "edgecolor": "black", "alpha": 0.7},
        whiskerprops={"color": "gray", "linewidth": 1.2},
        capprops={"color": "gray", "linewidth": 1.2},
    )

    # Alternate colors of boxes slightly based on RTG to make it pretty
    colors = ["#3498db" if float(label) < 0.8 else "#2ecc71" if float(label) <= 1.5 else "#e74c3c" for label in labels]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    # Highlight saturation boundaries at +0.95 and -0.95
    ax.axhline(0.95, color="red", linestyle=":", alpha=0.7, linewidth=1.5)
    ax.axhline(-0.95, color="red", linestyle=":", alpha=0.7, linewidth=1.5)
    ax.text(len(sorted_rtgs) - 0.3, 0.98, "+0.95 Threshold", color="red", fontsize=8, ha="right")
    ax.text(len(sorted_rtgs) - 0.3, -0.92, "-0.95 Threshold", color="red", fontsize=8, ha="right")

    ax.set_title("Action Distributions & Saturation Limits", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Target RTG (Commanded Goal)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Action Outputs (Target Positions)", fontsize=11, fontweight="bold")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    return fig

