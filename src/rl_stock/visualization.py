"""Plotting utilities for strategy results and trades."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def configure_matplotlib() -> None:
    """Configure fonts for plots generated from local scripts or notebooks."""
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_strategy_returns(results: dict, save_path: str | Path | None = None, show: bool = False):
    """Plot return comparison and improvement over the random baseline."""
    strategies = list(results.keys())
    return_rates = [(np.mean(results[key]["profits"]) - 1) * 100 for key in strategies]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
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
    arrow_drop = y_range * 0.06
    if len(return_rates) > 1:
        ax1.annotate(
            "",
            xy=(x[1], return_rates[1] - arrow_drop),
            xytext=(x[0], return_rates[0]),
            arrowprops={"arrowstyle": "->", "color": "green", "lw": 2},
        )
    if len(return_rates) > 2:
        ax1.annotate(
            "",
            xy=(x[2], return_rates[2] - arrow_drop),
            xytext=(x[0], return_rates[0]),
            arrowprops={"arrowstyle": "->", "color": "green", "lw": 2},
        )

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
        color=["#CCCCCC", "#4ECDC4", "#45B7D1"][: len(strategies)],
        edgecolor="black",
        linewidth=1.5,
    )
    max_imp = max(improvements)
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

    if baseline != 0:
        for i in range(1, len(strategies)):
            ratio = return_rates[i] / baseline
            ax2.text(
                x[i],
                improvements[i] * 0.5,
                f"{ratio:.1f}x",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white",
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def visualize_trading(
    model,
    env,
    model_name: str,
    seed: int = 42,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Run one episode and plot buy/sell points."""
    obs, _ = env.reset(seed=seed)
    if model is None:
        env.action_space.seed(seed)

    done = False
    info = {}
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    fig = env.render_with_trades(title=f"{model_name} T+1")

    print(f"\n{model_name} Trading Statistics:")
    print(f"  Number of Buys: {len(info['buy_ticks'])}")
    print(f"  Number of Sells: {len(info['sell_ticks'])}")
    print(f"  Final Profit: {info['total_profit']:.4f} ({(info['total_profit'] - 1) * 100:.2f}%)")

    if info["buy_ticks"] and info["sell_ticks"]:
        print("\n  Trade Details:")
        for i, (buy_tick, sell_tick) in enumerate(zip(info["buy_ticks"], info["sell_ticks"])):
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
